import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from terratorch.datasets.utils import generate_bands_intervals
from torchgeo.datasets.geo import NonGeoDataset

from geoid_flood.io import read_raster

log = logging.getLogger(__name__)

# Modalities that have temporal dimension (pre/post); static modalities (e.g. dem) are not filtered by image_scope
TIME_VARYING_MODALITIES = {"s1grd", "s1rtc", "s2l2a"}

MODALITY_CHANNEL_COUNTS = {"s1grd": 2, "s1rtc": 2, "s2l2a": 12, "dem": 1}

# On-disk S2 L2A band order: B01-B08, B8A, B09, B11, B12 (also the GeoTIFF band names)
S2L2A_BAND_SELECT_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # B02..B08, B8A, B11, B12 (skip B01, B09)

# Channels per modality after band selection (e.g. s2l2a -> 10 TerraMind bands)
MODALITY_OUTPUT_CHANNELS = {
    "s1grd": 2,
    "s1rtc": 2,
    "s2l2a": len(S2L2A_BAND_SELECT_INDICES),
    "dem": 1,
}

# When return_image_as_dict=True, keys use these names to match backbone_modalities (e.g. TerraMind expects "sen1grd")
BACKBONE_MODALITY_KEYS = {"s1grd": "sen1grd", "s1rtc": "sen1grd", "s2l2a": "sen2l2a", "dem": "dem"}


def _event_folder_from_label_id(label_id: str) -> str:
    """Derive event folder"""
    chip_prefix = label_id.replace("_label", "")
    return chip_prefix.rsplit("-", 1)[0]


# Placeholder date when tile_id has no trailing YYYYMMDDTHHMMSS (e.g. mid-year 2024)
_PLACEHOLDER_ACQUISITION_DATE = pd.Timestamp("2024-06-15")


def _acquisition_date_from_tile_id(tile_id: str) -> pd.Timestamp:
    """Parse acquisition date from tile_id.

    Expected suffix: _YYYYMMDDTHHMMSS (e.g. EMSR696-3-2_s1grd_post_20230913T043337).
    If the suffix is missing or invalid, returns a placeholder date.
    """
    # Match last segment like 20230913T043337
    match = re.search(r"_(\d{8}T\d{6})$", str(tile_id).strip())
    if match:
        try:
            return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S")
        except (ValueError, TypeError):
            pass
    return _PLACEHOLDER_ACQUISITION_DATE


class GEOIDFloodDataset(NonGeoDataset):
    """Dataset for the GEOID-Flood dataset with CSV-driven metadata.

    Reads from csv. Each row = one tile. Supports
    image_scope (pre/post), label remapping for simple training.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modalities: list[str] | None = None,
        modalities_pre: list[str] | None = None,
        modalities_post: list[str] | None = None,
        image_scope: list[str] | None = None,
        metadata_filename: str = "data_tiles_s256_st128.csv",
        label_folder: str = "label",
        max_cloud_cover: float = 1.0,
        min_positive: float = 0.0,
        min_valid_proportion: float = 0.0,
        no_data_replace: float = 0.0,
        no_label_replace: int = 255,
        transform=None,
        output_bands: list[str] | None = None,
        rgb_indices: list[int] | None = None,
        return_timestamps: bool = False,
        return_image_as_dict: bool = False,
        paired: bool = False,
        fuse_paired: bool = False,
        skip_bad_samples: bool = True,
        max_skip_retries: int = 16,
        **kwargs,
    ):
        """Initialize GEOIDFloodDataset.

        Args:
            data_root: Root directory of data/geoid-flood.
            split: Data split (train, val, test).
            modalities: Modalities to load for both pre and post when modalities_pre/post are unset.
            modalities_pre: Modalities for pre images in paired mode (e.g. ["s2l2a"]).
            modalities_post: Modalities for post images in paired mode (e.g. ["s1grd"]).
            image_scope: Temporal scope - ["pre"], ["post"], or ["pre", "post"].
            metadata_filename: CSV filename in data_root.
            label_folder: Name of label subfolder.
            max_cloud_cover: Maximum allowed cloud cover.
            min_positive: Minimum positive_proportion.
            min_valid_proportion: Minimum valid_proportion.
            no_data_replace: Value to replace no-data pixels.
            no_label_replace: Value to replace no-label pixels (ignore_index).
            transform: Albumentations transform pipeline.
            output_bands: Bands to output (e.g. ["VV", "VH"] for S1).
            rgb_indices: Indices for RGB plotting.
            return_timestamps: If True, add "timestamps" to each sample (e.g. for OlmoEarth backbone). Default False.
            return_image_as_dict: If True, return "image" as a dict {modality: tensor} for each modality (e.g. for
                TerraMind). If False, return a single concatenated tensor (channels in modality order). Default False.
            fuse_paired: If True (requires paired=True), early-fusion mode: concatenate pre and post images along the
                channel dimension into a single "image" tensor and return the raw three-class mask (0=background,
                1=permanent water, 2=flood) without binary remapping. A single shared transform keeps pre/post
                co-registered. Default False.
        """
        super().__init__(**kwargs)
        self.data_root = Path(data_root)
        self.return_timestamps = return_timestamps
        self.return_image_as_dict = return_image_as_dict
        self.paired = paired
        self.fuse_paired = fuse_paired
        self.skip_bad_samples = skip_bad_samples
        self.max_skip_retries = max(0, int(max_skip_retries))
        assert self.data_root.exists(), f"data_root {self.data_root} does not exist"
        assert split in ("train", "val", "test", "all"), f"split must be train/val/test/all, got {split}"
        self.split = split
        base_modalities = modalities or ["s1grd"]
        if modalities_pre is not None or modalities_post is not None:
            self.modalities_pre = list(modalities_pre if modalities_pre is not None else base_modalities)
            self.modalities_post = list(modalities_post if modalities_post is not None else base_modalities)
            self.all_modalities = list(dict.fromkeys(self.modalities_pre + self.modalities_post))
            self.modalities = list(modalities) if modalities is not None else self.all_modalities
        else:
            self.modalities = list(base_modalities)
            self.modalities_pre = list(self.modalities)
            self.modalities_post = list(self.modalities)
            self.all_modalities = list(self.modalities)
        self.image_scope = image_scope if image_scope is not None else ["pre", "post"]
        self.label_folder = label_folder
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.transform = transform
        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices

        self.output_bands = generate_bands_intervals(output_bands) if output_bands else None
        if self.output_bands:
            self.filter_indices = list(range(len(self.output_bands)))
        else:
            self.filter_indices = None

        metadata_path = self.data_root / metadata_filename
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        if self.split != "all":
            metadata = metadata[metadata["split"] == self.split]
        else:
            metadata = metadata[metadata["split"].isin(["train", "val", "test"])]
        print(f"GEOIDFloodDataset {self.split}: after split filter: {len(metadata)} rows", flush=True)

        metadata = metadata[metadata["cloud_cover"] <= max_cloud_cover]
        print(f"  after cloud_cover <= {max_cloud_cover}: {len(metadata)} rows", flush=True)

        metadata = metadata[metadata["positive_proportion"] >= min_positive]
        print(f"  after min_positive >= {min_positive}: {len(metadata)} rows", flush=True)

        if "valid_proportion" in metadata.columns and min_valid_proportion > 0:
            metadata = metadata[metadata["valid_proportion"] >= min_valid_proportion]
            print(f"  after min_valid_proportion >= {min_valid_proportion}: {len(metadata)} rows", flush=True)

        metadata = metadata[metadata["modality"].isin(self.all_modalities)]
        print(f"  after modality in {self.all_modalities}: {len(metadata)} rows", flush=True)

        # Apply image_scope only to time-varying modalities (s1grd, s2l2a); static (e.g. dem) are kept as-is
        in_scope = (
            metadata["modality"].isin(TIME_VARYING_MODALITIES) & metadata["image_time"].isin(self.image_scope)
        ) | (~metadata["modality"].isin(TIME_VARYING_MODALITIES))
        metadata = metadata[in_scope]
        n_pre = int((metadata["image_time"] == "pre").sum())
        n_post = int((metadata["image_time"] == "post").sum())
        print(
            f"  after image_scope (time-varying only) in {self.image_scope}: {len(metadata)} rows (pre: {n_pre}, post: {n_post})",
            flush=True,
        )

        metadata = metadata.reset_index(drop=True)

        if not self.paired:
            if len(self.modalities) == 1:
                self.rows = metadata
                self.samples = None
                n_pre_final = int((self.rows["image_time"] == "pre").sum())
                n_post_final = int((self.rows["image_time"] == "post").sum())
                print(
                    f"GEOIDFloodDataset {self.split}: total {len(self.rows)} tiles (pre: {n_pre_final}, post: {n_post_final})",
                    flush=True,
                )
            else:
                # Build one sample per (label_id, image_time, x, y, size) with a row for every modality
                time_varying = [m for m in self.modalities if m in TIME_VARYING_MODALITIES]
                static = [m for m in self.modalities if m not in TIME_VARYING_MODALITIES]
                # Index time-varying rows by (label_id, image_time, x, y, size) -> {modality: row}
                tv_groups = {}
                for _, row in metadata.iterrows():
                    if row["modality"] in time_varying:
                        key = (row["label_id"], row["image_time"], int(row["x"]), int(row["y"]), int(row["size"]))
                        if key not in tv_groups:
                            tv_groups[key] = {}
                        tv_groups[key][row["modality"]] = row
                # Index static rows by (label_id, x, y, size) -> row
                static_index = {}
                for _, row in metadata.iterrows():
                    if row["modality"] in static:
                        key = (row["label_id"], int(row["x"]), int(row["y"]), int(row["size"]))
                        static_index[key] = row
                self.samples = []
                for (label_id, image_time, x, y, size), mod_rows in tv_groups.items():
                    if set(mod_rows.keys()) != set(time_varying):
                        continue
                    sample_rows = dict(mod_rows)
                    for mod in static:
                        row_key = (label_id, x, y, size)
                        if row_key not in static_index:
                            break
                        sample_rows[mod] = static_index[row_key]
                    else:
                        self.samples.append(
                            {
                                "label_id": label_id,
                                "image_time": image_time,
                                "x": int(x),
                                "y": int(y),
                                "size": int(size),
                                "rows": sample_rows,
                            }
                        )
                self.rows = None
                n_pre_final = sum(1 for s in self.samples if s["image_time"] == "pre")
                n_post_final = sum(1 for s in self.samples if s["image_time"] == "post")
                print(
                    f"GEOIDFloodDataset {self.split}: total {len(self.samples)} multimodal tiles (pre: {n_pre_final}, post: {n_post_final})",
                    flush=True,
                )
        else:
            # Paired mode: build one sample per (label_id, x, y, size) with pre/post modality sets
            time_varying_pre = [m for m in self.modalities_pre if m in TIME_VARYING_MODALITIES]
            time_varying_post = [m for m in self.modalities_post if m in TIME_VARYING_MODALITIES]
            static_pre = [m for m in self.modalities_pre if m not in TIME_VARYING_MODALITIES]
            static_post = [m for m in self.modalities_post if m not in TIME_VARYING_MODALITIES]
            static = list(dict.fromkeys(static_pre + static_post))

            # Index rows by (label_id, image_time, x, y, size, modality)
            per_key: dict[tuple[str, int, int, int], dict[str, dict[str, pd.Series]]] = {}
            for _, row in metadata.iterrows():
                label_id = row["label_id"]
                x, y, size = int(row["x"]), int(row["y"]), int(row["size"])
                image_time = row["image_time"]
                modality = row["modality"]
                if modality in TIME_VARYING_MODALITIES:
                    key = (label_id, x, y, size)
                    per_key.setdefault(key, {}).setdefault(image_time, {})[modality] = row

            # Index static rows by (label_id, x, y, size) -> row
            static_index = {}
            for _, row in metadata.iterrows():
                if row["modality"] in static:
                    key = (row["label_id"], int(row["x"]), int(row["y"]), int(row["size"]))
                    static_index[key] = row

            samples: list[dict] = []
            for (label_id, x, y, size), by_time in per_key.items():
                if "pre" not in by_time or "post" not in by_time:
                    continue
                pre_rows = by_time["pre"]
                post_rows = by_time["post"]
                if not set(time_varying_pre).issubset(pre_rows.keys()):
                    continue
                if not set(time_varying_post).issubset(post_rows.keys()):
                    continue
                sample_rows_pre = {m: pre_rows[m] for m in time_varying_pre}
                sample_rows_post = {m: post_rows[m] for m in time_varying_post}
                for mod in static_pre:
                    row_key = (label_id, x, y, size)
                    static_row = static_index.get(row_key)
                    if static_row is None:
                        break
                    sample_rows_pre[mod] = static_row
                else:
                    for mod in static_post:
                        row_key = (label_id, x, y, size)
                        static_row = static_index.get(row_key)
                        if static_row is None:
                            break
                        sample_rows_post[mod] = static_row
                    else:
                        samples.append(
                            {
                                "label_id": label_id,
                                "x": int(x),
                                "y": int(y),
                                "size": int(size),
                                "rows_pre": sample_rows_pre,
                                "rows_post": sample_rows_post,
                            }
                        )

            self.rows = None
            self.samples = samples
            print(
                f"GEOIDFloodDataset {self.split}: total {len(self.samples)} paired tiles "
                f"(pre: {self.modalities_pre}, post: {self.modalities_post})",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.samples) if self.samples is not None else len(self.rows)

    def _read_s1grd(self, path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
        """Read S1 GRD/RTC (linear sigma0) and convert to dB."""
        image = read_raster(path, window=window, nodata_value=self.no_data_replace)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        image = image.astype(np.float32)
        # RTC tiles can contain inf/nan or huge fill values (especially post); mask before log10.
        fill = self.no_data_replace if self.no_data_replace is not None else 0.0
        invalid = ~np.isfinite(image) | (image <= 0) | (image > 1e3)
        image[invalid] = fill
        image = np.maximum(image, np.finfo(np.float32).eps)
        image = 10 * np.log10(image)
        return image

    def _read_dem(self, path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
        """Read DEM (single-band elevation)."""
        image = read_raster(path, window=window, nodata_value=self.no_data_replace)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        return image.astype(np.float32)

    def _read_s2l2a(self, path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
        """Read S2 L2A reflectance (12 on-disk bands) and select TerraMind-compatible bands."""
        image = read_raster(path, window=window, nodata_value=self.no_data_replace)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        image = image.astype(np.float32)
        if image.shape[0] == len(S2L2A_BAND_SELECT_INDICES):
            return image
        if image.shape[0] >= max(S2L2A_BAND_SELECT_INDICES) + 1:
            return image[S2L2A_BAND_SELECT_INDICES, ...]
        raise ValueError(f"Unexpected s2l2a band count {image.shape[0]} in {path}")

    def _read_modality(self, mod: str, path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
        if mod in ("s1grd", "s1rtc"):
            return self._read_s1grd(path, window)
        if mod == "dem":
            return self._read_dem(path, window)
        if mod == "s2l2a":
            return self._read_s2l2a(path, window)
        raise NotImplementedError(f"Modality {mod} not implemented")

    def _stack_modalities(self, rows: dict[str, pd.Series], modality_list: list[str], event_folder: str, window: tuple) -> np.ndarray:
        parts = []
        for mod in modality_list:
            row = rows[mod]
            tile_id = row["tile_id"]
            image_path = self.data_root / event_folder / mod / f"{tile_id}.tif"
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            parts.append(self._read_modality(mod, image_path, window))
        image = np.concatenate(parts, axis=0)
        if self.filter_indices is not None and len(self.filter_indices) < image.shape[0]:
            image = image[self.filter_indices, ...]
        return image

    def _split_to_dict(self, img: torch.Tensor, modality_list: list[str]) -> dict[str, torch.Tensor]:
        offset = 0
        image_dict = {}
        for mod in modality_list:
            nch = MODALITY_OUTPUT_CHANNELS.get(mod, MODALITY_CHANNEL_COUNTS.get(mod, 1))
            key = BACKBONE_MODALITY_KEYS.get(mod, mod)
            image_dict[key] = img[offset : offset + nch].clone()
            offset += nch
        return image_dict

    def _remap_label(self, mask: np.ndarray, image_time: str) -> np.ndarray:
        """Remap label per simple training: pre=mask flood, post=merge 1+2."""
        out = np.full_like(mask, self.no_label_replace, dtype=np.int64)
        if image_time == "pre":
            out[mask == 0] = 0
            out[mask == 1] = 1
            out[mask == 2] = 0
            out[mask == 255] = self.no_label_replace
        else:
            out[mask == 0] = 0
            out[mask == 1] = 1
            out[mask == 2] = 1
            out[mask == 255] = self.no_label_replace
        return out

    def _is_skippable_read_error(self, exc: Exception) -> bool:
        """Return True for raster/file read exceptions that should be skipped."""
        if isinstance(exc, (rasterio.errors.RasterioError, OSError, FileNotFoundError)):
            return True
        msg = str(exc)
        return any(
            token in msg
            for token in (
                "TIFFReadEncodedTile() failed",
                "IReadBlock failed",
                "Using code not yet in table",
                "Read failed. See previous exception",
            )
        )

    def __getitem__(self, idx: int) -> dict:
        """Load sample with bounded fallback retries for transient/corrupt reads."""
        if not self.skip_bad_samples:
            return self._getitem_impl(idx)

        n = len(self)
        if n == 0:
            raise IndexError("Dataset is empty")

        last_exc: Exception | None = None
        attempts = min(self.max_skip_retries + 1, n)
        for offset in range(attempts):
            candidate_idx = (idx + offset) % n
            try:
                return self._getitem_impl(candidate_idx)
            except Exception as exc:  # pragma: no cover - defensive path for runtime data issues
                last_exc = exc
                if not self._is_skippable_read_error(exc):
                    raise
                log.warning(
                    "Skipping bad sample idx=%s (attempt=%s/%s): %s",
                    candidate_idx,
                    offset + 1,
                    attempts,
                    exc,
                )

        raise RuntimeError(
            f"Failed to load a valid sample after {attempts} attempts starting from idx={idx}. "
            f"Last error: {last_exc}"
        ) from last_exc

    def _getitem_impl(self, idx: int) -> dict:
        if not self.paired:
            # Resolve (label_id, image_time, x, y, size) and modality -> row for both single and multi modality
            if self.samples is not None:
                sample = self.samples[idx]
                label_id = sample["label_id"]
                image_time = sample["image_time"]
                x, y, size = sample["x"], sample["y"], sample["size"]
                rows = sample["rows"]
            else:
                row = self.rows.iloc[idx]
                label_id = row["label_id"]
                image_time = row["image_time"]
                x, y, size = int(row["x"]), int(row["y"]), int(row["size"])
                rows = {row["modality"]: row}

            event_folder = _event_folder_from_label_id(label_id)
            label_path = self.data_root / event_folder / self.label_folder / f"{label_id}.tif"
            if not label_path.exists():
                raise FileNotFoundError(f"Label not found: {label_path}")
            window = (x, y, size, size)

            image = self._stack_modalities(rows, self.modalities, event_folder, window)

            mask = read_raster(label_path, window=window, nodata_value=self.no_label_replace)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask = self._remap_label(mask.astype(np.int64), image_time)

            if self.transform is not None:
                image_nhwc = image.transpose(1, 2, 0)
                transformed = self.transform(image=image_nhwc, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                if not isinstance(image, torch.Tensor):
                    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask)
                mask = mask.long()
            else:
                image = torch.from_numpy(image).float()
                mask = torch.from_numpy(mask).long()

            if self.return_image_as_dict:
                image = self._split_to_dict(image, self.modalities)

            ref_mod = next((m for m in self.modalities if m in TIME_VARYING_MODALITIES), self.modalities[0])
            ref_row = rows[ref_mod]
            acquisition_date = _acquisition_date_from_tile_id(ref_row["tile_id"])

            chip_id = str(ref_row["tile_id"])
            split_origin = str(ref_row.get("split", ""))
            out = {
                "image": image,
                "mask": mask,
                "filename": str(label_path),
                "chip_id": chip_id,
                "split": split_origin,
                "image_time": str(image_time),
            }
            if self.return_timestamps:
                out["timestamps"] = torch.tensor(
                    [[acquisition_date.day, acquisition_date.month - 1, acquisition_date.year]],
                    dtype=torch.long,
                )
            return out

        # Paired mode: return pre/post images and masks in a single sample
        sample = self.samples[idx]
        label_id = sample["label_id"]
        x, y, size = sample["x"], sample["y"], sample["size"]
        rows_pre = sample["rows_pre"]
        rows_post = sample["rows_post"]

        event_folder = _event_folder_from_label_id(label_id)
        label_path = self.data_root / event_folder / self.label_folder / f"{label_id}.tif"
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")
        window = (x, y, size, size)

        image_pre_np = self._stack_modalities(rows_pre, self.modalities_pre, event_folder, window)
        image_post_np = self._stack_modalities(rows_post, self.modalities_post, event_folder, window)

        mask = read_raster(label_path, window=window, nodata_value=self.no_label_replace)
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        if self.fuse_paired:
            # Early fusion: stack pre+post channels into one image, keep the raw three-class
            # mask (0=background, 1=permanent water, 2=flood; no_label_replace=ignore), and apply
            # a single shared transform so pre/post stay co-registered with the mask.
            image_np = np.concatenate([image_pre_np, image_post_np], axis=0)
            mask3 = mask.astype(np.int64)
            if self.transform is not None:
                transformed = self.transform(image=image_np.transpose(1, 2, 0), mask=mask3)
                image = transformed["image"]
                mask_t = transformed["mask"]
                if not isinstance(image, torch.Tensor):
                    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
                if not isinstance(mask_t, torch.Tensor):
                    mask_t = torch.from_numpy(mask_t)
                mask_t = mask_t.long()
            else:
                image = torch.from_numpy(image_np).float()
                mask_t = torch.from_numpy(mask3).long()

            ref_mod_pre = next(
                (m for m in self.modalities_pre if m in TIME_VARYING_MODALITIES), self.modalities_pre[0]
            )
            ref_row_pre = rows_pre[ref_mod_pre]
            fused_out: dict[str, object] = {
                "image": image,
                "mask": mask_t,
                "filename": str(label_path),
                "chip_id": str(ref_row_pre["tile_id"]),
                "split": str(ref_row_pre.get("split", "")),
            }
            return fused_out

        mask_pre = self._remap_label(mask.astype(np.int64), "pre")
        mask_post = self._remap_label(mask.astype(np.int64), "post")

        if self.transform is not None:
            image_pre_nhwc = image_pre_np.transpose(1, 2, 0)
            image_post_nhwc = image_post_np.transpose(1, 2, 0)
            transformed_pre = self.transform(image=image_pre_nhwc, mask=mask_pre)
            transformed_post = self.transform(image=image_post_nhwc, mask=mask_post)
            image_pre = transformed_pre["image"]
            mask_pre_t = transformed_pre["mask"]
            image_post = transformed_post["image"]
            mask_post_t = transformed_post["mask"]
            if not isinstance(image_pre, torch.Tensor):
                image_pre = torch.from_numpy(np.transpose(image_pre, (2, 0, 1)))
            if not isinstance(image_post, torch.Tensor):
                image_post = torch.from_numpy(np.transpose(image_post, (2, 0, 1)))
            if not isinstance(mask_pre_t, torch.Tensor):
                mask_pre_t = torch.from_numpy(mask_pre_t)
            if not isinstance(mask_post_t, torch.Tensor):
                mask_post_t = torch.from_numpy(mask_post_t)
            mask_pre_t = mask_pre_t.long()
            mask_post_t = mask_post_t.long()
        else:
            image_pre = torch.from_numpy(image_pre_np).float()
            image_post = torch.from_numpy(image_post_np).float()
            mask_pre_t = torch.from_numpy(mask_pre).long()
            mask_post_t = torch.from_numpy(mask_post).long()

        if self.return_image_as_dict:
            image_pre = self._split_to_dict(image_pre, self.modalities_pre)
            image_post = self._split_to_dict(image_post, self.modalities_post)

        ref_mod_pre = next((m for m in self.modalities_pre if m in TIME_VARYING_MODALITIES), self.modalities_pre[0])
        ref_mod_post = next((m for m in self.modalities_post if m in TIME_VARYING_MODALITIES), self.modalities_post[0])
        ref_row_pre = rows_pre[ref_mod_pre]
        ref_row_post = rows_post[ref_mod_post]
        acquisition_date_pre = _acquisition_date_from_tile_id(ref_row_pre["tile_id"])
        acquisition_date_post = _acquisition_date_from_tile_id(ref_row_post["tile_id"])

        chip_id = str(ref_row_pre["tile_id"])
        split_origin = str(ref_row_pre.get("split", ""))
        out: dict[str, object] = {
            "image_pre": image_pre,
            "image_post": image_post,
            "mask_pre": mask_pre_t,
            "mask_post": mask_post_t,
            "filename": str(label_path),
            "chip_id": chip_id,
            "split": split_origin,
        }
        if self.return_timestamps:
            out["timestamps_pre"] = torch.tensor(
                [[acquisition_date_pre.day, acquisition_date_pre.month - 1, acquisition_date_pre.year]],
                dtype=torch.long,
            )
            out["timestamps_post"] = torch.tensor(
                [[acquisition_date_post.day, acquisition_date_post.month - 1, acquisition_date_post.year]],
                dtype=torch.long,
            )
        return out

    def _create_s1grd_composite(self, s1grd_img: np.ndarray) -> np.ndarray:
        """Create RGB composite from S1 GRD (VV, VH) with per-band min-max normalization.

        R=VV, G=VH, B=VH (or VV if single band); each channel normalized to [0, 1].
        """
        image = np.transpose(s1grd_img, (1, 2, 0))  # H, W, C
        image = np.asarray(image, dtype=np.float64)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        for c in range(image.shape[-1]):
            band = image[..., c]
            lo, hi = float(band.min()), float(band.max())
            if hi - lo > 1e-8:
                image[..., c] = (band - lo) / (hi - lo)
            else:
                image[..., c] = 0.5
        image = np.clip(image, 0, 1).astype(np.float32)
        if image.shape[-1] == 2:
            image = np.concatenate([image, image[..., :1]], axis=-1)
        return image

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        suptitle: str | None = None,
        show_axes: bool = False,
    ) -> Figure:
        """Plot a sample (S1 GRD composite + label)."""
        image = sample["image"]
        if isinstance(image, dict):
            image = torch.cat(list(image.values()), dim=0)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.ndim == 3 and image.shape[0] >= 2:
            image = self._create_s1grd_composite(image)
        else:
            image = np.transpose(image, (1, 2, 0))
            image = (image - image.min(axis=(0, 1))) / (image.max(axis=(0, 1)) - image.min(axis=(0, 1)) + 1e-8)
            image = np.clip(image, 0, 1)
            if image.shape[-1] == 2:
                image = np.concatenate([image, image[..., :1]], axis=-1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        prediction_mask = None
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        label_cmap = colors.ListedColormap(["black", "blue", "purple", "green"])
        label_norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 255.5], label_cmap.N)

        n_panels = 3 if showing_predictions else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), layout="compressed")
        if n_panels == 2:
            axes = [axes[0], axes[1]]
        axes[0].imshow(image)
        axes[0].set_title("S1 GRD (R=VV, G=VH, B=VH)")
        axes[0].axis("on" if show_axes else "off")
        axes[1].imshow(label_mask, cmap=label_cmap, norm=label_norm, interpolation="nearest")
        axes[1].set_title("Label (0=bg, 1=water)")
        axes[1].axis("on" if show_axes else "off")
        if showing_predictions:
            axes[2].imshow(prediction_mask, cmap=label_cmap, norm=label_norm, interpolation="nearest")
            axes[2].set_title("Prediction")
            axes[2].axis("on" if show_axes else "off")
        if suptitle:
            fig.suptitle(suptitle)
        return fig
