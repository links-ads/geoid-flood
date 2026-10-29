"""Mid-fusion (Siamese) encoder-decoder for paired pre/post segmentation.

Two encoder branches process the pre- and post-event images independently; their
multi-scale feature pyramids are merged per scale (default: element-wise difference
``post - pre``) before a single decoder + head produces the multi-class output.

This is the paper's *late-fusion / Siamese* change-focused architecture. It reuses
the ``fuse_paired`` data mode unchanged: the input is the channel-stacked
``[pre | post]`` image, split internally at ``pre_channels``.

Heavy terratorch model symbols are imported lazily inside methods to avoid circular
imports when this module is loaded during terratorch registry init.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


class MidFusionModel(nn.Module):
    """Two-encoder Siamese model with feature-level fusion.

    Args:
        base: A terratorch ``PixelWiseModel`` providing the pre-branch encoder + neck
            and the shared decoder + head (plus rescale / patch_size / padding).
        encoder_post: Independent encoder for the post-event branch.
        neck_post: Independent neck for the post-event branch.
        pre_channels: Number of leading channels of the fused input that belong to the
            pre-event image (the remainder are the post-event image).
        fusion: Per-scale merge op: ``"diff"`` (post - pre), ``"abs_diff"`` or
            ``"learned"`` (concat + 1x1 conv back to the original channel count).
    """

    def __init__(
        self,
        base: nn.Module,
        encoder_post: nn.Module,
        neck_post: nn.Module,
        pre_channels: int | None = None,
        fusion: str = "diff",
        pre_modalities: list | None = None,
        post_modalities: list | None = None,
    ) -> None:
        super().__init__()
        if fusion not in ("diff", "abs_diff", "learned"):
            msg = f"Unknown fusion '{fusion}'. Expected one of: diff, abs_diff, learned."
            raise ValueError(msg)
        if pre_channels is None and not (pre_modalities and post_modalities):
            msg = "Provide either pre_channels (tensor split) or both pre_modalities and post_modalities (dict split)."
            raise ValueError(msg)
        self.base = base
        self.encoder_post = encoder_post
        self.neck_post = neck_post
        self.pre_channels = int(pre_channels) if pre_channels is not None else None
        self.fusion = fusion
        # Native multi-modality: ordered [[backbone_key, n_channels], ...]. When set, the fused input
        # tensor is sliced into per-branch modality dicts so the backbone uses each modality's own
        # (pretrained) patch embedding instead of treating everything as a single stacked modality.
        self.pre_modalities = pre_modalities
        self.post_modalities = post_modalities

        if fusion == "learned":
            channel_list = self._infer_channel_list(base)
            self.merge = nn.ModuleList([nn.Conv2d(2 * c, c, kernel_size=1) for c in channel_list])

    @staticmethod
    def _infer_channel_list(base: nn.Module) -> list[int]:
        """Per-scale channel counts feeding the decoder (final neck output)."""
        neck = getattr(base, "neck", None)
        if isinstance(neck, nn.Sequential) and len(neck) > 0 and hasattr(neck[-1], "channel_list"):
            return list(neck[-1].channel_list)
        msg = "Could not infer channel_list for learned fusion from base.neck."
        raise RuntimeError(msg)

    @staticmethod
    def _slice_to_dict(x: torch.Tensor, modalities: list, start: int = 0) -> dict[str, torch.Tensor]:
        """Slice a fused tensor into {backbone_key: tensor} per the ordered modality spec."""
        image_dict: dict[str, torch.Tensor] = {}
        offset = start
        for key, nch in modalities:
            image_dict[str(key)] = x[:, offset : offset + int(nch)]
            offset += int(nch)
        return image_dict

    def _encode(self, encoder: nn.Module, neck: nn.Module, x, **kwargs) -> list[torch.Tensor]:
        """Run one branch: pad -> encoder -> neck, mirroring PixelWiseModel.forward.

        ``x`` is a tensor (single-modality) or a {modality: tensor} dict (native multi-modality).
        """
        from terratorch.models.utils import pad_images

        if isinstance(x, dict):
            image_size = next(iter(x.values())).shape[-2:]
        else:
            patch_size = getattr(self.base, "patch_size", None)
            if patch_size:
                x = pad_images(x, patch_size, self.base.padding)
            image_size = x.shape[-2:]
        features = encoder(x, **kwargs)
        features = neck(features, image_size=image_size)
        return features

    def _merge(self, feats_pre: list[torch.Tensor], feats_post: list[torch.Tensor]) -> list[torch.Tensor]:
        merged = []
        for i, (f_pre, f_post) in enumerate(zip(feats_pre, feats_post)):
            if self.fusion == "diff":
                m = f_post - f_pre
            elif self.fusion == "abs_diff":
                m = torch.abs(f_post - f_pre)
            else:  # learned
                m = self.merge[i](torch.cat([f_pre, f_post], dim=1))
            merged.append(m)
        return merged

    def forward(self, x: torch.Tensor, **kwargs):
        from terratorch.models.model import ModelOutput

        image_size = x.shape[-2:]
        if self.pre_modalities is not None and self.post_modalities is not None:
            # Native multi-modality: build per-branch modality dicts from the fused tensor.
            x_pre = self._slice_to_dict(x, self.pre_modalities, start=0)
            pre_total = sum(int(n) for _, n in self.pre_modalities)
            x_post = self._slice_to_dict(x, self.post_modalities, start=pre_total)
        else:
            x_pre = x[:, : self.pre_channels]
            x_post = x[:, self.pre_channels :]

        feats_pre = self._encode(self.base.encoder, self.base.neck, x_pre, **kwargs)
        feats_post = self._encode(self.encoder_post, self.neck_post, x_post, **kwargs)
        merged = self._merge(feats_pre, feats_post)

        decoder_output = self.base.decoder([f.clone() for f in merged])
        mask = self.base.head(decoder_output)
        if self.base.rescale and mask.shape[-2:] != image_size:
            mask = F.interpolate(mask, size=image_size, mode="bilinear")
        mask = mask[..., : image_size[0], : image_size[1]]
        return ModelOutput(output=mask, auxiliary_heads={})

    def freeze_encoder(self) -> None:
        self.base.freeze_encoder()
        if hasattr(self.encoder_post, "freeze"):
            self.encoder_post.freeze()
        else:
            _freeze_module(self.encoder_post)

    def freeze_decoder(self) -> None:
        self.base.freeze_decoder()

    def freeze_head(self) -> None:
        if hasattr(self.base, "freeze_head"):
            self.base.freeze_head()


class MidFusionEncoderDecoderFactory:
    """
    Accepts the same ``model_args`` as ``EncoderDecoderFactory`` plus extra keys:
    ``fusion`` (merge op), ``pre_channels`` (split point of the fused pre|post input for the
    single-modality case), and, for native multi-modality, ``post_overrides`` (model_args to
    override on the post branch, e.g. fewer modalities/channels) plus ``pre_modalities`` /
    ``post_modalities`` (ordered ``[[backbone_key, n_channels], ...]`` specs used to slice the
    fused tensor into per-branch modality dicts). Two independent backbones are built (the post
    branch keeps only its encoder + neck).
    """

    def build_model(
        self,
        task: str,
        *,
        pre_channels: int | None = None,
        fusion: str = "diff",
        aux_decoders=None,
        post_overrides: dict | None = None,
        pre_modalities: list | None = None,
        post_modalities: list | None = None,
        **model_args,
    ):
        from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory

        factory = EncoderDecoderFactory()
        base = factory.build_model(task, aux_decoders=aux_decoders, **model_args)
        post_args = dict(model_args)
        if post_overrides:
            post_args.update(post_overrides)
        base_post = factory.build_model(task, aux_decoders=None, **post_args)
        return MidFusionModel(
            base,
            base_post.encoder,
            base_post.neck,
            pre_channels=pre_channels,
            fusion=fusion,
            pre_modalities=pre_modalities,
            post_modalities=post_modalities,
        )


from terratorch.registry import MODEL_FACTORY_REGISTRY  # noqa: E402

MODEL_FACTORY_REGISTRY.register(MidFusionEncoderDecoderFactory)
