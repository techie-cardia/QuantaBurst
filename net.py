"""QuantaBurst — transposed channel-attention network for 384-channel single-photon input."""
import importlib.util, torch, torch.nn as nn

# Direct import of architecture file (avoids basicsr __init__ dependency chain)
_spec = importlib.util.spec_from_file_location(
    "_backbone_arch",
    "/mnt/zone/A/external/Restormer/basicsr/models/archs/restormer_arch.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_BackboneNet = _mod.Restormer


class QuantaBurst(nn.Module):
    """
    Transposed channel-attention encoder-decoder with inp_channels=384,
    ingesting all temporal channels directly.
    Global photon-mean residual is added at the output; no lift bottleneck.
    """
    TEMPORAL_FRAMES = 128

    def __init__(self):
        super().__init__()

        # 384 channels fed directly — no lift bottleneck
        self.trunk = _BackboneNet(
            inp_channels=384, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="WithBias",
        )

    @staticmethod
    def _photon_mean(x):
        """(B, 384, H, W) → (B, 3, H, W) photon-mean anchor."""
        B, _, H, W = x.shape
        return x.view(B, QuantaBurst.TEMPORAL_FRAMES, 3, H, W).mean(dim=1)

    def forward(self, x):
        """
        x: (B, 384, H, W) binary photon tensor.
        Returns: (B, 3, H, W) reconstructed RGB.
        """
        anchor = self._photon_mean(x)        # (B, 3, H, W)

        # Run trunk WITHOUT its built-in residual
        # patch_embed ingests all 384 channels directly
        r = self.trunk
        inp_enc_level1 = r.patch_embed(x)
        out_enc_level1 = r.encoder_level1(inp_enc_level1)

        inp_enc_level2 = r.down1_2(out_enc_level1)
        out_enc_level2 = r.encoder_level2(inp_enc_level2)

        inp_enc_level3 = r.down2_3(out_enc_level2)
        out_enc_level3 = r.encoder_level3(inp_enc_level3)

        inp_enc_level4 = r.down3_4(out_enc_level3)
        latent = r.latent(inp_enc_level4)

        inp_dec_level3 = r.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = r.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = r.decoder_level3(inp_dec_level3)

        inp_dec_level2 = r.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = r.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = r.decoder_level2(inp_dec_level2)

        inp_dec_level1 = r.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = r.decoder_level1(inp_dec_level1)

        out_dec_level1 = r.refinement(out_dec_level1)
        residual = r.output(out_dec_level1)   # (B, 3, H, W)

        return anchor + residual              # photon-mean residual
