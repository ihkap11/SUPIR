import torch
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from sgm.util import instantiate_from_config
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
import random
from SUPIR.utils.colorfix import wavelet_reconstruction, adaptive_instance_normalization
from pytorch_lightning import seed_everything
from SUPIR.utils.tilevae import VAEHook


class SUPIRModel(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        control_stage_config,
        ae_dtype="fp32",
        diffusion_dtype="fp32",
        p_p="",
        n_p="",
        *args,
        **kwargs
    ):
        super().__init__(vae=vae, unet=unet, scheduler=scheduler, *args, **kwargs)
        control_model = instantiate_from_config(control_stage_config)
        self.unet.load_control_model(control_model)
        self.vae.denoise_encoder = copy.deepcopy(self.vae.encoder)
        self.sampler_config = kwargs["sampler_config"]
        assert (ae_dtype in ["fp32", "fp16", "bf16"]) and (
            diffusion_dtype in ["fp32", "fp16", "bf16"]
        )
        if ae_dtype == "fp32":
            ae_dtype = torch.float32
        elif ae_dtype == "fp16":
            raise RuntimeError("fp16 cause NaN in AE")
        elif ae_dtype == "bf16":
            ae_dtype = torch.bfloat16
        if diffusion_dtype == "fp32":
            diffusion_dtype = torch.float32
        elif diffusion_dtype == "fp16":
            diffusion_dtype = torch.float16
        elif diffusion_dtype == "bf16":
            diffusion_dtype = torch.bfloat16
        self.ae_dtype = ae_dtype
        self.unet.dtype = diffusion_dtype
        self.p_p = p_p
        self.n_p = n_p

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            z = self.vae.encode(x).latent_dist.sample()
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def encode_first_stage_with_denoise(self, x, use_sample=True, is_stage1=False):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            if is_stage1:
                h = self.vae.denoise_encoder_s1(x)
            else:
                h = self.vae.denoise_encoder(x)
            moments = self.vae.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            if use_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", dtype=self.ae_dtype):
            out = self.vae.decode(z).sample
        return out.float()

    @torch.no_grad()
    def batchify_denoise(self, x, is_stage1=False):
        """
        [N, C, H, W], [-1, 1], RGB
        """
        x = self.encode_first_stage_with_den
