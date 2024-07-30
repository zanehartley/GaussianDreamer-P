from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import numpy as np
from PIL import Image
from transformers import pipeline
from torchvision.transforms.functional import to_pil_image

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

@threestudio.register("controlnet-guidance")
class ControlnetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        #self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        #self.controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
        self.controlnet_conditioning_scale = 0.5
        self.controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0-small", variant="fp16", torch_dtype=torch.float16, use_safetensors=True, addition_embed_type=None)
        self.vae  = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        pipe_kwargs = {
            "controlnet": self.controlnet,
            "vae": self.vae,
            "variant": "fp16",
            "use_safetensors": True,
            "torch_dtype": self.weights_dtype,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
        }

        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        #del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        down_block_additional_residuals: Float[Tensor, "..."], 
        mid_block_additional_residual: Float[Tensor, "..."],
        added_cond_kwargs: Dict
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            down_block_additional_residuals=down_block_additional_residuals, 
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 768 768"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 768 768"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
        
    def get_depth_map(self, image_batch, depth_estimator):
        # Move channel dimension to the last position if necessary (NHWC -> NCHW)
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 3, 1, 2)
        elif image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(0)
        # Convert tensors to PIL images (assuming CHW format)
        depth_maps = []
        for img in image_batch:
            img_np = img.cpu().numpy()
            img_np = img_np * 255
            img_np = img_np.astype(np.uint8)
            image = Image.fromarray(img_np)
            dp = depth_estimator(image)["depth"]
            dp = np.array(dp)
            dp = dp[:, :, None]
            dp = np.concatenate([dp, dp, dp], axis=2)
            depth_tensor = torch.from_numpy(np.array(dp)[None]).float()
            depth_tensor = depth_tensor.squeeze().permute(2, 0, 1)
            if depth_tensor.dim() == 3:
                depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension (if needed)
            depth_maps.append(depth_tensor)

        # Concatenate depth maps along the batch dimension (axis=0)
        depth_maps_tensor = torch.cat(depth_maps, dim=0)

        return depth_maps_tensor


    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        depths: Float[Tensor, "B 4 768 768"],
        images: Float[Tensor, "B H W C"],
        t: Int[Tensor, "B"],
        prompt: str,
        negative_prompt: str,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

                #depth_map = self.get_depth_map(images, self.depth_estimator).unsqueeze(0).to(self.device).half()
                depth_map = depths

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    added_cond_kwargs=added_cond_kwargs,
                )

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings, 
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                
                #depth_map = self.get_depth_map(images, self.depth_estimator).to(self.device).half()
                depth_map = depths.permute(0, 3, 1, 2)
                depth_map = torch.cat([depth_map] * 2, dim=0)


                #============================================================================================

                prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds= self.pipe.encode_prompt(
                    prompt=prompt,
                    device='cuda:0',
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt
                    )

                text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)

                original_size = (1024, 1024)
                target_size = (1024, 1024)
                (height, width) = (1024, 1024)
                crops_coords_top_left = (0,0)
                negative_crops_coords_top_left = (0,0)
                aesthetic_score = 6
                negative_aesthetic_score = 2.5
                
                if isinstance(depths, list):
                    original_size = original_size or depths[0].shape[-2:]
                else:
                    original_size = original_size or depths.shape[-2:]
                target_size = target_size or (height, width)

                negative_original_size = original_size
                negative_target_size = target_size
                add_text_embeds = pooled_prompt_embeds

                if self.pipe.text_encoder_2 is None:
                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                else:
                    text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

                add_time_ids, add_neg_time_ids = self.pipe._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    aesthetic_score,
                    negative_aesthetic_score,
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=text_embeddings.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )

                add_time_ids = add_time_ids.repeat(batch_size * 1, 1)

                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_neg_time_ids = add_neg_time_ids.repeat(batch_size * 1, 1)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(self.device)
                add_text_embeds = add_text_embeds.to(self.device)
                add_time_ids = add_time_ids.to(self.device)


                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                #============================================================================================

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    added_cond_kwargs=added_cond_kwargs,
                )

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "images": images,
            "depth": depths,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        images: Float[Tensor, "B H W C"],
        depths: Float[Tensor, "B H W C"],
        t: Int[Tensor, "B"],
        prompt: str,
        negative_prompt: str,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)

                #depth_map = self.get_depth_map(images, self.depth_estimator).unsqueeze(0).to(self.device).half()
                depth_map = depths

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    added_cond_kwargs=added_cond_kwargs,
                )

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)

                #depth_map = self.get_depth_map(images, self.depth_estimator).unsqueeze(0).to(self.device).half()
                depth_map = depths

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    added_cond_kwargs=added_cond_kwargs,
                )

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "depth": depths,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B H W C"],
        prompt: str,
        negative_prompt: str,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        
#====================== Prepare the prompt =============================

        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds= self.pipe.encode_prompt(
        prompt=prompt,
        device='cuda:0',
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt
        )

        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)

        self.batch_size = rgb.shape[0]

        #depth = depth.to(torch.float16)

#====================== Prepare the Controlnet Image =============================

        control_image = self.pipe.prepare_control_image(
            image=depth,
            width = 768,
            height= 768,
            batch_size= self.batch_size,
            num_images_per_prompt= 1,
            device=self.device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=True,
            guess_mode=False,
        )


#====================== Prepare the Timestep =====================================

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [self.batch_size],
            dtype=torch.long,
            device=self.device,
        )

#======================= Prepare latents =======================

        rgb_BCHW = rgb.permute(0, 3, 1, 2).half()
        latents = self.pipe.prepare_latents(
            rgb_BCHW,
            t,
            self.batch_size,
            1,
            prompt_embeds.dtype,
            self.device,
            None,
            True,
        )
        
        '''
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (768, 768), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)     
        '''


        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, rgb, depth, t, prompt, negative_prompt, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents, rgb, depth, t, prompt, negative_prompt, prompt_utils, elevation, azimuth, camera_distances
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / self.batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_eval_out.update({"depth": depth})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        image,
        depth,
        t,
        prompt,
        negative_prompt,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        #============================================================================================

        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds= self.pipe.encode_prompt(
            prompt=prompt,
            device='cuda:0',
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt
            )

        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)

        original_size = (1024, 1024)
        target_size = (1024, 1024)
        (height, width) = (1024, 1024)
        crops_coords_top_left = (0,0)
        negative_crops_coords_top_left = (0,0)
        aesthetic_score = 6
        negative_aesthetic_score = 2.5
        
        if isinstance(depth, list):
            original_size = original_size or depth[0].shape[-2:]
        else:
            original_size = original_size or depth.shape[-2:]
        target_size = target_size or (height, width)

        negative_original_size = original_size
        negative_target_size = target_size
        add_text_embeds = pooled_prompt_embeds

        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self.pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=text_embeddings.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        add_time_ids = add_time_ids.repeat(batch_size * 1, 1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size * 1, 1)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)


        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        #============================================================================================

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

            #depth_map = self.get_depth_map(image, self.depth_estimator).unsqueeze(0).to(self.device).half()
            depth_map = depth

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    added_cond_kwargs=added_cond_kwargs,
                )

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples, 
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            #depth_map = self.get_depth_map(image, self.depth_estimator).to(self.device).half()
            depth_map = depth
            
            latent_model_input  = latent_model_input.to(torch.float16)
            text_embeddings  = text_embeddings.to(torch.float16)

            depth_map = depth_map.permute(2,0,1)
            depth_map = depth_map.to(torch.float16)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    added_cond_kwargs=added_cond_kwargs,
                )

            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples, 
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        prompt,
        negative_prompt,
        t_orig,
        text_embeddings,
        images,
        depth,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, images[b], depth[b], t, prompt, negative_prompt, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
