from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
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
import inspect
import random

def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    device = None,
    timesteps = None,
    sigmas = None,
    **kwargs,
):
    
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

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

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
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

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        print(f"num_train_timesteps: {self.num_train_timesteps}")
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
        latents = posterior.sample() * 0.18215 #self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 128,
        latent_width: int = 128,
    ) -> Float[Tensor, "B 3 768 768"]:
        input_dtype = latents.dtype
        #print(latents.size())
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / 0.18215 * latents #self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def zanes_decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 128,
        latent_width: int = 128,
    ) -> Float[Tensor, "B 3 768 768"]:
        input_dtype = latents.dtype

        #latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / 0.18215 * latents #self.vae.config.scaling_factor * latents)
        image = self.pipe.vae.decode(latents.to(self.weights_dtype)).sample
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
        images: Float[Tensor, "B H W C"],
        depth: Float[Tensor, "B 4 768 768"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        added_cond_kwargs,
        prompt_embeds,
    ):
        batch_size = elevation.shape[0]

        #============================================================================================
        #text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)
        #============================================================================================

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                #noise = torch.randn_like(latents)

                num_channels_latents = self.pipe.unet.config.in_channels

                noise = self.pipe.prepare_latents(
                    self.batch_size * 1,
                    num_channels_latents,
                    768,
                    768,
                    depth.dtype,
                    self.device,
                    None,
                    None,
                )

                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depths,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    #added_cond_kwargs=added_cond_kwargs,
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
                
                #======================= Prepare latents =======================

                shape = latents.shape
                noise = torch.randn(shape, generator=None, device=self.device, dtype=latents.dtype, layout=None).to(self.device)

                print(t)
                print(self.pipe.scheduler.timesteps)

                latents_noisy = self.pipe.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=depth,
                    conditioning_scale = 0.75,
                    return_dict=False,
                )

                noise_pred = self.forward_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_uncond - noise_pred_text)

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
            "use_perp_neg": prompt_utils.use_perp_neg,
            "t_orig": t,
            "images": images,
            "depth": depth,
            "latents_orig": latents,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "added_cond_kwargs": added_cond_kwargs,
            "prompt_embeds": prompt_embeds,
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        images: Float[Tensor, "B H W C"],
        depths: Float[Tensor, "B H W C"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        added_cond_kwargs,
        prompt_embeds,
    ):
        batch_size = elevation.shape[0]

        #============================================================================================

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

                depth_map = depths

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_map,
                    conditioning_scale = 1.0,
                    return_dict=False,
                    #added_cond_kwargs=added_cond_kwargs,
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
                    #added_cond_kwargs=added_cond_kwargs,
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
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_uncond - noise_pred_text)

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "t_orig": t,
            "images": images,
            "depth": depths,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
            "added_cond_kwargs": added_cond_kwargs,
            "prompt_embeds": prompt_embeds,
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
        num_inference_steps=50,
        **kwargs,
    ):
        
#====================== Prepare the Controlnet Image =============================
        self.batch_size = depth.shape[0]

        depth = depth.permute(0, 3, 1, 2)
        depth = self.pipe.image_processor.preprocess(depth, height=768, width=768).to(dtype=torch.float16)

        control_image = self.pipe.prepare_image(
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
        control_image = control_image.to(dtype=torch.float16)


#====================== Prepare the Timestep =====================================

        timesteps, num_inference_steps = retrieve_timesteps(
        self.pipe.scheduler, 
        num_inference_steps, 
        self.device, 
        None, 
        None
        )
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        # Randomly select an index
        index = torch.randint(len(timesteps), (1,), device=timesteps.device)
        # Get the corresponding timestep
        t = timesteps[index]

#======================= Prepare latent image =======================
        
        rgb_p = rgb.permute(0, 3, 1, 2)
        imgs = rgb_p * 2.0 - 1.0
        posterior = self.pipe.vae.encode(imgs.to(dtype=torch.float16)).latent_dist
        image_latents = posterior.sample() * 0.18215 #self.vae.config.scaling_factor
        image_latents = image_latents.to(dtype=torch.float16)


#====================== Prepare the prompt =============================

        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds= self.pipe.encode_prompt(
        prompt=prompt,
        device='cuda:0',
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt
        )

#======================== Prepare added time ids & embeddings =======================================

        original_size = (768, 768)
        target_size = (768, 768)
        (height, width) = (768, 768)
        crops_coords_top_left = (0,0)
        negative_crops_coords_top_left = (0,0)
        aesthetic_score = 6
        negative_aesthetic_score = 2.5

        add_text_embeds = pooled_prompt_embeds

        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(self.batch_size * 1, 1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        #add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        add_neg_time_ids = add_time_ids
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)


        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

#====================== Denoising loop =========================================

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                image_latents, rgb, control_image, t, prompt_utils, elevation, azimuth, camera_distances, added_cond_kwargs
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                image_latents, rgb, control_image, t, prompt_utils, elevation, azimuth, camera_distances, added_cond_kwargs, prompt_embeds
            )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (image_latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(image_latents, target, reduction="sum") / self.batch_size

        print(f"Loss_sds = {loss_sds}")

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils) ####### <--------------- This is where the main loop happens
            texts = []

            print(f"guidance_eval_out: {guidance_eval_out['noise_levels']}")
            print(f"elevation: {elevation}")
            print(f"elevation: {azimuth}")
            print(f"elevation: {camera_distances}")

            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}")
            guidance_eval_out.update({"texts": texts})
            guidance_eval_out.update({"depth": depth})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out


    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        images,
        depth,
        latents_orig,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        added_cond_kwargs=None,
        prompt_embeds=None,
    ):
        
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50, device=self.device)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        t = self.scheduler.timesteps_gpu[0]

        large_enough_idxs = self.scheduler.timesteps_gpu.expand([-1]) > t_orig.unsqueeze(-1)  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = (t / self.scheduler.config.num_train_timesteps).cpu().numpy()
        imgs_noisy = self.decode_latents(latents_noisy[:1]).permute(0,2, 3, 1)

        step_output = self.scheduler.step(noise_pred[0:1], t, latents_noisy[0:1])
        latents_1step = step_output["prev_sample"].squeeze(0)
        #pred_1orig = step_output["pred_original_sample"].squeeze(0)
        pred_1orig = step_output["prev_sample"].squeeze(0)
        imgs_1step = self.decode_latents(latents_1step.unsqueeze(0)).permute(0,2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig.unsqueeze(0)).permute(0, 2, 3, 1)

        latents = latents_1step
        text_emb = prompt_embeds[[0, 1], ...]
        neg_guid = None

        batch_size = images.size()[0]     

        for i, t in enumerate(tqdm(self.scheduler.timesteps[idxs + 1:], leave=False)):    

            latent_model_input = latents.unsqueeze(dim=0)
            latent_model_input_copy = latent_model_input.clone()
            latent_model_input = torch.cat([latent_model_input, latent_model_input_copy], dim=0).squeeze()
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t).half()

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=depth[0],
                conditioning_scale = 0.75,
                return_dict=False,
                #added_cond_kwargs=added_cond_kwargs,
            )

            noise_pred = self.forward_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples, 
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
            )

            noise_pred_uncond ,noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # get prev latent
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        latents_final = latents

        imgs_final = self.zanes_decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": batch_size,
            "noise_levels": fracs, #formally fracs
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
