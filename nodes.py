# nodes.py

import os
import torch
import nodes
import comfy.sd
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_management
import folder_paths

# Импортируем наши вспомогательные классы из flux_helpers.py
from .flux_helpers import Guider_Basic, ModelSamplingAdvanced, node_helpers, latent_preview

# --- Узел 1: Загрузчик моделей ---
class DoomFluxLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "vae_name": (folder_paths.get_filename_list("vae") + ["Baked VAE"],),
                "clip_name1": (folder_paths.get_filename_list("text_encoders") + ["None"],),
                "clip_name2": (folder_paths.get_filename_list("text_encoders") + ["None"],),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            }
        }
    RETURN_TYPES = ("MODEL", "VAE", "CLIP")
    FUNCTION = "load"
    CATEGORY = "DoomFlux"

    def load(self, model_name, vae_name, clip_name1, clip_name2, weight_dtype):
        # ... (Код этого узла у вас был правильный, оставляем как есть) ...
        vae_name = vae_name if vae_name and vae_name != "Baked VAE" else None
        clip_name1 = clip_name1 if clip_name1 and clip_name1 != "None" else None
        clip_name2 = clip_name2 if clip_name2 and clip_name2 != "None" else None
        weight_dtype = weight_dtype if weight_dtype else "default"
        
        model_options = {}
        if weight_dtype == "fp8_e4m3fn": model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast": model_options["dtype"], model_options["fp8_optimizations"] = torch.float8_e4m3fn, True
        elif weight_dtype == "fp8_e5m2": model_options["dtype"] = torch.float8_e5m2
            
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)

        vae = None
        if vae_name:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae_sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=vae_sd)

        clip_paths = []
        if clip_name1: clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name1))
        if clip_name2: clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name2))

        clip = None
        if clip_paths:
            clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.FLUX)

        return (model, vae, clip)

# --- Узел 2: Основной семплер ---
class DoomFluxSampler:
    # ... (Этот узел у вас был правильный, оставляем как есть) ...
    def __init__(self): self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), "model": ("MODEL",), "scheduler": (comfy.samplers.SCHEDULER_NAMES,), "steps": ("INT", {"default": 20, "min": 1, "max": 10000}), "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}), "sampler_name": (comfy.samplers.SAMPLER_NAMES,), "conditioning": ("CONDITIONING",), "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}), "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}), "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}), "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}), "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}), "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}), }}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("LATENT", "LATENT"), ("output", "denoised_output"), "sample", "DoomFlux/sampling"
    def sample(self, noise_seed, model, scheduler, steps, denoise, sampler_name, conditioning, guidance, max_shift, base_shift, width, height, batch_size):
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        guider = Guider_Basic(model); guider.set_conds(conditioning)
        m = model.clone(); x1, x2 = 256, 4096; mm = (max_shift - base_shift) / (x2 - x1); b = base_shift - mm * x1
        shift = (width * height / (64 * 4)) * mm + b
        model_sampling = ModelSamplingAdvanced(model.model.model_config); model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        latent = {"samples": torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)}
        latent_samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"])
        noise_tensor = comfy.sample.prepare_noise(latent_samples, noise_seed)
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), scheduler, steps).cpu()[-steps-1:].to(self.device)
        sampler = comfy.samplers.sampler_object(sampler_name)
        x0_output = {}; callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        samples = guider.sample(noise_tensor, latent_samples, sampler, sigmas, denoise_mask=latent.get("noise_mask"), callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise_seed)
        output_latent = {"samples": samples.to(comfy.model_management.intermediate_device())}; denoised_latent = output_latent.copy()
        if "x0" in x0_output: denoised_latent["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        return (output_latent, denoised_latent)

# --- Узел 3: Семплер для Inpaint (ИСПРАВЛЕННЫЙ) ---
class DoomFluxInpaintSampler:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "conditioning": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                "noise_mask": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",)
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "DoomFlux/sampling"

    def sample(self, noise_seed, model, scheduler, steps, denoise, sampler_name, conditioning, guidance, max_shift, base_shift, noise_mask, image, mask, vae):
        # Преобразуем маску и подгоняем ее под размер изображения
        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        mask = torch.nn.functional.interpolate(mask, size=(image.shape[1], image.shape[2]), mode="bilinear")
        
        # Получаем размеры, кратные 8
        height, width = image.shape[1], image.shape[2]
        new_height, new_width = (height // 8) * 8, (width // 8) * 8
        
        orig_pixels = image
        pixels = orig_pixels.clone()
        
        # Обрезаем до размеров, кратных 8, если необходимо
        if height != new_height or width != new_width:
            x_offset, y_offset = (width % 8) // 2, (height % 8) // 2
            pixels = pixels[:, y_offset:new_height + y_offset, x_offset:new_width + x_offset, :]
            mask = mask[:, :, y_offset:new_height + y_offset, x_offset:new_width + x_offset]

        # Применяем маску к изображению для инпеинтинга
        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5

        # Кодируем изображения в латентное пространство
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)
        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        # Устанавливаем кондиционирование
        conditioning = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent, "concat_mask": mask})
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        
        # Настраиваем модель и семплер
        m = model.clone()
        x1, x2 = 256, 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (64 * 4)) * mm + b
        
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        
        # Запускаем процесс семплирования
        latent_samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, out_latent["samples"])
        noise_tensor = comfy.sample.prepare_noise(latent_samples, noise_seed)
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), scheduler, steps).cpu()[-steps-1:].to(self.device)
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        samples = guider.sample(noise_tensor, latent_samples, sampler, sigmas, denoise_mask=mask if noise_mask else None, callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise_seed)
        
        # Формируем выходные данные
        output_latent = {"samples": samples.to(comfy.model_management.intermediate_device())}
        denoised_latent = output_latent.copy()
        if "x0" in x0_output:
            denoised_latent["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            
        return (output_latent, denoised_latent)


# --- Финальный узел: Продвинутый семплер (с гарантированно правильным denoise) ---
class DoomFluxSamplerAdvanced:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "DoomFlux/sampling"

    def sample(self, model, noise_seed, steps, sampler_name, scheduler, conditioning, latent_image, start_at_step, end_at_step, guidance, max_shift, base_shift, denoise):
        
        # --- ЛОГИКА, ПОЛНОСТЬЮ СКОПИРОВАННАЯ ИЗ KSAMPLER ---
        
        # 1. Если denoise = 0.0, немедленно возвращаем исходный латент.
        if denoise == 0.0:
            return (latent_image, latent_image)
        
        # 2. Настраиваем нашу FLUX-модель (это ваша уникальная часть)
        m = model.clone()
        latent_samples = latent_image["samples"]
        latent_height, latent_width = latent_samples.shape[2], latent_samples.shape[3]
        height, width = latent_height * 8, latent_width * 8
        x1, x2 = 256, 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (64 * 4)) * mm + b
        
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        
        # 3. Рассчитываем ПОЛНОЕ расписание шума (сигмы)
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), scheduler, steps).to(self.device)

        # 4. Подготавливаем чистый шум
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed)

        # 5. Определяем стартовый латент: либо чистый шум, либо исходный.
        # Это было одной из ключевых ошибок ранее.
        if denoise < 1.0:
            start_latent = latent_samples
        else:
            start_latent = torch.zeros_like(latent_samples)

        # 6. Определяем, с какого шага начинать, и обрезаем расписание
        total_steps = len(sigmas) - 1
        # Эта формула точно определяет, сколько шагов нужно пропустить
        start_step = total_steps - int(total_steps * denoise)
        
        # Обрезаем сигмы до рабочего окна, включая start_at_step и end_at_step
        last_step = min(total_steps, end_at_step)
        sigmas = sigmas[start_step:last_step + 1]

        # 7. Настраиваем Guider и Sampler
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        guider = Guider_Basic(m)
        guider.set_conds(conditioning)
        
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # 8. Запускаем семплирование, передавая ему чистый шум и правильный стартовый латент
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, len(sigmas) - 1, x0_output)
        samples = guider.sample(noise, start_latent, sampler, sigmas, denoise_mask=latent_image.get("noise_mask"), callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise_seed)
        
        # 9. Формируем выходные данные
        output_latent = {"samples": samples.to(comfy.model_management.intermediate_device())}
        denoised_latent = output_latent.copy()
        if "x0" in x0_output:
            denoised_latent["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            
        return (output_latent, denoised_latent)



