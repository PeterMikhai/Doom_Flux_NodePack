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
import comfy.clip_vision
import logging
from .flux_helpers import Guider_Basic, ModelSamplingAdvanced, node_helpers, latent_preview

# Словарь для удобного выбора типа данных (dtype)
DTYPE_MAP = {
    "default": {},
    "fp16": {"dtype": torch.float16},
    "bf16": {"dtype": torch.bfloat16},
    "fp8_e4m3fn": {"dtype": torch.float8_e4m3fn},
    "fp8_e4m3fn_fast": {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True},
    "fp8_e5m2": {"dtype": torch.float8_e5m2},
}

# Список всех доступных типов CLIP, объединенный из обоих загрузчиков
CLIP_TYPES = [
    "stable_diffusion", "stable_cascade", "sdxl", "sd3", "flux", "hidream",
    "hunyuan_video", "stable_audio", "mochi", "ltxv", "pixart", "cosmos",
    "lumina2", "wan", "chroma", "ace", "omnigen2"
]
# --- Глобальные настройки и импорты GGUF ---

def _register_unet_gguf_key():
    key = "unet_gguf"
    if key in folder_paths.folder_names_and_paths:
        return
    base_paths, _ = folder_paths.folder_names_and_paths.get("diffusion_models", ([], {}))
    base_paths = list(base_paths) if isinstance(base_paths, (list, tuple, set)) else [base_paths]
    folder_paths.folder_names_and_paths[key] = (base_paths, {".gguf"})

_register_unet_gguf_key()

_GGUF_OK = False
try:
    from .gguf.loader import gguf_sd_loader, gguf_clip_loader
    from .gguf.ops import GGMLOps
    from .gguf.patcher import GGUFModelPatcher
    from .gguf.tools.convert import detect_arch # Возвращаемся к detect_arch
    _GGUF_OK = True
except ImportError as e:
    logging.error(f"[DoomFluxLoader] CRITICAL: GGUF backend failed to load: {e}. Check file structure and dependencies.")

# --- Узел 1: Загрузчик моделей для Flux ---
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
    
# --- Узел 2: Расширенный загрузчик с GGUF ---
class DoomFluxLoader_GGUF:
    @classmethod
    def get_clip_filename_list(cls):
        """Возвращает список CLIP файлов включая GGUF"""
        # Получаем обычные CLIP файлы
        regular_clips = folder_paths.get_filename_list("text_encoders")
        
        # Получаем GGUF файлы из папки clip
        clip_dirs = folder_paths.get_folder_paths("clip")
        gguf_files = []
        if clip_dirs:
            for clip_dir in clip_dirs:
                if os.path.exists(clip_dir):
                    for file in os.listdir(clip_dir):
                        if file.lower().endswith('.gguf'):
                            gguf_files.append(file)
        
        # Объединяем списки и убираем дубликаты
        all_clips = list(set(regular_clips + gguf_files))
        return sorted(all_clips)
    
    @classmethod
    def INPUT_TYPES(cls):
        clip_types = sorted([
            "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
            "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream",
            "chroma", "ace", "omnigen2", "qwen_image", "flux",
        ])
        
        # Используем метод для получения всех CLIP файлов включая GGUF
        clip_files = cls.get_clip_filename_list() + ["None"]
        
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models") + ["GGUF_ONLY"],),
                "gguf_unet_name": (folder_paths.get_filename_list("unet_gguf") + ["None"],),
                "vae_name": (folder_paths.get_filename_list("vae") + ["Baked VAE"],),
                "clip_name1": (clip_files,),
                "clip_name2": (clip_files,),
                "clip_type": (clip_types,),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            },
            "optional": { "device": (["default", "cpu"], {"advanced": True}), },
        }

    RETURN_TYPES = ("MODEL", "VAE", "CLIP")
    FUNCTION = "load"
    CATEGORY = "DoomFlux/loaders"

    def _build_model_options(self, weight_dtype, device):
        mo = {}
        if device == "cpu": 
            mo["load_device"] = mo["offload_device"] = torch.device("cpu")
        return mo

    def _to_clip_type(self, type_str):
        return getattr(comfy.sd.CLIPType, type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

    def _load_gguf_model(self, gguf_unet_name, model_options):
        logging.info(f"[DoomFluxLoader] Loading GGUF UNet: {gguf_unet_name}")
        gpath = folder_paths.get_full_path_or_raise("unet_gguf", gguf_unet_name)
        
        try:
            state_dict, _ = gguf_sd_loader(gpath, return_arch=True)
            logging.info("[DoomFluxLoader] GGUF state_dict loaded.")
            model_options["custom_operations"] = GGMLOps()
            
            # Создаем временную модель, чтобы извлечь из нее 'голую' unet
            temp_model_patcher = comfy.sd.load_diffusion_model_state_dict(state_dict, model_options)
            if temp_model_patcher is None:
                raise RuntimeError("Failed to create a model from GGUF state_dict.")
            
            unet_model = temp_model_patcher.model
            logging.info(f"[DoomFluxLoader] Created base model of type: {type(unet_model).__name__}")

            # Создаем наш GGUFModelPatcher, передавая все 3 обязательных аргумента
            load_device = model_options.get("load_device", comfy.model_management.get_torch_device())
            offload_device = model_options.get("offload_device", comfy.model_management.unet_offload_device())
            
            model = GGUFModelPatcher(unet_model, load_device=load_device, offload_device=offload_device)
            model.model_options = temp_model_patcher.model_options.copy()
            
            logging.info("[DoomFluxLoader] GGUF UNet wrapped in Patcher and loaded successfully.")
            return model
            
        except Exception as e:
            logging.error(f"[DoomFluxLoader] GGUF UNet load failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _load_fp_model(self, model_name, model_options):
        if model_name == "GGUF_ONLY":
            raise ValueError("A GGUF file was not selected in 'gguf_unet_name', but 'GGUF_ONLY' is set. Please select a GGUF file or a regular model.")
        logging.info(f"[DoomFluxLoader] Loading FP model: {model_name}")
        mpath = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return comfy.sd.load_diffusion_model(mpath, model_options=model_options)

    def _load_vae(self, vae_name):
        if vae_name and vae_name != "Baked VAE":
            vpath = folder_paths.get_full_path_or_raise("vae", vae_name)
            vsd = comfy.utils.load_torch_file(vpath)
            return comfy.sd.VAE(sd=vsd)
        return None

    def _load_clip(self, clip_name1, clip_name2, clip_type, device):
        paths = []
        
        # Загружаем clip_name1 с проверкой, является ли файл GGUF
        if clip_name1 and clip_name1 != "None":
            if clip_name1.lower().endswith('.gguf'):
                paths.append(folder_paths.get_full_path_or_raise("clip", clip_name1))
            else:
                paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name1))
        
        # Загружаем clip_name2 с проверкой, является ли файл GGUF
        if clip_name2 and clip_name2 != "None":
            if clip_name2.lower().endswith('.gguf'):
                paths.append(folder_paths.get_full_path_or_raise("clip", clip_name2))
            else:
                paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name2))
        
        if not paths:
            return None  # нет CLIP, вернем None

        ctype = self._to_clip_type(clip_type)
        has_gguf = any(os.path.splitext(p)[1].lower() == ".gguf" for p in paths)

        # Ветвь GGUF: смешанный ввод (.gguf + не-.gguf)
        if has_gguf:
            if not _GGUF_OK:
                logging.warning("[DoomFluxLoader] CLIP .gguf selected, but backend is disabled. Skipping CLIP.")
                return None  # безопасный выход, чтобы не падать
            try:
                clip_data = []
                for p in paths:
                    if p.lower().endswith(".gguf"):
                        # Загрузить GGUF энкодер через плагин
                        clip_data.append(gguf_clip_loader(p))  # возвращает совместимые sd dict/структуры
                    else:
                        # Обычные весы как sd dict
                        clip_data.append(comfy.utils.load_torch_file(p, safe_load=True))

                mo = {
                    "custom_operations": GGMLOps(),  # требуется для ggml-операций
                    "initial_device": comfy.model_management.text_encoder_offload_device(),
                }
                clip = comfy.sd.load_text_encoder_state_dicts(
                    ctype, clip_data, mo, folder_paths.get_folder_paths("embeddings")
                )  # современный путь загрузки TE

                # Выравнивание патчинга под GGUF унификацию
                if hasattr(clip, "patcher") and clip.patcher is not None:
                    clip.patcher = GGUFModelPatcher.clone(clip.patcher)  # делает поведение единым
                return clip
            except Exception as e:
                logging.error(f"[DoomFluxLoader] GGUF CLIP load failed: {e}")
                import traceback
                traceback.print_exc()
                return None  # не роняем весь загрузчик

        # Ветвь без GGUF: классическая загрузка
        mo = {}
        if device == "cpu":
            mo["load_device"] = mo["offload_device"] = torch.device("cpu")
        return comfy.sd.load_clip(paths, folder_paths.get_folder_paths("embeddings"), ctype, mo)

    def load(self, model_name, gguf_unet_name, vae_name, clip_name1, clip_name2, clip_type, weight_dtype, device="default"):
        mo = self._build_model_options(weight_dtype, device)
        if gguf_unet_name and gguf_unet_name != "None":
            model = self._load_gguf_model(gguf_unet_name, mo)
        else:
            model = self._load_fp_model(model_name, mo)
        from .flux_helpers import ModelSamplingAdvanced
        model.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config))
        vae = self._load_vae(vae_name)
        clip = self._load_clip(clip_name1, clip_name2, clip_type, device)
        return (model, vae, clip)

# --- Узел 3: Основной семплер --
class DoomFluxSampler:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        # Убрали 'denoise'
        return {
            "required": {
                "model": ("MODEL",),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "conditioning": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step":0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("LATENT", "LATENT"), ("output", "denoised_output"), "sample", "DoomFlux/sampling"

    def sample(self, model, noise_seed, steps, sampler_name, scheduler, conditioning, guidance, max_shift, base_shift, width, height, batch_size):
        from .flux_helpers import Guider_Basic
        
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        
        m = model.clone()
        # Получаем model_sampling, который уже существует
        model_sampling = m.get_model_object("model_sampling")
        
        x1, x2 = 256, 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (64 * 4)) * mm + b
        model_sampling.set_parameters(shift=shift)
        
        latent = {"samples": torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)}
        latent_samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"])
        noise_tensor = comfy.sample.prepare_noise(latent_samples, noise_seed)
        
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), scheduler, steps).cpu()[-steps-1:].to(self.device)
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        
        samples = guider.sample(noise_tensor, latent_samples, sampler, sigmas, denoise_mask=latent.get("noise_mask"), callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise_seed)
        
        output_latent = {"samples": samples.to(comfy.model_management.intermediate_device())}
        denoised_latent = output_latent.copy()
        
        if "x0" in x0_output:
            denoised_latent["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            
        return (output_latent, denoised_latent)


# --- Узел 4: Семплер для Inpaint ---
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

# --- Узел 5: Продвинутый семплер с Динамическим CFG ---
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg_start": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "DoomFlux/sampling"

    def sample(self, model, noise_seed, steps, sampler_name, scheduler, conditioning, latent_image, 
               start_at_step, end_at_step, denoise, cfg_start, cfg_end, max_shift, base_shift):
        
        from .flux_helpers import ModelSamplingAdvanced, Guider_Basic
        import torch
        
        if denoise <= 0.0:
            return (latent_image, latent_image)

        # Настройка guidance
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": cfg_start})
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        
        m = model.clone()
        
        # Настройка model sampling
        latent_samples = latent_image["samples"]
        height, width = latent_samples.shape[2] * 8, latent_samples.shape[3] * 8
        
        x1, x2 = 256, 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (64 * 4)) * mm + b
        
        model_sampling = ModelSamplingAdvanced(m.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        
        # Подготовка семплирования
        latent_samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_samples)
        
        # Люксовая логика денойза
        noise_tensor = comfy.sample.prepare_noise(latent_samples, noise_seed)
        starting_latent = latent_samples * (1.0 - denoise) + noise_tensor * denoise
        
        # Вычисляем sigmas для полного процесса
        full_sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), scheduler, steps)
        
        # Масштабируем sigmas по denoise
        max_sigma = full_sigmas[0].item()
        scaled_sigmas = []
        
        for i, sigma in enumerate(full_sigmas):
            progress = i / (len(full_sigmas) - 1) if len(full_sigmas) > 1 else 1.0
            current_denoise_level = denoise * (1.0 - progress)
            scaled_sigma = sigma.item() * current_denoise_level / max_sigma
            scaled_sigmas.append(scaled_sigma)
        
        # Применяем start_at_step и end_at_step
        actual_start = max(0, start_at_step)
        actual_end = min(end_at_step, steps)
        
        if actual_start >= actual_end:
            return (latent_image, latent_image)
        
        final_sigmas = torch.tensor(scaled_sigmas[actual_start:actual_end + 1], 
                                   device=self.device, dtype=torch.float32)
        
        # Запускаем семплирование
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, final_sigmas.shape[-1] - 1, x0_output)
        
        samples = guider.sample(starting_latent, starting_latent, sampler, final_sigmas,
                              denoise_mask=latent_image.get("noise_mask"),
                              callback=callback,
                              disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                              seed=noise_seed)
        
        output_latent = {"samples": samples.to(comfy.model_management.intermediate_device())}
        denoised_latent = output_latent.copy()
        
        if "x0" in x0_output:
            denoised_latent["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        
        return (output_latent, denoised_latent)
    
# --- Узел 6: Универсальный "супер-загрузчик", который объединяет загрузку чекпоинтов, diffusion-моделей, VAE, до 3-х CLIP-моделей, CLIPVision и StyleModel. Позволяет гибко переопределять компоненты из чекпоинта. ---
class DoomFluxSuperLoader:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "checkpoint_name": (["None"] + folder_paths.get_filename_list("checkpoints"),),
                "model_name": (["None"] + folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (list(DTYPE_MAP.keys()),),
                "vae_name": (["None", "Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_name1": (["None"] + folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (["None"] + folder_paths.get_filename_list("text_encoders"),),
                "clip_name3": (["None"] + folder_paths.get_filename_list("text_encoders"),),
                "clip_type": (CLIP_TYPES,),
                "clip_device": (["default", "cpu"],),
                "clip_vision_name": (["None"] + folder_paths.get_filename_list("clip_vision"),),
                "style_model_name": (["None"] + folder_paths.get_filename_list("style_models"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION", "STYLE_MODEL")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "CLIP_VISION", "STYLE_MODEL")
    FUNCTION = "load"
    CATEGORY = "DoomFlux"

    def load(self, checkpoint_name="None", model_name="None", weight_dtype="default",
            vae_name="None", clip_name1="None", clip_name2="None", clip_name3="None",
            clip_type="flux", clip_device="default", clip_vision_name="None", style_model_name="None"):

        model, clip, vae, clip_vision, style_model = None, None, None, None, None

        if checkpoint_name and checkpoint_name != "None":
            print(f"DoomFluxSuperLoader: Loading from checkpoint '{checkpoint_name}'")
            ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
            model, clip, vae = out[:3]

        if model_name and model_name != "None":
            print(f"DoomFluxSuperLoader: Loading/overriding diffusion model '{model_name}' with dtype: {weight_dtype}")
            model_path = folder_paths.get_full_path("diffusion_models", model_name)
            model_options = DTYPE_MAP.get(weight_dtype, {})
            model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)

        if vae_name and vae_name not in ["None", "Baked VAE"]:
            print(f"DoomFluxSuperLoader: Loading/overriding VAE '{vae_name}'")
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        clip_paths = []
        if clip_name1 and clip_name1 != "None": clip_paths.append(folder_paths.get_full_path("text_encoders", clip_name1))
        if clip_name2 and clip_name2 != "None": clip_paths.append(folder_paths.get_full_path("text_encoders", clip_name2))
        if clip_name3 and clip_name3 != "None": clip_paths.append(folder_paths.get_full_path("text_encoders", clip_name3))

        if clip_paths:
            print(f"DoomFluxSuperLoader: Loading/overriding CLIPs ({len(clip_paths)} models) with type '{clip_type}'")
            clip_model_options = {}
            if clip_device == "cpu":
                clip_model_options["load_device"] = torch.device("cpu")
                clip_model_options["offload_device"] = torch.device("cpu")
            clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type_enum,
                model_options=clip_model_options
            )

        if clip_vision_name and clip_vision_name != "None":
            print(f"DoomFluxSuperLoader: Loading CLIP Vision model '{clip_vision_name}'")
            cv_path = folder_paths.get_full_path("clip_vision", clip_vision_name)
            clip_vision = comfy.clip_vision.load(cv_path)
            
        if style_model_name and style_model_name != "None":
            print(f"DoomFluxSuperLoader: Loading Style model '{style_model_name}'")
            style_path = folder_paths.get_full_path("style_models", style_model_name)
            # ИСПРАВЛЕНИЕ: Используем правильную функцию для загрузки StyleModel
            style_model = comfy.sd.load_style_model(style_path)

        return (model, clip, vae, clip_vision, style_model)




