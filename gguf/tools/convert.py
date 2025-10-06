# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
import logging
import argparse
from tqdm import tqdm
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"
    shape_fix = False
    keys_detect = []
    keys_banned = []
    keys_hiprec = []
    keys_ignore = []
    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [("transformer_blocks.0.attn.norm_added_k.weight",), ("double_blocks.0.img_attn.proj.weight",)]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [("transformer_blocks.0.attn.add_q_proj.weight",), ("joint_blocks.0.x_block.attn.qkv.weight",)]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [("double_layers.3.modX.1.weight",), ("joint_transformer_blocks.3.ff_context.out_projection.weight",)]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = ["caption_projection.0.linear.weight", "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"]
    keys_hiprec = [".ff_i.gate.weight", "img_emb.emb_pos"]

class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = ["blocks.0.mlp.layer1.weight", "blocks.0.adaln_modulation_cross_attn.1.weight"]
    keys_hiprec = ["pos_embedder"]
    keys_ignore = ["_extra_state", "accum_"]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = ["double_blocks.0.img_attn_proj.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight"]
    def handle_nd_tensor(self, key, data):
        path = f"./fix_5d_tensors_{self.arch}.safetensors"
        if os.path.isfile(path): raise RuntimeError(f"5D tensor fix file already exists! {path}")
        fsd = {key: torch.from_numpy(data)}
        tqdm.write(f"5D key found in state dict! Manual fix required! - {key} {data.shape}")
        save_file(fsd, path)

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = ["blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation"]
    keys_hiprec = [".modulation"]

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = ["adaln_single.emb.timestep_embedder.linear_2.weight", "transformer_blocks.27.scale_shift_table", "caption_projection.linear_2.weight"]
    keys_hiprec = ["scale_shift_table"]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",), ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",), ("label_emb.0.0.weight",)]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [("down_blocks.0.downsamplers.0.conv.weight",), ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight")]

class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")]

# --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
# Создаем экземпляры классов, а не просто перечисляем типы
arch_list = [ModelFlux(), ModelSD3(), ModelAura(), ModelHiDream(), CosmosPredict2(),
             ModelLTXV(), ModelHyVid(), ModelWan(), ModelSDXL(), ModelSD1(), ModelLumina2()]

# Добавляем в arch_remap все известные архитектуры
arch_remap = {x.arch: x.__class__ for x in arch_list}
# --- КОНЕЦ ИСПРАВЛЕНИЯ ---

def is_model_arch(model, state_dict):
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        # Убедимся, что match_list - это кортеж или список
        if not isinstance(match_list, (list, tuple)):
            match_list = (match_list,)
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    if invalid:
        raise RuntimeError("Model architecture not allowed for conversion! (i.e. reference VS diffusers format)")
    return matched

def detect_arch(state_dict):
    for arch_instance in arch_list:
        if is_model_arch(arch_instance, state_dict):
            return arch_instance
    raise RuntimeError("Unknown model architecture!")

# ... (остальной код файла остается без изменений) ...

def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output unet gguf file.")
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        parser.error("No input provided!")
    return args

def strip_prefix(state_dict):
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any(x.startswith(pfx) for x in state_dict.keys()):
            prefix = pfx
            break
    if prefix is None:
        for pfx in ["net."]:
            if all(x.startswith(pfx) for x in state_dict.keys()):
                prefix = pfx
                break
    if prefix is not None:
        logging.info(f"State dict prefix found: '{prefix}'")
        sd = {k.replace(prefix, ""): v for k, v in state_dict.items() if prefix in k}
    else:
        logging.debug("State dict has no prefix")
        sd = state_dict
    return sd

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module", "state_dict"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20: raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        state_dict = load_file(path)
    return strip_prefix(state_dict)

def handle_tensors(writer, state_dict, model_arch):
    # ... (остальной код функции без изменений) ...
    pass

def convert_file(path, dst_path=None, interact=True, overwrite=False):
    # ... (остальной код функции без изменений) ...
    pass

if __name__ == "__main__":
    args = parse_args()
    convert_file(args.src, args.dst)
