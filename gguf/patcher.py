# Содержимое для gguf/patcher.py

import torch
import collections
import comfy.utils
import comfy.model_patcher
from .dequant import is_quantized
from .ops import move_patch_to_device

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ---
    def __init__(self, model, load_device, offload_device, size=0, **kwargs):
        # Вызываем родительский конструктор, передавая ему все аргументы,
        # включая любые неожиданные, такие как 'weight_inplace_update'.
        super().__init__(model, load_device, offload_device, size, **kwargs)
        self.patch_on_device = False
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)
        try: from comfy.lora import calculate_weight
        except Exception: calculate_weight = self.calculate_weight
        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight','inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)
            out_weight = calculate_weight(self.backup[key].weight, patches, key)
        if inplace_update: comfy.utils.copy_to_param(self.model, key, out_weight)
        else: comfy.utils.set_attr_param(self.model, key, out_weight)

    def clone(self, *args, **kwargs):
        # Метод clone теперь будет работать корректно, т.к. __init__ совместим
        n = super().clone(*args, **kwargs)
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n

