# flux_helpers.py
import torch
import comfy.samplers
import comfy.model_sampling
import comfy.utils

class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    pass

# Добавляем отсутствующие модули
class node_helpers:
    @staticmethod
    def conditioning_set_values(conditioning, values):
        """Добавляет значения в conditioning"""
        c = []
        for cond in conditioning:
            n = {}
            for k in values:
                n[k] = values[k]
            n.update(cond[1])
            c.append([cond[0], n])
        return c

class latent_preview:
    @staticmethod
    def prepare_callback(model_patcher, steps, x0_output):
        """Подготавливает callback для preview"""
        def callback(step, x0, x, total_steps):
            if x0 is not None:
                x0_output["x0"] = x0
            
        return callback if comfy.utils.PROGRESS_BAR_ENABLED else None
