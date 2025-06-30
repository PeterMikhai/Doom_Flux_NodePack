# flux_helpers.py

import comfy.samplers
import node_helpers
import latent_preview

# Вспомогательный класс для управления CFG
class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

# Класс для расширенного семплирования Flux
class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    pass
