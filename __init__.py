from .nodes import (
    DoomFluxLoader,
    DoomFluxLoader_GGUF,
    DoomFluxSampler,
    DoomFluxInpaintSampler,
    DoomFluxSamplerAdvanced,
    DoomFluxSuperLoader,
)

NODE_CLASS_MAPPINGS = {
    "DoomFluxLoader": DoomFluxLoader,
    "DoomFluxLoader_GGUF": DoomFluxLoader_GGUF,
    "DoomFluxSampler": DoomFluxSampler,
    "DoomFluxInpaintSampler": DoomFluxInpaintSampler,
    "DoomFluxSamplerAdvanced": DoomFluxSamplerAdvanced,
    "DoomFluxSuperLoader": DoomFluxSuperLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoomFluxLoader": "DoomFlux Loader",
    "DoomFluxLoader_GGUF": "DoomFlux Loader (GGUF)",
    "DoomFluxSampler": "DoomFlux Sampler",
    "DoomFluxInpaintSampler": "DoomFlux Inpaint Sampler",
    "DoomFluxSamplerAdvanced": "DoomFlux Sampler Advanced",
    "DoomFluxSuperLoader": "DoomFlux Super Loader",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

print("DoomFLUX Nodes v1.3.0 Loaded successfully.")
