# __init__.py

# Импортируем все классы узлов из файла nodes.py
from .nodes import DoomFluxLoader, DoomFluxSampler, DoomFluxInpaintSampler, DoomFluxSamplerAdvanced

# Словарь для сопоставления внутренних имен классов с их реализациями
NODE_CLASS_MAPPINGS = {
    "DoomFluxLoader": DoomFluxLoader,
    "DoomFluxSampler": DoomFluxSampler,
    "DoomFluxInpaintSampler": DoomFluxInpaintSampler,
    "DoomFluxSamplerAdvanced": DoomFluxSamplerAdvanced,
}

# Словарь для сопоставления внутренних имен с отображаемыми именами в интерфейсе
NODE_DISPLAY_NAME_MAPPINGS = {
    "DoomFluxLoader": "DoomFLUX Loader",
    "DoomFluxSampler": "DoomFLUX Sampler",
    "DoomFluxInpaintSampler": "DoomFLUX Inpaint Sampler",
    "DoomFluxSamplerAdvanced": "DoomFLUX Sampler (Advanced)",
}

# Версия вашего пакета и индикатор загрузки
__version__ = "1.2.0" # Обновили версию, так как добавили новый узел
print(f"### DoomFLUX Nodes v{__version__}: Loaded successfully.")

