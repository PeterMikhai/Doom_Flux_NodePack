# Импортируем все классы узлов из файла nodes.py
from .nodes import DoomFluxLoader, DoomFluxSampler, DoomFluxInpaintSampler

# Словарь для сопоставления внутренних имен классов с их реализациями
NODE_CLASS_MAPPINGS = {
    "DoomFluxLoader": DoomFluxLoader,
    "DoomFluxSampler": DoomFluxSampler,
    "DoomFluxInpaintSampler": DoomFluxInpaintSampler,
}

# Словарь для сопоставления внутренних имен с отображаемыми именами в интерфейсе
NODE_DISPLAY_NAME_MAPPINGS = {
    "DoomFluxLoader": "DoomFLUX Loader",
    "DoomFluxSampler": "DoomFLUX Sampler",
    "DoomFluxInpaintSampler": "DoomFLUX Inpaint Sampler",
}

# Сообщаем, что экспорт завершен
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("------------------------------------------")
print("DoomFLUX Nodes: Successfully loaded.")
print("------------------------------------------")
