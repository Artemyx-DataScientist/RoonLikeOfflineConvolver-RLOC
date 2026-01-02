from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class FilterPath:
    """Настройки отдельного фильтра/конволюции."""

    in_gains: List[float]
    out_gains: List[float]
    ir_path: str
    ir_channel: int
    delay_ms: float = 0.0

    def __post_init__(self) -> None:
        self._validate_gains("in_gains", self.in_gains)
        self._validate_gains("out_gains", self.out_gains)
        if not self.ir_path:
            raise ValueError("ir_path должен быть непустой строкой")
        if self.ir_channel < 0:
            raise ValueError("ir_channel должен быть неотрицательным")

    @staticmethod
    def _validate_gains(name: str, gains: List[float]) -> None:
        if not gains:
            raise ValueError(f"{name} должен содержать хотя бы одно значение")
        if not all(isinstance(value, (int, float)) for value in gains):
            raise TypeError(f"{name} должен содержать только числа")


@dataclass(slots=True)
class FilterConfig:
    """Конфигурация DSP-фильтров."""

    sample_rate: int
    num_in: int
    num_out: int
    paths: List[FilterPath] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate должен быть положительным")
        if self.num_in <= 0:
            raise ValueError("num_in должен быть положительным")
        if self.num_out <= 0:
            raise ValueError("num_out должен быть положительным")
        if not isinstance(self.paths, list):
            raise TypeError("paths должен быть списком объектов FilterPath")
        for path in self.paths:
            if not isinstance(path, FilterPath):
                raise TypeError("каждый элемент paths должен быть экземпляром FilterPath")
