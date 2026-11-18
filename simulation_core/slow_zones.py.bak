"""
Yavaslatma bolgeleri.

Daire seklindeki bolgelerin icindeki drone sabit bir oranla yavaslar.
Bolgeler algoritmaya (SimulationContext) hic verilmez; ajan yalnizca
kendi gerceklesen hareketinden etkisini hissedebilir.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class SlowZone:
    x: float
    y: float
    radius: float

    def contains(self, x: float, y: float) -> bool:
        return math.hypot(x - self.x, y - self.y) <= self.radius


@dataclass(frozen=True)
class SlowZoneConfig:
    enabled: bool = True
    count_min: int = 2
    count_max: int = 4
    radius_min: float = 60.0
    radius_max: float = 140.0
    speed_factor: float = 0.5
    canvas_width: int = 960
    canvas_height: int = 760
    regenerate_on_start: bool = True

    def validate(self) -> None:
        if self.count_min < 0 or self.count_max < self.count_min:
            raise ValueError("Bolge sayisi araligi gecersiz.")
        if self.radius_min <= 0 or self.radius_max < self.radius_min:
            raise ValueError("Bolge yaricap araligi gecersiz.")
        if not 0.0 < self.speed_factor <= 1.0:
            raise ValueError("Yavaslatma orani (0, 1] araliginda olmali.")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Canvas boyutu pozitif olmali.")


def generate_slow_zones(rng: random.Random, config: SlowZoneConfig) -> List[SlowZone]:
    config.validate()
    if not config.enabled:
        return []
    count = rng.randint(config.count_min, config.count_max)
    zones: List[SlowZone] = []
    for _ in range(count):
        r = rng.uniform(config.radius_min, config.radius_max)
        x = rng.uniform(0.0, float(config.canvas_width))
        y = rng.uniform(0.0, float(config.canvas_height))
        zones.append(SlowZone(x=x, y=y, radius=r))
    return zones


def speed_factor_at(
    zones: Sequence[SlowZone], x: float, y: float, factor: float
) -> float:
    """Herhangi bir bolgenin icindeyse sabit oran, degilse 1.0 (ust uste binme carpilmaz)."""
    for zone in zones:
        if zone.contains(x, y):
            return factor
    return 1.0
