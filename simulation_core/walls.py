"""
Duz duvarlar (hareket engeli).

Duvar bir dogru parcasidir. Bir drone'un bu adimdaki hareketi bir duvari
keserse (ya da ona degerse) drone o adim yerinde kalir. Duvarlar
haberlesmeyi etkilemez ve algoritmaya (SimulationContext) hic verilmez;
ajan yalnizca kendi hareketinden etkisini hissedebilir.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
_EPS = 1e-9


@dataclass(frozen=True)
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class WallConfig:
    enabled: bool = True
    count_min: int = 2
    count_max: int = 4
    length_min: float = 80.0
    length_max: float = 250.0
    clearance: float = 25.0  # drone/hedef/ground'a en az bu kadar uzak
    canvas_width: int = 960
    canvas_height: int = 760
    regenerate_on_start: bool = True

    def validate(self) -> None:
        if self.count_min < 0 or self.count_max < self.count_min:
            raise ValueError("Duvar sayisi araligi gecersiz.")
        if self.length_min <= 0 or self.length_max < self.length_min:
            raise ValueError("Duvar uzunluk araligi gecersiz.")
        if self.clearance < 0:
            raise ValueError("Duvar mesafesi negatif olamaz.")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("Canvas boyutu pozitif olmali.")


# ---------------------------------------------------------------------- #
# Geometri                                                               #
# ---------------------------------------------------------------------- #

def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (
        min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS
        and min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS
    )


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Kapali dogru parcalari kesisiyor ya da degiyor mu."""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and (
        (d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)
    ):
        return True
    if abs(d1) <= _EPS and _on_segment(q1, q2, p1):
        return True
    if abs(d2) <= _EPS and _on_segment(q1, q2, p2):
        return True
    if abs(d3) <= _EPS and _on_segment(p1, p2, q1):
        return True
    if abs(d4) <= _EPS and _on_segment(p1, p2, q2):
        return True
    return False


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= _EPS:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def movement_blocked(walls: Sequence[Wall], start: Point, end: Point) -> bool:
    if start == end:
        return False
    for w in walls:
        if segments_intersect(start, end, (w.x1, w.y1), (w.x2, w.y2)):
            return True
    return False


# ---------------------------------------------------------------------- #
# Uretim                                                                 #
# ---------------------------------------------------------------------- #

def generate_walls(
    rng: random.Random,
    config: WallConfig,
    avoid_points: Sequence[Point] = (),
) -> List[Wall]:
    """Canvas icinde, verilen noktalara (drone, hedef, ground) en az
    clearance uzaklikta rastgele duz duvarlar uretir."""
    config.validate()
    if not config.enabled:
        return []
    w, h = float(config.canvas_width), float(config.canvas_height)
    count = rng.randint(config.count_min, config.count_max)
    walls: List[Wall] = []
    for _ in range(count):
        for _ in range(400):
            length = rng.uniform(config.length_min, config.length_max)
            angle = rng.uniform(0.0, math.pi)
            cx, cy = rng.uniform(0.0, w), rng.uniform(0.0, h)
            dx, dy = 0.5 * length * math.cos(angle), 0.5 * length * math.sin(angle)
            a = (cx - dx, cy - dy)
            b = (cx + dx, cy + dy)
            if not (0.0 <= a[0] <= w and 0.0 <= b[0] <= w and 0.0 <= a[1] <= h and 0.0 <= b[1] <= h):
                continue
            if any(point_segment_distance(p, a, b) < config.clearance for p in avoid_points):
                continue
            walls.append(Wall(a[0], a[1], b[0], b[1]))
            break
    return walls
