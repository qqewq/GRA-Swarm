"""
GRA-Swarm Foam Module / Модуль Пены
Расчёт пены роя и метрики разнообразия.
"""

from gra_swarm.foam.calculator import FoamCalculator
from gra_swarm.foam.diversity import (
    compute_diversity,
    kl_divergence,
    symmetric_kl_divergence,
    jensen_shannon_divergence
)
from gra_swarm.foam.metrics import FoamMetrics, SwarmMetrics

__all__ = [
    "FoamCalculator",
    "compute_diversity",
    "kl_divergence",
    "symmetric_kl_divergence",
    "jensen_shannon_divergence",
    "FoamMetrics",
    "SwarmMetrics",
]