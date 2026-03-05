"""
GRA-Swarm: Роевой Сверхинтеллект через Минимизацию Когнитивной Пены
Swarm Superintelligence via Cognitive Foam Minimization

Version: 0.1.0
Author: YOUR_NAME
"""

__version__ = "0.1.0"
__author__ = "YOUR_NAME"
__email__ = "your.email@example.com"

from gra_swarm.core.agent import GRAAgent, AgentConfig
from gra_swarm.foam.calculator import FoamCalculator
from gra_swarm.foam.diversity import compute_diversity, kl_divergence
from gra_swarm.optimizer.gra_optimizer import GRAOptimizer

__all__ = [
    "GRAAgent",
    "AgentConfig",
    "FoamCalculator",
    "compute_diversity",
    "kl_divergence",
    "GRAOptimizer",
]