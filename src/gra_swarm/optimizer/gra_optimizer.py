"""
GRA Optimizer / GRA-Оптимизатор
Оптимизатор для минимизации функционала пены.
"""

import numpy as np
from typing import List, Optional
from gra_swarm.core.agent import GRAAgent
from gra_swarm.foam.calculator import FoamCalculator


class GRAOptimizer:
    """
    Оптимизатор на основе градиентного спуска.
    
    Ψ(t+1) = Ψ(t) - η · ∇J_swarm
    """
    
    def __init__(
        self, 
        agents: List[GRAAgent],
        foam_calculator: Optional[FoamCalculator] = None,
        learning_rate: float = 1e-3
    ):
        self.agents = agents
        self.foam_calculator = foam_calculator or FoamCalculator()
        self.learning_rate = learning_rate
        self.foam_history = []
    
    def compute_gradient(
        self, 
        agent: GRAAgent, 
        reward: float,
        foam_total: float
    ) -> np.ndarray:
        """
        Вычисление градиента для агента.
        ∇J = ∇Φ_agent + ∇Φ_coordination - λ·∇I
        """
        # Простая аппроксимация градиента через reward
        gradient = np.random.randn(*agent.policy_weights.shape)
        gradient *= (1.0 - reward + foam_total)
        
        return gradient
    
    def step(
        self, 
        reward: float, 
        foam_total: float,
        collective_state: Optional[np.ndarray] = None
    ):
        """Один шаг оптимизации для всех агентов."""
        for agent in self.agents:
            gradient = self.compute_gradient(agent, reward, foam_total)
            agent.update_policy(gradient, self.learning_rate)
        
        self.foam_history.append(foam_total)
    
    def get_foam_history(self) -> List[float]:
        """История пены за все эпизоды."""
        return self.foam_history.copy()
    
    def reset(self):
        """Сброс оптимизатора."""
        self.foam_history = []
        for agent in self.agents:
            agent.reset_trajectory()