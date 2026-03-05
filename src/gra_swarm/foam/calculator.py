"""
Foam Calculator / Калькулятор Пены
Вычисление функционала пены для роя агентов.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from gra_swarm.core.agent import GRAAgent
from gra_swarm.foam.diversity import compute_diversity


class FoamCalculator:
    """
    Вычисление полной пены роя Φ_swarm.
    
    Φ_swarm = (1/N)ΣΦ_agent + d(Ξ, P_G(Ξ)) - λ·I
    """
    
    def __init__(
        self, 
        lambda_div: float = 0.1, 
        coordination_weight: float = 1.0
    ):
        self.lambda_div = lambda_div
        self.coordination_weight = coordination_weight
    
    def compute_swarm_foam(
        self, 
        agents: List[GRAAgent], 
        collective_state: np.ndarray,
        projector: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Вычисление полной пены роя."""
        N = len(agents)
        
        # 1. Индивидуальная пена
        individual_foam = np.mean([
            agent.compute_foam() for agent in agents
        ])
        
        # 2. Коллективная несогласованность
        coordination_loss = 0.0
        if projector is not None:
            projected_state = projector(collective_state)
            coordination_loss = float(np.linalg.norm(
                collective_state - projected_state
            ))
        
        # 3. Разнообразие (бонус)
        test_state = np.random.randn(agents[0].config.action_space_dim)
        policies = [agent.get_policy(test_state) for agent in agents]
        diversity_bonus = compute_diversity(policies)
        
        # Итоговый функционал
        foam_total = (
            individual_foam + 
            self.coordination_weight * coordination_loss - 
            self.lambda_div * diversity_bonus
        )
        
        return {
            'total': float(foam_total),
            'individual': float(individual_foam),
            'coordination': float(coordination_loss),
            'diversity': float(diversity_bonus),
            'lambda': self.lambda_div
        }
    
    def compute_level_foam(
        self,
        agents: List[GRAAgent],
        level: int,
        target_projector: Callable
    ) -> float:
        """Пена для конкретного уровня иерархии l."""
        foam_level = 0.0
        N = len(agents)
        
        if N < 2:
            return 0.0
        
        test_state = np.random.randn(agents[0].config.action_space_dim)
        
        for i in range(N):
            for j in range(i + 1, N):
                psi_i = agents[i].get_policy(test_state)
                psi_j = agents[j].get_policy(test_state)
                
                projected = target_projector(psi_i)
                overlap = float(np.abs(np.dot(projected, psi_j)) ** 2)
                foam_level += overlap
        
        return foam_level / (N * (N - 1))