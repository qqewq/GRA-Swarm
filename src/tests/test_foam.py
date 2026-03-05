"""
Tests for Foam Module / Тесты Модуля Пены
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gra_swarm.foam.diversity import (
    kl_divergence,
    symmetric_kl_divergence,
    compute_diversity
)
from gra_swarm.foam.calculator import FoamCalculator
from gra_swarm.core.agent import GRAAgent, AgentConfig


class TestDiversity:
    """Тесты для метрик разнообразия."""
    
    def test_kl_divergence_same(self):
        """KL дивергенция одинаковых распределений = 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) < 1e-10
    
    def test_kl_divergence_positive(self):
        """KL дивергенция всегда >= 0."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.3, 0.3, 0.4])
        assert kl_divergence(p, q) >= 0
    
    def test_symmetric_kl(self):
        """Симметричная KL дивергенция."""
        p = np.array([0.6, 0.4])
        q = np.array([0.4, 0.6])
        
        skl = symmetric_kl_divergence(p, q)
        assert skL >= 0
    
    def test_compute_diversity(self):
        """Вычисление разнообразия роя."""
        policies = [
            np.array([0.7, 0.3]),
            np.array([0.3, 0.7]),
            np.array([0.5, 0.5])
        ]
        
        diversity = compute_diversity(policies)
        assert diversity >= 0


class TestFoamCalculator:
    """Тесты для калькулятора пены."""
    
    @pytest.fixture
    def agents(self):
        agents = []
        for i in range(4):
            config = AgentConfig(
                agent_id=i,
                personality_seed=42 + i * 17,
                action_space_dim=10
            )
            agent = GRAAgent(config)
            agent.set_optimal_distribution(np.ones(10) / 10)
            agents.append(agent)
        return agents
    
    def test_swarm_foam_computation(self, agents):
        """Тест вычисления пены роя."""
        calculator = FoamCalculator(lambda_div=0.1)
        collective_state = np.random.randn(10)
        
        metrics = calculator.compute_swarm_foam(agents, collective_state)
        
        assert 'total' in metrics
        assert 'individual' in metrics
        assert 'diversity' in metrics
        assert metrics['total'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])