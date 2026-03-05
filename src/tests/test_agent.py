"""
Tests for Agent Module / Тесты Модуля Агентов
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gra_swarm.core.agent import GRAAgent, AgentConfig
from gra_swarm.core.personality import PersonalityGenerator, PersonalityType


class TestGRAAgent:
    """Тесты для GRAAgent."""
    
    @pytest.fixture
    def agent(self):
        config = AgentConfig(
            agent_id=1,
            personality_seed=42,
            action_space_dim=10
        )
        return GRAAgent(config)
    
    def test_agent_initialization(self, agent):
        """Тест инициализации агента."""
        assert agent.agent_id == 1
        assert agent.foam == float('inf')
        assert len(agent.trajectory_history) == 0
    
    def test_agent_act(self, agent):
        """Тест действия агента."""
        state = np.random.randn(10)
        action = agent.act(state)
        
        assert isinstance(action, int)
        assert 0 <= action < 10
        assert len(agent.trajectory_history) == 1
    
    def test_agent_policy(self, agent):
        """Тест политики агента."""
        state = np.random.randn(10)
        policy = agent.get_policy(state)
        
        assert policy.shape == (10,)
        assert np.isclose(policy.sum(), 1.0)
        assert np.all(policy >= 0)
    
    def test_agent_foam_computation(self, agent):
        """Тест вычисления пены."""
        agent.set_optimal_distribution(np.ones(10) / 10)
        
        state = np.random.randn(10)
        agent.act(state)
        
        foam = agent.compute_foam()
        assert foam >= 0
    
    def test_agent_personality_diversity(self):
        """Тест разнообразия личностей."""
        gen = PersonalityGenerator(seed=42)
        configs = gen.generate_diverse_swarm(6)
        
        # Все 6 типов должны присутствовать
        types = [c.personality_type for c in configs]
        assert len(set(types)) == 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])