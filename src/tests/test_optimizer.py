"""
Tests for Optimizer Module / Тесты Модуля Оптимизатора
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gra_swarm.optimizer.gra_optimizer import GRAOptimizer
from gra_swarm.optimizer.scheduler import LearningRateScheduler, SchedulerType
from gra_swarm.core.agent import GRAAgent, AgentConfig


class TestGRAOptimizer:
    """Тесты для GRAOptimizer."""
    
    @pytest.fixture
    def optimizer_setup(self):
        agents = [
            GRAAgent(AgentConfig(
                agent_id=i,
                personality_seed=42 + i * 17,
                action_space_dim=10
            ))
            for i in range(4)
        ]
        optimizer = GRAOptimizer(agents, learning_rate=0.01)
        return optimizer, agents
    
    def test_optimizer_initialization(self, optimizer_setup):
        """Тест инициализации оптимизатора."""
        optimizer, agents = optimizer_setup
        assert len(optimizer.agents) == 4
        assert len(optimizer.foam_history) == 0
    
    def test_optimizer_step(self, optimizer_setup):
        """Тест шага оптимизатора."""
        optimizer, agents = optimizer_setup
        
        optimizer.step(reward=0.5, foam_total=0.1)
        
        assert len(optimizer.foam_history) == 1
        assert optimizer.foam_history[0] == 0.1
    
    def test_optimizer_reset(self, optimizer_setup):
        """Тест сброса оптимизатора."""
        optimizer, agents = optimizer_setup
        
        optimizer.step(reward=0.5, foam_total=0.1)
        optimizer.reset()
        
        assert len(optimizer.foam_history) == 0


class TestLearningRateScheduler:
    """Тесты для планировщика скорости обучения."""
    
    def test_constant_scheduler(self):
        """Тест постоянного планировщика."""
        scheduler = LearningRateScheduler(
            scheduler_type=SchedulerType.CONSTANT,
            initial_lr=0.001
        )
        
        lr1 = scheduler.get_lr(0)
        scheduler.step()
        lr2 = scheduler.get_lr(1)
        
        assert lr1 == lr2 == 0.001
    
    def test_cosine_scheduler(self):
        """Тест косинусного планировщика."""
        scheduler = LearningRateScheduler(
            scheduler_type=SchedulerType.COSINE,
            initial_lr=0.001,
            min_lr=1e-6,
            total_steps=100
        )
        
        lr_start = scheduler.get_lr(0)
        scheduler.current_step = 50
        lr_mid = scheduler.get_lr(50)
        scheduler.current_step = 100
        lr_end = scheduler.get_lr(100)
        
        assert lr_start >= lr_mid >= lr_end


if __name__ == '__main__':
    pytest.main([__file__, '-v'])