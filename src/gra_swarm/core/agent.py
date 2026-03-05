"""
GRA Agent Module / Модуль Агента GRA
Базовый класс ИИ-агента с индивидуальностью.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Конфигурация агента / Agent configuration."""
    agent_id: int
    personality_seed: int
    action_space_dim: int
    learning_rate: float = 1e-3
    level: int = 0
    hidden_dim: int = 64


class GRAAgent:
    """
    ИИ-агент с вероятностной политикой.
    AI agent with probabilistic policy.
    
    Attributes:
        policy: Вероятностное распределение действий / Action probability distribution
        optimal_dist: Идеальное распределение P_i^opt / Ideal distribution
        foam: Текущая пена агента / Current agent foam
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        
        # Инициализация с уникальным зерном личности
        # Initialize with unique personality seed
        np.random.seed(config.personality_seed)
        
        self.policy_weights = np.random.randn(
            config.hidden_dim, 
            config.action_space_dim
        ) * 0.1
        
        self.hidden_weights = np.random.randn(
            config.action_space_dim,
            config.hidden_dim
        ) * 0.1
        
        self.optimal_dist: Optional[np.ndarray] = None
        self.foam: float = float('inf')
        self.trajectory_history: List[Tuple] = []
    
    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """Возвращает распределение вероятностей действий."""
        hidden = np.dot(state, self.hidden_weights)
        hidden = np.maximum(0, hidden)  # ReLU
        logits = np.dot(hidden, self.policy_weights)
        return self._softmax(logits)
    
    def act(self, state: np.ndarray) -> int:
        """Выбор действия согласно политике."""
        probs = self.get_policy(state)
        action = np.random.choice(len(probs), p=probs)
        self.trajectory_history.append((state.copy(), action, probs.copy()))
        return action
    
    def compute_foam(self, trajectory: Optional[List] = None) -> float:
        """Вычисляет индивидуальную пену агента Φ_agent."""
        from gra_swarm.foam.diversity import kl_divergence
        
        if self.optimal_dist is None:
            raise ValueError("optimal_dist не установлен / not set")
        
        if trajectory is None:
            trajectory = self.trajectory_history
        
        if len(trajectory) == 0:
            return 0.0
        
        # Используем последнее состояние
        last_state = trajectory[-1][0]
        Q_i = self.get_policy(last_state)
        P_opt = self.optimal_dist
        
        self.foam = kl_divergence(Q_i, P_opt)
        return self.foam
    
    def update_policy(self, gradient: np.ndarray, lr: Optional[float] = None):
        """Обновление политики: Ψ(t+1) = Ψ(t) - η · ∇J"""
        learning_rate = lr if lr is not None else self.config.learning_rate
        self.policy_weights -= learning_rate * gradient
    
    def set_optimal_distribution(self, optimal_dist: np.ndarray):
        """Установка идеального распределения P_i^opt."""
        self.optimal_dist = optimal_dist / optimal_dist.sum()
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Стабильный softmax / Stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def reset_trajectory(self):
        """Очистка истории траекторий."""
        self.trajectory_history = []
    
    def __repr__(self):
        return f"GRAAgent(id={self.agent_id}, L{self.config.level}, foam={self.foam:.4f})"