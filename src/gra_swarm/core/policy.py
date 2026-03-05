"""
Policy Network / Сеть Политики
Вероятностные политики для агентов.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    """Конфигурация политики / Policy configuration."""
    input_dim: int
    output_dim: int
    hidden_dim: int = 64
    activation: str = "relu"
    softmax_temperature: float = 1.0


class PolicyNetwork:
    """
    Нейросетевая политика агента.
    Neural network policy for agents.
    """
    
    def __init__(self, config: PolicyConfig, seed: int = 42):
        self.config = config
        np.random.seed(seed)
        
        # Инициализация весов
        self.W1 = np.random.randn(config.input_dim, config.hidden_dim) * 0.1
        self.b1 = np.zeros(config.hidden_dim)
        self.W2 = np.random.randn(config.hidden_dim, config.output_dim) * 0.1
        self.b2 = np.zeros(config.output_dim)
        
        self.temperature = config.softmax_temperature
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через сеть."""
        # Скрытый слой
        hidden = np.dot(x, self.W1) + self.b1
        hidden = self._activate(hidden, self.config.activation)
        
        # Выходной слой
        logits = np.dot(hidden, self.W2) + self.b2
        
        # Softmax с температурой
        probs = self._softmax(logits / self.temperature)
        
        return probs
    
    def _activate(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Функция активации."""
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Стабильный softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def get_weights(self) -> np.ndarray:
        """Получить все веса как вектор."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
    
    def set_weights(self, weights: np.ndarray):
        """Установить веса из вектора."""
        idx = 0
        
        size = self.W1.size
        self.W1 = weights[idx:idx+size].reshape(self.W1.shape)
        idx += size
        
        size = self.b1.size
        self.b1 = weights[idx:idx+size].reshape(self.b1.shape)
        idx += size
        
        size = self.W2.size
        self.W2 = weights[idx:idx+size].reshape(self.W2.shape)
        idx += size
        
        size = self.b2.size
        self.b2 = weights[idx:idx+size].reshape(self.b2.shape)
    
    def compute_gradient(
        self, 
        x: np.ndarray, 
        action: int, 
        advantage: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Вычисление градиентов для обновления."""
        # Прямой проход с сохранением промежуточных значений
        hidden = np.dot(x, self.W1) + self.b1
        hidden_act = self._activate(hidden, self.config.activation)
        logits = np.dot(hidden_act, self.W2) + self.b2
        probs = self._softmax(logits / self.temperature)
        
        # Градиент по выходному слою
        d_logits = probs.copy()
        d_logits[action] -= 1
        d_logits *= advantage
        
        dW2 = np.dot(hidden_act.T, d_logits)
        db2 = d_logits.sum(axis=0)
        
        # Градиент по скрытому слою
        d_hidden = np.dot(d_logits, self.W2.T)
        d_hidden[hidden <= 0] = 0  # ReLU градиент
        
        dW1 = np.dot(x.T, d_hidden)
        db1 = d_hidden.sum(axis=0)
        
        return dW1, db1, dW2, db2


class ProbabilisticPolicy:
    """
    Вероятностная политика с семплированием.
    Probabilistic policy with sampling.
    """
    
    def __init__(self, network: PolicyNetwork):
        self.network = network
        self.action_history = []
    
    def select_action(self, state: np.ndarray) -> int:
        """Выбор действия согласно политике."""
        probs = self.network.forward(state)
        action = np.random.choice(len(probs), p=probs)
        self.action_history.append((state.copy(), action, probs.copy()))
        return action
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        """Получить распределение действий."""
        return self.network.forward(state)
    
    def compute_entropy(self, state: np.ndarray) -> float:
        """Вычисление энтропии политики."""
        probs = self.get_action_distribution(state)
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))
    
    def clear_history(self):
        """Очистка истории действий."""
        self.action_history = []