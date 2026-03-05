"""
Learning Rate Scheduler / Планировщик Скорости Обучения
Динамическая настройка гиперпараметров обучения.
"""

import numpy as np
from typing import List, Optional, Callable
from enum import Enum


class SchedulerType(Enum):
    """Типы планировщиков."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"


class LearningRateScheduler:
    """
    Планировщик скорости обучения.
    Learning rate scheduler.
    """
    
    def __init__(
        self, 
        scheduler_type: SchedulerType = SchedulerType.COSINE,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6,
        total_steps: int = 1000,
        **kwargs
    ):
        self.scheduler_type = scheduler_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.kwargs = kwargs
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Получить скорость обучения для шага."""
        if step is not None:
            self.current_step = step
        
        progress = self.current_step / max(self.total_steps, 1)
        
        if self.scheduler_type == SchedulerType.CONSTANT:
            lr = self.initial_lr
        
        elif self.scheduler_type == SchedulerType.LINEAR:
            lr = self.initial_lr * (1 - progress)
        
        elif self.scheduler_type == SchedulerType.EXPONENTIAL:
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            lr = self.initial_lr * (decay_rate ** self.current_step)
        
        elif self.scheduler_type == SchedulerType.STEP:
            step_size = self.kwargs.get('step_size', 100)
            gamma = self.kwargs.get('gamma', 0.5)
            lr = self.initial_lr * (gamma ** (self.current_step // step_size))
        
        elif self.scheduler_type == SchedulerType.COSINE:
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))
        
        elif self.scheduler_type == SchedulerType.ADAPTIVE:
            # Адаптивный на основе градиентов
            lr = self._adaptive_lr()
        
        else:
            lr = self.initial_lr
        
        return max(lr, self.min_lr)
    
    def _adaptive_lr(self) -> float:
        """Адаптивная скорость обучения."""
        # Может быть расширено для мониторинга градиентов
        return self.initial_lr * (0.99 ** self.current_step)
    
    def step(self):
        """Шаг планировщика."""
        self.current_step += 1
    
    def reset(self):
        """Сброс планировщика."""
        self.current_step = 0


class TrainingScheduler:
    """
    Планировщик обучения роя.
    Training scheduler for swarm.
    """
    
    def __init__(
        self,
        n_episodes: int = 100,
        foam_threshold: float = 0.001,
        early_stopping_patience: int = 20,
        checkpoint_every: int = 10
    ):
        self.n_episodes = n_episodes
        self.foam_threshold = foam_threshold
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_every = checkpoint_every
        
        self.best_foam = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def check_early_stopping(self, current_foam: float) -> bool:
        """Проверка ранней остановки."""
        if current_foam < self.best_foam:
            self.best_foam = current_foam
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        self.should_stop = self.patience_counter >= self.early_stopping_patience
        return self.should_stop
    
    def check_convergence(self, foam_history: List[float]) -> bool:
        """Проверка сходимости."""
        if len(foam_history) < 10:
            return False
        
        recent = foam_history[-10:]
        return np.std(recent) < self.foam_threshold
    
    def should_checkpoint(self, episode: int) -> bool:
        """Проверка необходимости сохранения."""
        return episode % self.checkpoint_every == 0
    
    def get_phase(self, episode: int) -> str:
        """Получить фазу обучения."""
        progress = episode / self.n_episodes
        
        if progress < 0.3:
            return "exploration"
        elif progress < 0.7:
            return "optimization"
        else:
            return "convergence"