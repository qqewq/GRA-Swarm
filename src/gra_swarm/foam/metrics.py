"""
Foam Metrics / Метрики Пены
Дополнительные метрики для анализа роя.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FoamMetrics:
    """Метрики пены для одного агента."""
    individual_foam: float = 0.0
    optimal_distance: float = 0.0
    entropy: float = 0.0
    consistency: float = 1.0


@dataclass
class SwarmMetrics:
    """
    Метрики пены для всего роя.
    Foam metrics for entire swarm.
    """
    total_foam: float = 0.0
    individual_foam: float = 0.0
    coordination_loss: float = 0.0
    diversity_bonus: float = 0.0
    diversity_raw: float = 0.0
    lambda_diversity: float = 0.1
    
    # Статистика по агентам
    agent_foams: List[float] = field(default_factory=list)
    agent_entropies: List[float] = field(default_factory=list)
    
    # История
    history: List[float] = field(default_factory=list)
    
    def add_to_history(self, foam: float):
        """Добавить значение в историю."""
        self.history.append(foam)
    
    def get_statistics(self) -> Dict[str, float]:
        """Получить статистику метрик."""
        if len(self.history) < 2:
            return {
                'mean': self.total_foam,
                'std': 0.0,
                'min': self.total_foam,
                'max': self.total_foam
            }
        
        return {
            'mean': float(np.mean(self.history)),
            'std': float(np.std(self.history)),
            'min': float(np.min(self.history)),
            'max': float(np.max(self.history)),
            'trend': float(self.history[-1] - self.history[0])
        }
    
    def is_converged(self, threshold: float = 0.001, window: int = 10) -> bool:
        """Проверка сходимости."""
        if len(self.history) < window:
            return False
        
        recent = self.history[-window:]
        return float(np.std(recent)) < threshold
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            'total_foam': self.total_foam,
            'individual_foam': self.individual_foam,
            'coordination_loss': self.coordination_loss,
            'diversity_bonus': self.diversity_bonus,
            'diversity_raw': self.diversity_raw,
            'statistics': self.get_statistics()
        }


class MetricsCollector:
    """
    Коллектор метрик для мониторинга обучения.
    Metrics collector for training monitoring.
    """
    
    def __init__(self):
        self.swarm_metrics = SwarmMetrics()
        self.agent_metrics: Dict[int, FoamMetrics] = {}
        self.episode = 0
    
    def record_episode(
        self, 
        foam_total: float,
        agent_foams: List[float],
        diversity: float
    ):
        """Запись метрик эпизода."""
        self.swarm_metrics.add_to_history(foam_total)
        self.swarm_metrics.total_foam = foam_total
        self.swarm_metrics.diversity_raw = diversity
        self.agent_metrics = {
            i: FoamMetrics(individual_foam=f)
            for i, f in enumerate(agent_foams)
        }
        self.episode += 1
    
    def get_summary(self) -> Dict:
        """Получить сводку метрик."""
        return {
            'episode': self.episode,
            'swarm': self.swarm_metrics.to_dict(),
            'converged': self.swarm_metrics.is_converged()
        }
    
    def export_to_csv(self, filepath: str):
        """Экспорт истории в CSV."""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'total_foam', 'diversity'])
            
            for i, foam in enumerate(self.swarm_metrics.history):
                diversity = self.swarm_metrics.diversity_raw
                writer.writerow([i, foam, diversity])