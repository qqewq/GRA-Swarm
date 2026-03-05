"""
Hierarchy Levels / Уровни Иерархии
Реализация многоуровневой архитектуры GRA из мультиверса.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class LevelType(IntEnum):
    """Типы уровней иерархии."""
    LOCAL = 0      # Уровень 0: Локальные домены
    META = 1       # Уровень 1: Мета-системы
    META_META = 2  # Уровень 2: Мета-мета-системы
    MULTIVERSE = 3 # Уровень K: Мультиверс


@dataclass
class HierarchyLevel:
    """
    Уровень иерархии GRA.
    GRA hierarchy level.
    """
    level_id: int
    level_type: LevelType
    n_subsystems: int
    goal_dimension: int
    weight: float = 1.0
    
    # Цели уровня
    goals: Optional[np.ndarray] = None
    
    # Состояния подсистем
    states: List[np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        if self.states and len(self.states) != self.n_subsystems:
            raise ValueError(
                f"Number of states ({len(self.states)}) != "
                f"n_subsystems ({self.n_subsystems})"
            )
    
    def compute_level_foam(
        self, 
        projector: 'Projector'
    ) -> float:
        """
        Вычисление пены уровня.
        
        Φ^(l) = Σ_{a≠b} |⟨Ψ^(a)|P_Gl|Ψ^(b)⟩|^2
        """
        if len(self.states) < 2:
            return 0.0
        
        foam = 0.0
        for i in range(len(self.states)):
            for j in range(i + 1, len(self.states)):
                projected = projector.apply(self.states[i])
                overlap = np.abs(np.dot(projected, self.states[j])) ** 2
                foam += overlap
        
        return foam / (len(self.states) * (len(self.states) - 1))


@dataclass
class MultiLevelSystem:
    """
    Многоуровневая система GRA.
    Multi-level GRA system.
    """
    n_levels: int
    levels: List[HierarchyLevel] = field(default_factory=list)
    alpha: float = 0.7  # Коэффициент затухания
    lambda_0: float = 1.0
    
    def __post_init__(self):
        if not self.levels:
            self.levels = self._create_default_levels()
    
    def _create_default_levels(self) -> List[HierarchyLevel]:
        """Создание уровней по умолчанию."""
        levels = []
        for l in range(self.n_levels):
            level_type = LevelType(l) if l <= 3 else LevelType.MULTIVERSE
            weight = self.lambda_0 * (self.alpha ** l)
            
            level = HierarchyLevel(
                level_id=l,
                level_type=level_type,
                n_subsystems=4,  # По умолчанию
                goal_dimension=10,
                weight=weight
            )
            levels.append(level)
        
        return levels
    
    def compute_total_foam(
        self, 
        projectors: List['Projector']
    ) -> float:
        """
        Вычисление полной пены мультиверса.
        
        J_multiverse = Σ_l Λ_l · Φ^(l)
        """
        total_foam = 0.0
        
        for l, level in enumerate(self.levels):
            if l < len(projectors):
                level_foam = level.compute_level_foam(projectors[l])
                total_foam += level.weight * level_foam
        
        return total_foam
    
    def get_level_weights(self) -> List[float]:
        """Получить веса уровней."""
        return [level.weight for level in self.levels]
    
    def initialize_states(self, state_dim: int):
        """Инициализация состояний всех уровней."""
        for level in self.levels:
            level.states = [
                np.random.randn(state_dim) for _ in range(level.n_subsystems)
            ]