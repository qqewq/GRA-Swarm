"""
Projector Operators / Операторы Проекторов
Проекторы на пространства решений целей.
"""

import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod


class Projector(ABC):
    """
    Базовый класс проектора.
    Base projector class.
    """
    
    @abstractmethod
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Применить проектор к состоянию."""
        pass
    
    @abstractmethod
    def is_commuting(self, other: 'Projector') -> bool:
        """Проверка коммутативности с другим проектором."""
        pass


class GoalProjector(Projector):
    """
    Проектор на пространство решений цели.
    Projector onto goal solution space.
    """
    
    def __init__(
        self, 
        goal_space: np.ndarray,
        tolerance: float = 1e-6
    ):
        """
        Args:
            goal_space: Базис пространства цели
            tolerance: Численная точность
        """
        self.goal_space = goal_space
        self.tolerance = tolerance
        
        # QR-разложение для ортонормализации
        if goal_space.ndim == 2:
            Q, _ = np.linalg.qr(goal_space)
            self.orthonormal_basis = Q
        else:
            self.orthonormal_basis = goal_space.reshape(-1, 1)
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Проекция состояния на пространство цели.
        
        P_G |Ψ⟩ = Σ_i |e_i⟩⟨e_i|Ψ⟩
        """
        if self.orthonormal_basis.ndim == 1:
            basis = self.orthonormal_basis.reshape(-1, 1)
        else:
            basis = self.orthonormal_basis
        
        # Проекция: P = B(B^T B)^{-1} B^T
        coefficients = np.dot(basis.T, state)
        projected = np.dot(basis, coefficients)
        
        return projected
    
    def is_commuting(self, other: 'Projector', tolerance: float = 1e-6) -> bool:
        """
        Проверка коммутативности: [P1, P2] = 0
        """
        if not isinstance(other, GoalProjector):
            return False
        
        # Создаём тестовые векторы
        test_vectors = [
            np.random.randn(len(self.orthonormal_basis))
            for _ in range(10)
        ]
        
        for v in test_vectors:
            # P1(P2(v)) - P2(P1(v))
            result1 = self.apply(other.apply(v))
            result2 = other.apply(self.apply(v))
            
            if np.linalg.norm(result1 - result2) > tolerance:
                return False
        
        return True
    
    def compute_projection_error(self, state: np.ndarray) -> float:
        """Ошибка проекции."""
        projected = self.apply(state)
        return float(np.linalg.norm(state - projected))


class HierarchicalProjector(Projector):
    """
    Иерархический проектор для мультиверса.
    Hierarchical projector for multiverse.
    """
    
    def __init__(self, projectors: List[GoalProjector]):
        self.projectors = projectors
        self.n_levels = len(projectors)
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Последовательное применение проекторов."""
        result = state
        for projector in self.projectors:
            result = projector.apply(result)
        return result
    
    def apply_by_level(
        self, 
        state: np.ndarray, 
        level: int
    ) -> np.ndarray:
        """Применить проектор конкретного уровня."""
        if level >= self.n_levels:
            raise ValueError(f"Level {level} >= n_levels {self.n_levels}")
        
        return self.projectors[level].apply(state)
    
    def is_commuting(self, other: 'Projector') -> bool:
        """Проверка коммутативности всех проекторов."""
        if not isinstance(other, HierarchicalProjector):
            return False
        
        if self.n_levels != other.n_levels:
            return False
        
        for p1, p2 in zip(self.projectors, other.projectors):
            if not p1.is_commuting(p2):
                return False
        
        return True


class IdentityProjector(Projector):
    """Тождественный проектор (для базовых случаев)."""
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        return state.copy()
    
    def is_commuting(self, other: 'Projector') -> bool:
        return True