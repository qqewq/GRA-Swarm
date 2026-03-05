"""
Diversity Metrics / Меры Разнообразия
Меры индивидуальности политик агентов I^(l).
"""

import numpy as np
from typing import List, Union


def kl_divergence(
    p: np.ndarray, 
    q: np.ndarray, 
    epsilon: float = 1e-10
) -> float:
    """
    KL-дивергенция D_KL(p || q).
    
    Args:
        p: Распределение P
        q: Распределение Q
        epsilon: Для численной стабильности
        
    Returns:
        D_KL(p || q)
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Симметричная KL-дивергенция: D(p,q) = KL(p||q) + KL(q||p)"""
    return kl_divergence(p, q) + kl_divergence(q, p)


def compute_diversity(policies: List[np.ndarray]) -> float:
    """
    Мера индивидуальности роя I^(l).
    
    I^(l) = (1/N(N-1)) Σ_{i≠j} D_ij
    
    Args:
        policies: Список распределений [Q_1, Q_2, ..., Q_N]
        
    Returns:
        Мера разнообразия (чем больше, тем лучше)
    """
    N = len(policies)
    if N < 2:
        return 0.0
    
    diversity_sum = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            D_ij = symmetric_kl_divergence(policies[i], policies[j])
            diversity_sum += D_ij
    
    return diversity_sum / (N * (N - 1))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS-дивергенция (альтернатива симметричной KL)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def variance_diversity(policies: List[np.ndarray]) -> float:
    """Разнообразие через дисперсию от среднего."""
    if len(policies) < 2:
        return 0.0
    
    mean_policy = np.mean(policies, axis=0)
    variance = np.mean([np.sum((p - mean_policy) ** 2) for p in policies])
    return float(variance)