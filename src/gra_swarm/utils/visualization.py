"""
Visualization Tools / Инструменты Визуализации
Визуализация результатов экспериментов.
"""

import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FoamVisualizer:
    """Визуализатор пены роя."""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8')
    
    def plot_foam_convergence(
        self, 
        foam_history: List[float],
        title: str = "Convergence of Swarm Foam",
        save: bool = True
    ):
        """Плот сходимости пены."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(foam_history, linewidth=2, label='Φ_swarm')
        
        # Скользящее среднее
        if len(foam_history) >= 10:
            moving_avg = np.convolve(
                foam_history, 
                np.ones(10)/10, 
                mode='valid'
            )
            plt.plot(
                range(9, len(foam_history)), 
                moving_avg, 
                '--', 
                linewidth=2, 
                label='Moving Average (10)'
            )
        
        plt.xlabel('Episode')
        plt.ylabel('Foam')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save:
            plt.savefig(self.save_dir / 'foam_convergence.png', dpi=150)
        
        plt.close()
    
    def plot_foam_components(
        self,
        individual: List[float],
        coordination: List[float],
        diversity: List[float],
        total: List[float],
        save: bool = True
    ):
        """Плот компонентов пены."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(total, linewidth=2)
        axes[0, 0].set_title('Total Foam')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(individual, 'b-', linewidth=2)
        axes[0, 1].set_title('Individual Foam')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(coordination, 'g-', linewidth=2)
        axes[1, 0].set_title('Coordination Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(diversity, 'r-', linewidth=2)
        axes[1, 1].set_title('Diversity Bonus')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'foam_components.png', dpi=150)
        
        plt.close()


class SwarmVisualizer:
    """Визуализатор роя агентов."""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_agent_policies(
        self,
        policies: List[np.ndarray],
        agent_names: Optional[List[str]] = None,
        save: bool = True
    ):
        """Плот политик агентов."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        n_agents = len(policies)
        n_actions = len(policies[0])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_actions)
        width = 0.8 / n_agents
        
        for i, policy in enumerate(policies):
            name = agent_names[i] if agent_names else f'Agent {i}'
            ax.bar(
                x + i * width, 
                policy, 
                width, 
                label=name,
                alpha=0.7
            )
        
        ax.set_xlabel('Action')
        ax.set_ylabel('Probability')
        ax.set_title('Agent Policies Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.save_dir / 'agent_policies.png', dpi=150)
        
        plt.close()
    
    def plot_diversity_heatmap(
        self,
        diversity_matrix: np.ndarray,
        save: bool = True
    ):
        """Тепловая карта разнообразия."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            diversity_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            square=True
        )
        plt.title('Pairwise Diversity Matrix')
        plt.xlabel('Agent')
        plt.ylabel('Agent')
        
        if save:
            plt.savefig(self.save_dir / 'diversity_heatmap.png', dpi=150)
        
        plt.close()


def plot_foam_convergence(foam_history: List[float], save_path: str = None):
    """Утилита для быстрого плота."""
    viz = FoamVisualizer()
    viz.plot_foam_convergence(foam_history, save=(save_path is not None))


def plot_diversity(policies: List[np.ndarray], save_path: str = None):
    """Утилита для плота разнообразия."""
    viz = SwarmVisualizer()
    viz.plot_agent_policies(policies, save=(save_path is not None))