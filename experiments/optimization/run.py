"""
Experiment 2: Multi-Objective Optimization
Эксперимент 2: Многокритериальная Оптимизация
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import List, Dict

from gra_swarm.core.agent import GRAAgent, AgentConfig
from gra_swarm.foam.calculator import FoamCalculator
from gra_swarm.optimizer.gra_optimizer import GRAOptimizer


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimization_problem(n_objectives: int = 3) -> Dict:
    """Создание многокритериальной задачи."""
    # Целевые функции (конфликтующие)
    coefficients = np.random.randn(n_objectives, 10)
    
    return {
        'coefficients': coefficients,
        'n_objectives': n_objectives,
        'state': np.random.randn(10)
    }


def evaluate_objectives(
    action: int, 
    problem: Dict
) -> np.ndarray:
    """Оценка по всем критериям."""
    scores = np.dot(problem['coefficients'], np.eye(10)[:, action])
    return scores


def train_multi_objective_swarm(config: dict) -> dict:
    """Обучение роя на многокритериальной задаче."""
    n_agents = config['agent']['n_agents']
    n_episodes = config['optimization']['n_episodes']
    n_objectives = config['optimization']['n_objectives']
    
    agents = [
        GRAAgent(AgentConfig(
            agent_id=i,
            personality_seed=42 + i * 17,
            action_space_dim=config['agent']['action_space_dim']
        ))
        for i in range(n_agents)
    ]
    
    foam_calculator = FoamCalculator(
        lambda_div=config['swarm']['lambda_diversity'],
        coordination_weight=config['swarm']['coordination_weight']
    )
    
    optimizer = GRAOptimizer(agents, foam_calculator, config['agent']['learning_rate'])
    
    pareto_front = []
    foam_history = []
    
    for episode in range(n_episodes):
        problem = create_optimization_problem(n_objectives)
        
        # Каждый агент предлагает решение
        proposals = [agent.act(problem['state']) for agent in agents]
        
        # Оценка решений
        scores = [evaluate_objectives(p, problem) for p in proposals]
        
        # Нахождение Парето-оптимальных решений
        pareto_indices = find_pareto_front(scores)
        
        # Награда за разнообразие на Парето-фронте
        reward = len(pareto_indices) / n_agents
        
        collective_state = np.mean(
            [agent.get_policy(problem['state']) for agent in agents],
            axis=0
        )
        foam_metrics = foam_calculator.compute_swarm_foam(agents, collective_state)
        
        optimizer.step(reward, foam_metrics['total'])
        
        pareto_front.append(len(pareto_indices))
        foam_history.append(foam_metrics['total'])
    
    return {
        'pareto_front_history': pareto_front,
        'foam_history': foam_history,
        'final_pareto_size': pareto_front[-10:] if pareto_front else []
    }


def find_pareto_front(scores: List[np.ndarray]) -> List[int]:
    """Нахождение Парето-фронта."""
    pareto = []
    
    for i, score_i in enumerate(scores):
        is_dominated = False
        
        for j, score_j in enumerate(scores):
            if i != j:
                # Проверка доминирования
                if np.all(score_j >= score_i) and np.any(score_j > score_i):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto.append(i)
    
    return pareto


def main():
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("GRA-Swarm Experiment: Multi-Objective Optimization")
    print("=" * 60)
    
    results = train_multi_objective_swarm(config)
    
    print(f"\n📊 Final Pareto Front Size: {np.mean(results['final_pareto_size']):.2f}")
    print(f"📉 Final Foam: {results['foam_history'][-1]:.4f}")
    
    # Сохранение результатов
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'pareto_front_mean': float(np.mean(results['final_pareto_size'])),
            'final_foam': float(results['foam_history'][-1])
        }, f, indent=2)
    
    print("\n✅ Experiment completed!")


if __name__ == '__main__':
    main()