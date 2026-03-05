"""
Experiment 1: Math Problem Solving
Эксперимент 1: Решение Математических Задач
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

from gra_swarm.core.agent import GRAAgent, AgentConfig
from gra_swarm.foam.calculator import FoamCalculator
from gra_swarm.optimizer.gra_optimizer import GRAOptimizer


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_math_problem(difficulty: int = 1) -> Dict:
    """Генерация математической задачи."""
    a, b = np.random.randint(1, 20 * difficulty, 2)
    operation = np.random.choice(['+', '-', '*'])
    
    if operation == '+':
        answer = a + b
    elif operation == '-':
        answer = a - b
    else:
        answer = a * b
    
    return {
        'problem': f"{a} {operation} {b} = ?",
        'answer': answer,
        'state': np.array([a / 100.0, b / 100.0, {'+': 0, '-': 1, '*': 2}[operation]])
    }


def train_single_agent(config: dict, n_episodes: int = 100) -> float:
    """Обучение одиночного агента."""
    agent = GRAAgent(AgentConfig(
        agent_id=0,
        personality_seed=42,
        action_space_dim=config['agent']['action_space_dim']
    ))
    
    optimizer = GRAOptimizer([agent], learning_rate=config['agent']['learning_rate'])
    
    scores = []
    for episode in range(n_episodes):
        problem = create_math_problem()
        action = agent.act(problem['state'])
        
        reward = 1.0 if action % 10 == problem['answer'] % 10 else 0.0
        optimizer.step(reward, 0.0)
        scores.append(reward)
    
    return np.mean(scores[-10:])


def train_swarm(config: dict, n_agents: int = 4, n_episodes: int = 100) -> dict:
    """Обучение роя агентов."""
    agents = [
        GRAAgent(AgentConfig(
            agent_id=i,
            personality_seed=42 + i * 17,
            action_space_dim=config['agent']['action_space_dim'],
            level=0
        ))
        for i in range(n_agents)
    ]
    
    foam_calculator = FoamCalculator(
        lambda_div=config['swarm']['lambda_diversity'],
        coordination_weight=config['swarm']['coordination_weight']
    )
    
    optimizer = GRAOptimizer(agents, foam_calculator, config['agent']['learning_rate'])
    
    scores = []
    foam_history = []
    
    for episode in range(n_episodes):
        problem = create_math_problem()
        proposals = [agent.act(problem['state']) for agent in agents]
        
        swarm_answer = np.bincount([p % 10 for p in proposals]).argmax()
        reward = 1.0 if swarm_answer == problem['answer'] % 10 else 0.0
        
        collective_state = np.mean(
            [agent.get_policy(problem['state']) for agent in agents], 
            axis=0
        )
        foam_metrics = foam_calculator.compute_swarm_foam(agents, collective_state)
        
        optimizer.step(reward, foam_metrics['total'])
        scores.append(reward)
        foam_history.append(foam_metrics['total'])
    
    return {
        'score': float(np.mean(scores[-10:])),
        'foam_history': foam_history,
        'final_diversity': foam_metrics['diversity']
    }


def main():
    """Запуск эксперимента."""
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("GRA-Swarm Experiment: Math Problem Solving")
    print("=" * 60)
    
    print("\n📊 Training Single Agent...")
    single_score = train_single_agent(config)
    print(f"Single Agent Score: {single_score:.2%}")
    
    print("\n🐝 Training GRA-Swarm...")
    swarm_results = train_swarm(config)
    print(f"GRA-Swarm Score: {swarm_results['score']:.2%}")
    print(f"Final Diversity I^(l): {swarm_results['final_diversity']:.4f}")
    
    improvement = (swarm_results['score'] - single_score) / max(single_score, 0.01) * 100
    print(f"\n🎯 Improvement: +{improvement:.1f}%")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(swarm_results['foam_history'], label='Φ_swarm', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Foam')
    plt.title('Convergence of Swarm Foam')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'foam_convergence.png', dpi=150)
    
    # Сохранение результатов
    results = {
        'single_agent_score': single_score,
        'swarm_score': swarm_results['score'],
        'improvement_percent': improvement,
        'final_diversity': swarm_results['final_diversity']
    }
    
    with open(results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Experiment completed!")


if __name__ == '__main__':
    main()