"""
Experiment 3: LLM Debate Simulation
Эксперимент 3: Симуляция LLM-Дебатов
"""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import List, Dict
from enum import IntEnum

from gra_swarm.core.agent import GRAAgent, AgentConfig
from gra_swarm.foam.calculator import FoamCalculator
from gra_swarm.optimizer.gra_optimizer import GRAOptimizer


class DebateAction(IntEnum):
    """Действия в дебатах."""
    AGREE = 0
    DISAGREE = 1
    NEUTRAL = 2
    QUESTION = 3
    ELABORATE = 4


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def simulate_debate_round(
    agents: List[GRAAgent],
    topic_state: np.ndarray,
    previous_statements: List[int]
) -> List[int]:
    """Симуляция раунда дебатов."""
    statements = []
    
    for agent in agents:
        # Состояние включает тему и предыдущие высказывания
        state = np.concatenate([
            topic_state,
            np.bincount(previous_statements, minlength=5).astype(float) / len(agents)
        ])
        
        action = agent.act(state)
        statements.append(action)
    
    return statements


def compute_consensus(statements: List[int]) -> float:
    """Вычисление консенсуса."""
    if len(statements) < 2:
        return 0.0
    
    counts = np.bincount(statements, minlength=5)
    max_count = counts.max()
    
    return max_count / len(statements)


def train_debate_swarm(config: dict) -> dict:
    """Обучение роя для дебатов."""
    n_agents = config['agent']['n_agents']
    n_rounds = config['debate']['n_rounds']
    
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
    
    topic_state = np.random.randn(10)
    consensus_history = []
    foam_history = []
    
    for round_num in range(n_rounds):
        previous = consensus_history[-1] if consensus_history else []
        statements = simulate_debate_round(agents, topic_state, previous)
        
        consensus = compute_consensus(statements)
        consensus_history.append(consensus)
        
        # Награда за достижение консенсуса с сохранением разнообразия
        reward = consensus * 0.5 + (1 - consensus) * 0.5
        
        collective_state = np.mean(
            [agent.get_policy(topic_state) for agent in agents],
            axis=0
        )
        foam_metrics = foam_calculator.compute_swarm_foam(agents, collective_state)
        
        foam_history.append(foam_metrics['total'])
    
    return {
        'consensus_history': consensus_history,
        'foam_history': foam_history,
        'final_consensus': consensus_history[-1] if consensus_history else 0
    }


def main():
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(str(config_path))
    
    print("=" * 60)
    print("GRA-Swarm Experiment: LLM Debate Simulation")
    print("=" * 60)
    
    results = train_debate_swarm(config)
    
    print(f"\n📊 Final Consensus: {results['final_consensus']:.2%}")
    print(f"📉 Final Foam: {results['foam_history'][-1]:.4f}")
    
    # Сохранение результатов
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'final_consensus': float(results['final_consensus']),
            'final_foam': float(results['foam_history'][-1])
        }, f, indent=2)
    
    print("\n✅ Experiment completed!")


if __name__ == '__main__':
    main()