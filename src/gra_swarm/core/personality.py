"""
Personality Generator / Генератор Индивидуальности
Создание уникальных личностей для агентов роя.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class PersonalityType(Enum):
    """Типы личностей агентов / Agent personality types."""
    EXPLORER = "explorer"  # Исследователь
    CRITIC = "critic"      # Критик
    OPTIMIZER = "optimizer"  # Оптимизатор
    CONSERVATIVE = "conservative"  # Консерватор
    CREATIVE = "creative"  # Креатор
    ANALYST = "analyst"    # Аналитик


@dataclass
class PersonalityConfig:
    """Конфигурация личности / Personality configuration."""
    personality_type: PersonalityType
    seed: int
    exploration_rate: float = 0.1
    risk_tolerance: float = 0.5
    learning_speed: float = 1.0
    social_weight: float = 0.5


class PersonalityGenerator:
    """
    Генератор уникальных личностей для агентов.
    Personality generator for unique agent identities.
    
    Создаёт разнообразие через:
    - Разные типы личностей
    - Уникальные зёрна случайности
    - Вариации гиперпараметров
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        self.personality_templates = self._create_templates()
    
    def _create_templates(self) -> Dict[PersonalityType, Dict]:
        """Создание шаблонов личностей."""
        return {
            PersonalityType.EXPLORER: {
                'exploration_rate': 0.3,
                'risk_tolerance': 0.8,
                'learning_speed': 1.2,
                'social_weight': 0.3
            },
            PersonalityType.CRITIC: {
                'exploration_rate': 0.1,
                'risk_tolerance': 0.3,
                'learning_speed': 0.8,
                'social_weight': 0.7
            },
            PersonalityType.OPTIMIZER: {
                'exploration_rate': 0.15,
                'risk_tolerance': 0.5,
                'learning_speed': 1.5,
                'social_weight': 0.5
            },
            PersonalityType.CONSERVATIVE: {
                'exploration_rate': 0.05,
                'risk_tolerance': 0.2,
                'learning_speed': 0.5,
                'social_weight': 0.8
            },
            PersonalityType.CREATIVE: {
                'exploration_rate': 0.4,
                'risk_tolerance': 0.9,
                'learning_speed': 1.0,
                'social_weight': 0.2
            },
            PersonalityType.ANALYST: {
                'exploration_rate': 0.1,
                'risk_tolerance': 0.4,
                'learning_speed': 0.9,
                'social_weight': 0.6
            }
        }
    
    def generate(
        self, 
        agent_id: int, 
        personality_type: Optional[PersonalityType] = None
    ) -> PersonalityConfig:
        """
        Генерация конфигурации личности для агента.
        
        Args:
            agent_id: ID агента
            personality_type: Тип личности (или случайный)
            
        Returns:
            PersonalityConfig для агента
        """
        if personality_type is None:
            # Случайный выбор типа для разнообразия
            personality_type = np.random.choice(list(PersonalityType))
        
        template = self.personality_templates[personality_type]
        
        # Добавляем вариации на основе ID агента
        np.random.seed(self.seed + agent_id * 17)
        variation = np.random.uniform(-0.1, 0.1, 4)
        
        return PersonalityConfig(
            personality_type=personality_type,
            seed=self.seed + agent_id * 17,
            exploration_rate=max(0.01, template['exploration_rate'] + variation[0]),
            risk_tolerance=max(0.01, min(0.99, template['risk_tolerance'] + variation[1])),
            learning_speed=max(0.1, template['learning_speed'] + variation[2]),
            social_weight=max(0.01, min(0.99, template['social_weight'] + variation[3]))
        )
    
    def generate_diverse_swarm(
        self, 
        n_agents: int
    ) -> List[PersonalityConfig]:
        """
        Генерация разнообразного роя агентов.
        
        Args:
            n_agents: Количество агентов
            
        Returns:
            Список конфигураций личностей
        """
        configs = []
        personality_types = list(PersonalityType)
        
        for i in range(n_agents):
            # Циклическое распределение типов для баланса
            p_type = personality_types[i % len(personality_types)]
            config = self.generate(i, p_type)
            configs.append(config)
        
        return configs
    
    def compute_personality_diversity(
        self, 
        configs: List[PersonalityConfig]
    ) -> float:
        """
        Вычисление разнообразия личностей в рое.
        
        Returns:
            Мера разнообразия (0-1)
        """
        if len(configs) < 2:
            return 0.0
        
        # Разнообразие типов
        types = [c.personality_type for c in configs]
        unique_types = len(set(types))
        type_diversity = unique_types / len(PersonalityType)
        
        # Разнообразие параметров
        params = np.array([
            [c.exploration_rate, c.risk_tolerance, 
             c.learning_speed, c.social_weight]
            for c in configs
        ])
        param_variance = np.mean(np.var(params, axis=0))
        
        return 0.5 * type_diversity + 0.5 * param_variance