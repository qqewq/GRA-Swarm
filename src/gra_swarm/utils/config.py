"""
Configuration Loader / Загрузчик Конфигурации
Управление конфигурацией экспериментов.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента."""
    name: str = "gra_swarm_experiment"
    version: str = "1.0"
    seed: int = 42
    
    # Агенты
    n_agents: int = 4
    action_space_dim: int = 100
    learning_rate: float = 1e-3
    
    # Рой
    lambda_diversity: float = 0.1
    coordination_weight: float = 1.0
    min_individuality: float = 0.01
    
    # Обучение
    n_episodes: int = 100
    foam_threshold: float = 0.001
    
    # Иерархия
    n_levels: int = 2
    alpha: float = 0.7
    
    # Вывод
    save_results: bool = True
    visualization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, filepath: str):
        """Сохранить конфигурацию."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Загрузить конфигурацию."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ConfigLoader:
    """
    Загрузчик конфигураций из файлов.
    Configuration file loader.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def load_yaml(self, filepath: str) -> Dict:
        """Загрузка YAML конфигурации."""
        path = self.base_path / filepath
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_json(self, filepath: str) -> Dict:
        """Загрузка JSON конфигурации."""
        path = self.base_path / filepath
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_yaml(self, data: Dict, filepath: str):
        """Сохранение YAML конфигурации."""
        path = self.base_path / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    
    def merge_configs(
        self, 
        base_config: Dict, 
        override_config: Dict
    ) -> Dict:
        """Слияние конфигураций."""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result