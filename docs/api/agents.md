# Agents API / API Агентов

## GRAAgent

```python
from gra_swarm import GRAAgent, AgentConfig

config = AgentConfig(
    agent_id=1,
    personality_seed=42,
    action_space_dim=10
)
agent = GRAAgent(config)