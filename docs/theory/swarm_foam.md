# Swarm Foam / Пена Роя

## Definition / Определение

Swarm foam measures the inconsistency between agent policies and collective goals.

Пена роя измеряет несогласованность между политиками агентов и коллективными целями.

## Components / Компоненты

$$\Phi_{swarm} = \frac{1}{N}\sum_i \Phi_{agent}^{(i)} + d(\Xi, P_G(\Xi)) - \lambda \cdot \mathcal{I}$$

1. Individual foam / Индивидуальная пена
2. Coordination loss / Коллективная несогласованность
3. Diversity bonus / Бонус за разнообразие

## Optimization / Оптимизация

Minimize $J_{swarm}$ through gradient descent on agent policies.

Минимизация $J_{swarm}$ через градиентный спуск на политиках агентов.