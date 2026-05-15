# Traffic Route Optimization using Q-Learning Maze Solver

## Project Overview

This project uses Reinforcement Learning and Q-Learning to optimize traffic route navigation in a maze-based environment.

The maze represents a traffic road network where:
- Open cells represent available roads
- Blocked cells represent traffic congestion or closed roads
- The RL agent learns the optimal path from source to destination

---

## Objective

The objective of this project is to train an intelligent agent to find the shortest and most efficient traffic route using Q-Learning.

---

## SDG Mapping

- SDG 11: Sustainable Cities and Communities
- SDG 9: Industry, Innovation and Infrastructure

---

## Technologies Used

- Python
- NumPy
- Reinforcement Learning
- Q-Learning
- GitHub
- MLOps Workflow

---

## Project Structure

```text
Traffic-Route-Optimization-RL/

├── sim/
│   ├── maze_env.py
│   ├── train.py
│   ├── evaluate.py
│
├── configs/
│   ├── qlearning_v1.yaml
│   ├── qlearning_v2.yaml
│
├── experiments/
│
├── policies/
│   ├── policy_v1.pkl
│
├── plots/
│
├── logs/
│   ├── log.json
│
├── README.md
├── requirements.txt
└── .gitignore