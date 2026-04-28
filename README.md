# AI-Driven CPU Process Scheduling for Heterogeneous Multi-Core Processors

> A Deep Reinforcement Learning approach to intelligent task scheduling across heterogeneous (P-core / E-core) processors.

---

## Overview

Modern processors — like Intel's Alder Lake and ARM's big.LITTLE — expose heterogeneous cores with distinct performance and energy profiles. Traditional OS schedulers treat these cores uniformly, leaving significant efficiency gains on the table.

This project trains a **Deep Q-Network (DQN)** agent to learn a core-assignment policy that minimises the **Energy-Delay Product (EDP)** — a joint measure of execution time and energy consumption — across a simulated heterogeneous multi-core environment.

---

## Key Results

| Metric | Value |
|---|---|
| EDP — DQN agent | 1145.86 |
| EDP — Round Robin (baseline) | 1220.23 |
| **EDP reduction vs. Round Robin** | **6.09 %** |
| Execution time reduction vs. Round Robin | 12.81 % |
| Energy overhead vs. static assignment | 15.17 % |
| P-core utilisation (DQN) | 68.9 % |

The DQN scheduler consistently outperforms the Round Robin baseline on execution time while trading a modest increase in energy consumption — resulting in a net EDP improvement of **6.09 %**.

---

## Repository Structure

```
OS Research/
├── collab notebook/
│   └── OS_Research.ipynb   # Main experiment notebook
├── figures/
│   ├── fig1_learning_curve.png   # Reward vs. episode
│   ├── fig2_edp_comparison.png   # EDP: DQN vs. baselines
│   ├── fig3_exec_energy.png      # Execution time & energy breakdown
│   ├── fig4_core_policy.png      # Learned core-assignment policy
│   └── fig5_loss_curve.png       # Training loss curve
├── results/
│   └── results.json        # Numerical evaluation results
└── README.md
```

---

## Methodology

1. **Environment** — A custom OpenAI Gym-style environment simulates a heterogeneous CPU with P-cores (high-performance) and E-cores (high-efficiency). Each step presents an arriving process with a workload profile; the agent assigns it to a core type.

2. **State space** — Current queue lengths, per-core utilisation, and process workload features.

3. **Action space** — Discrete: assign to P-core or E-core.

4. **Reward** — Negative EDP of the completed process, encouraging the agent to jointly minimise latency and energy.

5. **Agent** — DQN with experience replay and a target network. Trained with ε-greedy exploration (ε decays to a final value of 0.05).

6. **Baselines** — Round Robin across all cores; static P-core-only assignment.

---

## Running the Notebook

1. Open `collab notebook/OS_Research.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — the notebook handles environment setup, training, evaluation, and figure generation.
3. Results are written to `results/results.json` and figures saved to `figures/`.

**Dependencies** (install via pip if running locally):
```
numpy  matplotlib  gymnasium  torch
```

---

## Authors

**Manu Vahan** · **Priyanshu Jangra**

*Operating Systems Research Project*
