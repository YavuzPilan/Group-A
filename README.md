# Connect 4 AI Project

## Group A
- **Abdulmuttalip Yusuf Pilan**
- **Attila Kékesi**
- **Mohamed Chedly Abdennebi**
- **Hao Tuan Tran**

---

## Introduction
This project implements an intelligent **Connect 4 AI agent** using **Monte Carlo Tree Search (MCTS) combined with a Neural Network (NN)**. The goal is to enhance decision-making by leveraging both search techniques and deep learning.

Our approach integrates:
- **MCTS**: A search algorithm that selects moves based on simulated games.
- **Neural Networks**: A **Convolutional Neural Network (CNN)** that helps in evaluating board positions and move probabilities.

This hybrid method results in a **stronger, faster, and more reliable** Connect 4 agent compared to traditional MCTS.

---

## Installing dependencies

```bash
pip install -r requirements.txt
```

## How to Play Against the Neural Network
To challenge the AI, simply **run the main script**:
```bash
python main.py
```
This will start a Connect 4 game where you can play against the AI.

---

### Key Components
- **`agents/`**: Different AI strategies including MCTS, Minimax, and random agents.
- **`neural_network/`**: Contains the neural network model and configuration.
- **`Data/`**: Training logs and datasets.
- **`plot/`**: Scripts for visualizing training progress and win rate comparisons.
- **`tests/`**: Unit tests to ensure functionality.
- **`main.py`**: Run this to play against the AI.
- **`train.py`**: Train the neural network using game data.
- **`simulate_games.py`**: Runs multiple AI self-play games for training.
- **`connect4_model.pth`**: Pre-trained model file.

---

## How It Works
### Monte Carlo Tree Search (MCTS)
1. **Selection**: The best move is chosen based on past simulations.
2. **Expansion**: The search tree grows by adding new possible moves.
3. **Simulation**: Random games are played to estimate outcomes.
4. **Backpropagation**: The results are used to update move evaluations.

**Weakness**: Standard MCTS relies on random simulations, which can be slow and lead to suboptimal decisions.

### Neural Network Integration
- **CNN Model**:
  - **3 convolutional layers** extract features from the board.
  - **Policy head** predicts move probabilities.
  - **Value head** estimates the win probability.

- **Training Process**:
  - The model is trained using **game data** (board states, moves, outcomes).
  - It optimizes **policy loss** and **win probability loss**.
  - Performance improves over time and is tracked using plots.

### MCTS + Neural Network
- **Policy head** helps MCTS prioritize strong moves.
- **Win probability estimation** improves simulation accuracy.

### Performance Comparison
| Method | Strength | Speed | Decision Quality |
|--------|---------|------|-----------------|
| MCTS (Random) | Moderate | Slow | Inconsistent |
| MCTS + NN | Strong | Fast | Smart & Reliable |

---

## Future Improvements
- Enhance **neural network architecture** for better performance.
- Use **self-play** to improve decision-making.
- Optimize **MCTS parameters** for more efficient searches.

---

## References
- https://link.springer.com/chapter/10.1007/978-3-540-87608-3_6

---

## Contact
For any questions or contributions, feel free to contact **Group A** members.

---

Thank you for checking out our **Connect 4 AI Project**! 🎮🤖

