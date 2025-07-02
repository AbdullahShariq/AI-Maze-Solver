# AI Maze Solver Game

## Overview

The **AI Maze Solver Game** is an interactive Python application built with `tkinter` that challenges players to navigate a maze while collecting keys and power-ups, avoiding obstacles, and reaching a goal. The game features dynamic elements like moving obstacles and offers lifelines such as wall-breaking, obstacle-freezing, and path-revealing capabilities. Additionally, it includes an AI system that can solve the maze using various pathfinding algorithms (BFS, DFS, A*, Greedy Best-First Search) and provides real-time comparison of their performance.

The project is designed for both casual gameplay and educational exploration of pathfinding algorithms, offering a visually engaging interface with a dark theme and detailed statistics tracking.

---

## Features

- **Interactive Maze Gameplay**: Navigate a randomly generated maze using arrow keys, collecting keys to unlock the goal while avoiding moving obstacles.
- **Dynamic Maze Elements**:
  - **Keys**: Collect a required number of keys to unlock the goal.
  - **Power-ups**: Gain bonuses like extra wall breaks, obstacle freezes, or path reveals.
  - **Moving Obstacles**: Avoid dynamic obstacles that move periodically, with collision penalties.
- **Lifelines**:
  - **Wall Break**: Break through walls (limited uses).
  - **Freeze Obstacles**: Temporarily halt obstacle movement (10 seconds).
  - **Reveal Path**: Show a portion of the optimal path using a selected algorithm.
- **AI Pathfinding**:
  - Supports BFS, DFS, A*, and Greedy Best-First Search algorithms.
  - Run AI to automatically solve the maze from the current position.
  - Pause/resume AI execution.
  - Find the optimal algorithm for the current maze state.
- **Algorithm Comparison**: Real-time table comparing algorithm performance (moves, score, power-ups collected, computation time).
- **Replay and Reset**: Replay the same maze or generate a new one.
- **Scoring System**: Earn points for collecting keys and power-ups, with bonuses for quick completion and fewer moves.
- **Dark Mode UI**: A sleek, modern interface with a dark theme for better visibility.

---

## Algorithm Details

- **BFS (Breadth-First Search)**: Finds the shortest path to keys and the goal, exploring level by level.
- **DFS (Depth-First Search)**: Explores deeply along each branch, potentially finding longer paths.
- **A***: Uses Manhattan distance heuristic for efficient shortest-path finding.
- **Greedy Best-First Search**: Prioritizes moves closer to the target based on heuristic, may not guarantee shortest paths.

---

## Installation

### Prerequisites

- Python 3.6 or higher
- Required libraries:
  - `tkinter` (usually included with Python)
  - `random`, `heapq`, `collections`, `time`, `copy` (standard Python libraries)

### Setup

1. **Clone or Download the Repository**:
   ```bash
   git clone https://github.com/AbdullahShariq/AI-Maze-Solver.git
   cd AI-Maze-Solver
   ```

2. **Ensure Dependencies**:
   Verify that Python is installed and `tkinter` is available. You can test this by running:
   ```bash
   python -c "import tkinter"
   ```
   
3. **Run the Application**:
   ```bash
   python AI_Maze_Solver.py
   ```

---

## Usage

1. **Launch the Game**:
   Run the script to open the game window. A maze is generated automatically.

2. **Controls**:
   - **Arrow Keys**: Move the agent (blue square) up, down, left, or right.
   - **Lifeline Buttons**:
     - **Wall Break**: Click to enter wall-break mode, then click a wall to destroy it.
     - **Freeze**: Temporarily stop obstacle movement.
     - **Reveal Path**: Show up to 10 steps of the optimal path using the selected algorithm.
   - **AI Controls**:
     - Select an algorithm (BFS, DFS, A*, Greedy) from the dropdown.
     - Click "Run AI" to let the AI navigate the maze.
     - Click "Pause AI" to pause/resume AI movement.
     - Click "Optimal Solver" to find and run the best algorithm for the current state.
   - **Game Controls**:
     - **Replay Maze**: Restart the current maze with the same layout.
     - **New Maze**: Generate a new random maze.

3. **Game Objective**:
   - Collect the required number of keys (yellow squares) to unlock the goal (green square).
   - Avoid obstacles (red squares) that move every 2 seconds and deduct 100 points on collision.
   - Reach the goal to win, earning a score based on keys collected, power-ups, moves taken, and time.

4. **Statistics and Comparison**:
   - The right panel displays real-time stats: score, moves, keys collected, power-ups, and elapsed time.
   - The "Live Algorithm Comparison" table shows simulated performance of all algorithms from the current state.

## Code Structure

- **MazeEnvironment Class**:
  - Manages the maze grid, agent, goal, keys, power-ups, and obstacles.
  - Implements maze generation with multiple paths using DFS and additional path creation.
  - Handles game mechanics: movement, lifelines, scoring, and obstacle dynamics.
  - Provides pathfinding algorithms (BFS, DFS, A*, Greedy) and simulation for comparison.

- **MazeApp Class**:
  - Creates the `tkinter` GUI with a canvas for the maze and panels for controls and stats.
  - Handles user input (keyboard, mouse clicks), AI animation, and periodic updates (obstacle movement, timers).
  - Manages UI elements like buttons, labels, and the algorithm comparison table.

- **Key Files**:
  - `AI_Maze_Solver.py`: The main script containing all game logic and UI code.

The AI simulates paths from the current state, collecting keys and reaching the goal, while the comparison table evaluates performance based on moves, score, power-ups collected, and computation time.
