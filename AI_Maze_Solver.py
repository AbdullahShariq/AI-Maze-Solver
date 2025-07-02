import tkinter as tk
from tkinter import messagebox, ttk
import random
import heapq
from collections import deque
import time
import copy #replay keliye

# Constants
GRID_WIDTH = 30
GRID_HEIGHT = 20
CELL_SIZE = 30
WALL = '#'
PATH = ' '
AGENT = 'A'
GOAL = 'G'
KEY = 'K'
POWERUP = 'P'
VISITED_PATH = '·' #AI path visualization
REVEALED_STEP = '*' # Revealed steps
OBSTACLE = 'O'

COLOR_MAP = {
    WALL: "#1C1C1C",           
    PATH: "#F0F0F0",            
    AGENT: "#007ACC",           
    GOAL: "#2ECC71",           
    KEY: "#FFD700",             
    POWERUP: "#FF69B4",        
    VISITED_PATH: "#ADD8E6",   
    REVEALED_STEP: "#FFA500",   
    OBSTACLE: "#B22222", 
}


class MazeEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.initial_grid_config = None #storing maze for replay
        self.initial_agent_pos = (1,1) #agent at top left
        self.initial_goal_pos = (width-2, height-2) #goal at bottom right
        self.initial_keys = []
        self.initial_powerups = []
        self.initial_obstacles = []
        self.initial_total_keys_required = 0
        
        self.grid = []
        self.agent_pos = self.initial_agent_pos
        self.goal_pos = self.initial_goal_pos
        self.keys = []
        self.powerups = []
        self.obstacles = []
        self.obstacle_direction = {}
        
        self.revealed_path_segments = []
        
        self.algorithm_paths = {} # storing path used by different algos
        self.algorithm_stats = {} # storing states for each algo
        
        # Game state
        self.wall_breaks = 0
        self.freeze_available = 0
        self.reveal_path_available = 0
        self.frozen_obstacles = False
        self.frozen_time = 0
        self.collected_keys = 0
        self.collected_powerups = 0
        self.moves_taken = 0
        self.start_time = None
        self.score = 0
        self.game_finished = False
        self.total_keys_required = 0
        
        self.generate_maze()

    def _initialize_game_state(self):
        self.wall_breaks = 2
        self.freeze_available = 2
        self.reveal_path_available = 2
        self.frozen_obstacles = False
        self.frozen_time = 0
        self.collected_keys = 0
        self.collected_powerups = 0
        self.moves_taken = 0
        self.score = 0
        self.start_time = None
        self.revealed_path_segments = []
        self.game_finished = False
        #initial values after resetting/replaying the game
        self.agent_pos = self.initial_agent_pos 
        self.goal_pos = self.initial_goal_pos
        self.keys = copy.deepcopy(self.initial_keys)
        self.powerups = copy.deepcopy(self.initial_powerups)
        self.obstacles = copy.deepcopy(self.initial_obstacles)
        self.obstacle_direction = {} 
        for pos in self.obstacles:
            self.obstacle_direction[pos] = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)]) #random movement of obstacles
        self.total_keys_required = self.initial_total_keys_required
        self.grid = copy.deepcopy(self.initial_grid_config) if self.initial_grid_config else []

    #Generation of maze
    def generate_maze(self):
        
        #generating multiple paths so user/AI can pursue optimal one
        self._generate_maze_with_multiple_paths()
        
        #blocking all side of maze
        for x in range(self.width):
            self.grid[0][x] = 1
            self.grid[self.height - 1][x] = 1
        for y in range(self.height):
            self.grid[y][0] = 1
            self.grid[y][self.width - 1] = 1
        
        #storing initial config for replay
        self.initial_grid_config = copy.deepcopy(self.grid)

        #setting initial position for agent and goal
        self.initial_agent_pos = (1,1) 
        self.initial_goal_pos = (self.width - 2, self.height - 2)

        #placing keys
        self.initial_keys = []
        key_count = random.randint(2, 3)
        self.initial_total_keys_required = key_count
        
        # avoiding (powerups,keys,obstacles) to be placed on similar position of agent and goal
        temp_exclude = {self.initial_agent_pos, self.initial_goal_pos}

        for _ in range(key_count):
            pos = self.get_random_empty_cell(exclude=temp_exclude)
            if pos:
                self.initial_keys.append(pos)
                temp_exclude.add(pos)
        
        # placing powerups
        self.initial_powerups = []
        for _ in range(random.randint(4, 5)):
            pos = self.get_random_empty_cell(exclude=temp_exclude)
            if pos:
                self.initial_powerups.append(pos)
                temp_exclude.add(pos)
        
        # placing obstacles
        self.initial_obstacles = []
        # obstacle_direction is part of dynamic state, initialized in _initialize_game_state
        for _ in range(random.randint(5, 6)):
            pos = self.get_random_empty_cell(exclude=temp_exclude)
            if pos:
                self.initial_obstacles.append(pos)
        
        # Now, fully initialize the game state based on these newly generated initial settings
        self._initialize_game_state()
        
        # Ensure the maze is solvable (at least to the first key or goal if no keys)
        initial_target = self.initial_goal_pos # Use initial_goal_pos
        if self.initial_keys:
            initial_target = self.initial_keys[0]

        if not self._has_path(self.initial_agent_pos, initial_target): # Use initial_agent_pos
            print("Warning: Initial path check failed. Regenerating maze.")
            self.generate_maze() # Recursive call, be cautious of potential infinite loop if generation is flawed


    def reset_state_for_replay(self):
        if self.initial_grid_config is None:
            print("Error: No initial configuration to replay. Generating new maze.")
            self.generate_maze() # This will set up initial_grid_config and then call _initialize_game_state
            return

        # Reset dynamic game state variables to their initial values or defaults
        self._initialize_game_state()


    def _generate_maze_with_multiple_paths(self, max_attempts=20):
        for _ in range(max_attempts):
            self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
            self._create_path_with_dfs((1,1)) # Start DFS from a fixed point like agent start
            self._create_additional_path()
            # Ensure agent and goal positions are clear
            self.grid[1][1] = 0 
            self.grid[self.height - 2][self.width - 2] = 0
            
            if self._count_paths((1,1), (self.width - 2, self.height - 2)) >= 2:
                return
        # Fallback if multiple paths not easily achieved
        print("Warning: Could not guarantee multiple paths within attempts.")


    def _create_path_with_dfs(self, start_pos):
        stack = [start_pos]
        visited = {start_pos}
        self.grid[start_pos[1]][start_pos[0]] = 0
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]: # Jump 2 cells
                nx, ny = x + dx, y + dy
                # Check bounds (nx,ny) and (x+dx/2, y+dy/2)
                if (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Carve path by clearing the wall between cells
                self.grid[y + (ny - y) // 2][x + (nx - x) // 2] = 0
                self.grid[ny][nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Ensure path to goal, if it's not already connected
        gx, gy = self.width - 2, self.height - 2
        if self.grid[gy][gx] == 1: # If goal is walled off
            self.grid[gy][gx] = 0
            # Try to connect goal to an existing path
            for dx_goal, dy_goal in [(0,1), (1,0), (0,-1), (-1,0)]:
                nnx, nny = gx + dx_goal, gy + dy_goal
                if 0 <= nnx < self.width and 0 <= nny < self.height and self.grid[nny][nnx] == 0:
                    break 
            else: 
                curr_x, curr_y = gx, gy
                while curr_x > 1 or curr_y > 1: 
                    if curr_x > 1 and (curr_x % 2 == 0 or self.grid[curr_y][curr_x-1] == 1):
                        self.grid[curr_y][curr_x-1] = 0
                        curr_x -=1
                    elif curr_y > 1 and (curr_y % 2 == 0 or self.grid[curr_y-1][curr_x] == 1):
                         self.grid[curr_y-1][curr_x] = 0
                         curr_y -=1
                    else: 
                        if random.choice([True, False]) and curr_x > 1: curr_x -=1
                        elif curr_y > 1: curr_y -=1
                        else: break
                        if 0 <= curr_y < self.height and 0 <= curr_x < self.width:
                            self.grid[curr_y][curr_x] = 0
                        else: break # out of bounds safety


    def _create_additional_path(self):
        num_walls_to_remove = int((self.width * self.height) * 0.05) # Remove 5% of walls
        
        for _ in range(num_walls_to_remove):
            wall_candidates = []
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.grid[y][x] == 1: 
                        if x > 0 and x < self.width -1 and self.grid[y][x-1] == 0 and self.grid[y][x+1] == 0:
                             wall_candidates.append((x,y))
                             continue
                        if y > 0 and y < self.height -1 and self.grid[y-1][x] == 0 and self.grid[y+1][x] == 0:
                             wall_candidates.append((x,y))
                             continue
            
            if wall_candidates:
                wx, wy = random.choice(wall_candidates)
                self.grid[wy][wx] = 0

    def _count_paths(self, start, goal, max_paths_to_find=2):
        path1 = self.bfs_to_target(start, goal, ignore_keys=True)
        if not path1:
            return 0
        if len(path1) <= 3: 
            return 1

        original_grid_val = self.grid[path1[len(path1)//2][1]][path1[len(path1)//2][0]]
        self.grid[path1[len(path1)//2][1]][path1[len(path1)//2][0]] = 1 
        
        path2 = self.bfs_to_target(start, goal, ignore_keys=True)
        
        self.grid[path1[len(path1)//2][1]][path1[len(path1)//2][0]] = original_grid_val
        
        return 2 if path2 else 1

    def _has_path(self, start, goal):
        return bool(self.bfs_to_target(start, goal, ignore_keys=True))
        
    def get_random_empty_cell(self, exclude=None):
        if exclude is None: exclude = set()
        empty_cells = []
        for y in range(1, self.height - 1): 
            for x in range(1, self.width - 1):
                # Check self.grid exists and is populated
                if self.grid and y < len(self.grid) and x < len(self.grid[y]) and \
                   self.grid[y][x] == 0 and (x, y) not in exclude:
                    empty_cells.append((x, y))
        return random.choice(empty_cells) if empty_cells else None
        
    def all_special_cells(self):
        special_cells = {self.agent_pos, self.goal_pos}
        special_cells.update(self.keys)
        special_cells.update(self.powerups)
        # Obstacles are not "special cells" in the sense that other items can't be placed there initially
        # but they are positions to avoid for item placement if they are already present.
        return special_cells

    def break_wall(self, wall_pos):
        x, y = wall_pos
        if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
            return False, "Cannot break boundary walls."

        if 0 < x < self.width-1 and 0 < y < self.height-1 and self.grid[y][x] == 1:
            if self.wall_breaks > 0:
                self.grid[y][x] = 0
                self.wall_breaks -= 1
                return True, "Wall broken!"
            return False, "No wall breaks left."
        return False, "Not a valid wall to break."

    def get_valid_neighbors(self, pos, for_obstacle=False):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                is_obstacle_collision = (nx,ny) in self.obstacles and not self.frozen_obstacles and not for_obstacle
                if self.grid[ny][nx] == 0 and not is_obstacle_collision:
                    neighbors.append((nx, ny))
        return neighbors

    def is_valid_move(self, pos): # For agent
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height): return False
        if self.grid[y][x] == 1: return False 
        if (x,y) in self.obstacles and not self.frozen_obstacles: return False 
        return True

    def use_freeze_obstacles(self):
        if self.freeze_available > 0 and not self.frozen_obstacles:
            self.frozen_obstacles = True
            self.freeze_available -= 1
            self.frozen_time = time.time()
            return True, "Obstacles frozen for 10 seconds!"
        elif self.frozen_obstacles:
            return False, "Obstacles are already frozen."
        return False, "No freeze power-ups left."
        
    def use_reveal_path(self, algorithm_name="a_star"):
        if self.reveal_path_available <= 0:
            return False, "No reveal path lifelines left."
            
        path = []
        # Pass current agent state for pathfinding
        # self.keys here refers to the *remaining* keys on the map
        if algorithm_name == "bfs": path = self._bfs_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
        elif algorithm_name == "dfs": path = self._dfs_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
        elif algorithm_name == "greedy": path = self._greedy_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
        else: path = self._a_star_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
            
        if path:
            self.revealed_path_segments = path[:min(10, len(path))] 
            self.reveal_path_available -= 1
            return True, f"Path revealed using {algorithm_name}!"
        return False, "Could not find a path to reveal."
        
    def move_agent(self, new_pos):
        if self.game_finished: return None

        if not self.start_time:
            self.start_time = time.time()
            
        pickup_message = None
        if new_pos in self.keys:
            self.keys.remove(new_pos)
            self.collected_keys += 1
            self.score += 50
            pickup_message = f"Key collected! ({self.collected_keys}/{self.total_keys_required})"
            
        if new_pos in self.powerups:
            self.powerups.remove(new_pos)
            self.collected_powerups += 1
            self.score += 30
            rand_effect = random.randint(0, 2)
            if rand_effect == 0 and self.wall_breaks < 3:
                self.wall_breaks += 1
                pickup_message = "Gained a Wall Break!"
            elif rand_effect == 1 and self.freeze_available < 3: # Max 3 freeze
                self.freeze_available += 1
                pickup_message = "Gained a Freeze Obstacles!"
            elif self.reveal_path_available < 3: # Max 3 reveal
                self.reveal_path_available += 1
                pickup_message = "Gained a Reveal Path!"
            else: 
                self.score += 20 
                pickup_message = "Bonus points!"
            
        self.agent_pos = new_pos
        self.moves_taken += 1
        self.revealed_path_segments = [] 
        
        if new_pos == self.goal_pos:
            if self.collected_keys >= self.total_keys_required:
                self.finish_game()
                return "Goal Reached! You Win!"
            else:
                return f"Reached goal, but need {self.total_keys_required - self.collected_keys} more key(s)!"
            
        return pickup_message 
        
    def finish_game(self, simulated_time_override=None):
        if not self.game_finished:
            self.game_finished = True
            
            if simulated_time_override is not None:
                completion_time = simulated_time_override
            elif self.start_time:
                completion_time = time.time() - self.start_time
            else:
                completion_time = 0 # Should not happen if game started properly

            time_bonus = max(0, 1000 - int(completion_time * 5)) 
            move_bonus = max(0, 500 - self.moves_taken * 5)
            self.score += 500 + time_bonus + move_bonus 
            
    def update_lifeline_timers(self):
        if self.frozen_obstacles and (time.time() - self.frozen_time > 10):
            self.frozen_obstacles = False
            return "Obstacles unfrozen."
        return None

    def move_obstacles(self):
        if self.frozen_obstacles or self.game_finished:
            return False # No collision if frozen or game over

        new_obstacles = []
        new_directions = {}
        collision_occurred = False

        for pos in self.obstacles:
            x, y = pos
            dx, dy = self.obstacle_direction.get(pos, random.choice([(0,1),(1,0),(0,-1),(-1,0)]))
            
            moved_this_turn = False
            for attempt in range(4):  # Try current direction, then others if blocked
                nx, ny = x + dx, y + dy
                
                # Check if valid move for obstacle (not wall, not goal, not key, not another new obstacle pos)
                if (0 < nx < self.width - 1 and 0 < ny < self.height - 1 and 
                    self.grid[ny][nx] == 0 and 
                    (nx, ny) != self.goal_pos and 
                    (nx, ny) not in self.keys and 
                    (nx, ny) not in self.powerups and # Obstacles shouldn't destroy powerups
                    (nx, ny) not in new_obstacles): # Avoid collision with other moving obstacles in same step
                    
                    new_obstacles.append((nx, ny))
                    new_directions[(nx, ny)] = (dx, dy)
                    moved_this_turn = True
                    
                    if (nx,ny) == self.agent_pos: # Check collision after this obstacle moved
                        collision_occurred = True
                    break 
                else:  # Try turning (e.g., right turn)
                    dx, dy = -dy, dx 
                    if attempt == 3: # Last attempt, stay put if still blocked
                        new_obstacles.append(pos)
                        new_directions[pos] = self.obstacle_direction.get(pos, (dx,dy)) # Keep original or last attempted dir
            
            if not moved_this_turn and pos not in new_obstacles : # Ensure it's added if it couldn't move
                 new_obstacles.append(pos)
                 new_directions[pos] = self.obstacle_direction.get(pos, (dx,dy))


        self.obstacles = new_obstacles
        self.obstacle_direction = new_directions
        
        if collision_occurred:
            self.score = max(0, self.score - 100) # Score doesn't go below 0 from this
            # Game doesn't necessarily end on collision, just a penalty
            return True  # Collision occurred
        return False


    # --- Pathfinding Algorithms ---
    def bfs_to_target(self, start, target, ignore_keys=False): 
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path + [current]
                
            for neighbor in self.get_valid_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [current]))
        return []

    def _bfs_solve_path(self, start_pos, current_keys_collected, keys_list_remaining):
        path_to_keys = []
        temp_agent_pos = start_pos
        temp_keys_collected_count = current_keys_collected
        temp_keys_map_list = copy.deepcopy(keys_list_remaining)

        current_path_segment_starts_with_start_pos = True

        while temp_keys_collected_count < self.total_keys_required:
            if not temp_keys_map_list: break 
            
            shortest_path_to_key_segment = []
            best_key_pos = None

            for key_pos in temp_keys_map_list:
                path_segment = self.bfs_to_target(temp_agent_pos, key_pos)
                if path_segment and (not shortest_path_to_key_segment or len(path_segment) < len(shortest_path_to_key_segment)):
                    shortest_path_to_key_segment = path_segment
                    best_key_pos = key_pos
            
            if shortest_path_to_key_segment:
                # If it's not the very first segment, skip the first node (current pos)
                to_extend = shortest_path_to_key_segment if current_path_segment_starts_with_start_pos else shortest_path_to_key_segment[1:]
                path_to_keys.extend(to_extend)
                current_path_segment_starts_with_start_pos = False 
                
                temp_agent_pos = best_key_pos
                temp_keys_map_list.remove(best_key_pos)
                temp_keys_collected_count +=1 
            else:
                return [] 

        path_to_goal_segment = self.bfs_to_target(temp_agent_pos, self.goal_pos)
        if path_to_goal_segment:
            to_extend = path_to_goal_segment if current_path_segment_starts_with_start_pos else path_to_goal_segment[1:]
            full_path = path_to_keys + to_extend
            
            # Ensure path always starts with the original start_pos if it's not already there due to logic
            if not full_path or full_path[0] != start_pos:
                if start_pos == self.goal_pos and temp_keys_collected_count >= self.total_keys_required: return [start_pos]
                # This case should ideally be handled by ensuring the first segment includes start_pos
                # For now, if full_path is S-K-G, and S != start_pos (which implies S was first key)
                # we might need to prepend. But the logic with current_path_segment_starts_with_start_pos should manage this.
            
            return full_path if full_path else ([start_pos] if start_pos == self.goal_pos and temp_keys_collected_count >= self.total_keys_required else [])

        # If no keys were needed or all collected, and agent is already at goal
        elif not path_to_keys and temp_keys_collected_count >= self.total_keys_required and start_pos == self.goal_pos:
             return [start_pos]
        return []


    def dfs_to_target(self, start, target):
        stack = [(start, [])]
        visited = set()
        while stack:
            current, path = stack.pop()
            if current == target: return path + [current]
            if current in visited: continue
            visited.add(current)
            # Randomize neighbor order for DFS to explore different paths if called multiple times
            neighbors = self.get_valid_neighbors(current)
            random.shuffle(neighbors)
            for neighbor in neighbors: 
                if neighbor not in visited:
                    stack.append((neighbor, path + [current]))
        return []

    def _dfs_solve_path(self, start_pos, current_keys_collected_count_player, keys_list_player_needs):
        # This DFS attempts to find *A* path, not necessarily optimal.
        # State for memoization: (current_position, frozenset_of_remaining_initial_keys_to_find)
        memo = {}
        
        # We need to operate on the full list of initial_keys for the mask logic
        # `player_has_already_collected_mask` identifies which of the `initial_keys` are already collected by player
        player_has_already_collected_mask = 0
        for i, ik_pos in enumerate(self.initial_keys):
            if ik_pos not in keys_list_player_needs: # if key is NOT in list player still needs, player has it
                player_has_already_collected_mask |= (1 << i)

        def solve_recursive_dfs(current_pos_sim, collected_mask_sim):
            state = (current_pos_sim, collected_mask_sim)
            if state in memo: return memo[state]

            # Check if all required keys (by total_keys_required) are now in the mask
            num_keys_in_mask = bin(collected_mask_sim).count('1')
            if num_keys_in_mask >= self.total_keys_required:
                path_to_goal_segment = self.dfs_to_target(current_pos_sim, self.goal_pos)
                memo[state] = path_to_goal_segment
                return path_to_goal_segment

            # Try to find paths to remaining (unmasked) initial keys
            # This part can be very slow if many keys and complex maze
            # For DFS, usually we'd pick one, recurse, then backtrack.
            # To make it somewhat directed, could try "closest" uncollected key first (heuristic for DFS)
            
            potential_next_keys_to_target = []
            for i, key_pos_initial in enumerate(self.initial_keys):
                if not (collected_mask_sim & (1 << i)): # If key 'i' is not in mask yet
                    potential_next_keys_to_target.append((key_pos_initial, i)) # (pos, index)
            
            # Optional: sort by distance to make DFS a bit more like Greedy in choice
            # random.shuffle(potential_next_keys_to_target) # Pure DFS exploration

            for key_pos_target, key_idx_target in potential_next_keys_to_target:
                path_to_this_key_segment = self.dfs_to_target(current_pos_sim, key_pos_target)
                if path_to_this_key_segment:
                    # Recursive call from this key_pos_target, with updated mask
                    remaining_path_segment = solve_recursive_dfs(key_pos_target, collected_mask_sim | (1 << key_idx_target))
                    if remaining_path_segment:
                        # Combine: path to this key (excluding its end) + path from this key onwards
                        full_segment_path = path_to_this_key_segment[:-1] + remaining_path_segment
                        memo[state] = full_segment_path # Cache and return first found solution
                        return full_segment_path
            
            memo[state] = [] # No path found from this state
            return []

        # Start the recursive DFS from the player's current state
        full_simulated_path = solve_recursive_dfs(start_pos, player_has_already_collected_mask)
        
        # solve_recursive_dfs returns path segments. If start_pos itself is first element of path, fine.
        # If full_simulated_path = [P1, P2, G] and start_pos = S, and S->P1 was first step
        # we need to ensure S is included.
        # dfs_to_target returns [S, P1], so path_to_this_key_segment[:-1] is [S]
        # remaining_path_segment from solve_recursive_dfs(P1, new_mask) might be [P1, P2, G]
        # So [S] + [P1, P2, G] -> [S, P1, P2, G]. This seems correct.

        if not full_simulated_path:
             # If already at goal and keys met
            if start_pos == self.goal_pos and bin(player_has_already_collected_mask).count('1') >= self.total_keys_required:
                return [start_pos]
            return []
        
        # Ensure start_pos is the first element if path is non-empty
        if full_simulated_path[0] != start_pos :
            # This can happen if the recursive solution starts from a key that was the start_pos.
            # However, dfs_to_target(start, key) should yield [start, ..., key]
            # For now, assume solve_recursive_dfs produces a path starting from its current_pos_sim argument.
             pass # Should be implicitly handled by how segments are combined

        return full_simulated_path


    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star_to_target(self, start, target):
        open_set = [(0 + self.manhattan_distance(start, target), 0, start, [])] 
        g_scores = {start: 0}
        # Using a set for closed_set for faster lookups, path reconstruction done via path list
        # For A*, we need to store path to reconstruct.
        # Let's use a dictionary for came_from and reconstruct path at the end for `a_star_to_target`
        # Or, pass path list along, which is fine for non-huge mazes.
        
        # Path list method: (f_score, g_score, current_pos, path_list_to_current_pos)
        # The path_list_to_current_pos should store the path *up to* current_pos but not including it,
        # then append current_pos before adding to queue for neighbor.

        queue = [(0 + self.manhattan_distance(start, target), 0, start, [start])] # f, g, pos, path_taken
        visited_g_scores = {start: 0} # To keep track of best g_score to reach a node

        while queue:
            f, g, current, path = heapq.heappop(queue)

            if g > visited_g_scores.get(current, float('inf')): # Found a longer path to an already visited node
                continue

            if current == target: return path

            for neighbor in self.get_valid_neighbors(current):
                tentative_g = g + 1
                if tentative_g < visited_g_scores.get(neighbor, float('inf')):
                    visited_g_scores[neighbor] = tentative_g
                    h = self.manhattan_distance(neighbor, target)
                    heapq.heappush(queue, (tentative_g + h, tentative_g, neighbor, path + [neighbor]))
        return []


    def _a_star_solve_path(self, start_pos, current_keys_collected_count, keys_list_remaining):
        path_accumulator = []
        current_pos_sim = start_pos
        keys_collected_sim_count = current_keys_collected_count
        remaining_keys_sim_list = copy.deepcopy(keys_list_remaining)
        
        first_segment = True

        while keys_collected_sim_count < self.total_keys_required:
            if not remaining_keys_sim_list: break
            
            best_path_to_key_segment = []
            chosen_key_pos = None

            for key_pos_target in remaining_keys_sim_list:
                segment = self.a_star_to_target(current_pos_sim, key_pos_target)
                if segment and (not best_path_to_key_segment or len(segment) < len(best_path_to_key_segment)):
                    best_path_to_key_segment = segment
                    chosen_key_pos = key_pos_target
            
            if best_path_to_key_segment:
                path_accumulator.extend(best_path_to_key_segment if first_segment else best_path_to_key_segment[1:])
                first_segment = False
                current_pos_sim = chosen_key_pos
                remaining_keys_sim_list.remove(chosen_key_pos)
                keys_collected_sim_count += 1
            else:
                return [] 

        path_to_goal_segment = self.a_star_to_target(current_pos_sim, self.goal_pos)
        if path_to_goal_segment:
            path_accumulator.extend(path_to_goal_segment if first_segment else path_to_goal_segment[1:])
            
            if not path_accumulator and start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required:
                return [start_pos] # Already at goal, all keys collected
            return path_accumulator
        
        # If no path to goal, but all keys collected and already at goal (e.g. goal was the last key spot)
        elif not path_accumulator and current_pos_sim == self.goal_pos and keys_collected_sim_count >= self.total_keys_required:
            if start_pos == self.goal_pos: return [start_pos] # Started at goal, no keys needed
            # This case implies path_accumulator might be non-empty if keys were collected
            # If path_accumulator is empty, it means start_pos = first_key = ... = last_key = goal_pos
            # This is handled by the previous `if path_to_goal_segment` return [start_pos] line.

        # If goal is unreachable but keys are collected, this means failure
        # If goal is reachable, path_accumulator will be populated.
        # If start_pos == goal_pos and no keys required, A* to target will return [start_pos] -> handled.
        return [] if not path_accumulator and not (start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required) else path_accumulator


    def greedy_to_target(self, start, target): # Greedy Best First Search
        # (heuristic_cost, current_pos, path_list_to_current_pos)
        queue = [(self.manhattan_distance(start, target), start, [start])] 
        visited = {start} # Greedy only needs to visit each node once

        while queue:
            _, current, path = heapq.heappop(queue)

            if current == target: return path

            # Sort neighbors by heuristic to target for Greedy
            neighbors_sorted = sorted(
                self.get_valid_neighbors(current),
                key=lambda n: self.manhattan_distance(n, target)
            )

            for neighbor in neighbors_sorted:
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(queue, (self.manhattan_distance(neighbor, target), neighbor, path + [neighbor]))
        return []

    def _greedy_solve_path(self, start_pos, current_keys_collected_count, keys_list_remaining):
        path_accumulator = []
        current_pos_sim = start_pos
        keys_collected_sim_count = current_keys_collected_count
        remaining_keys_sim_list = copy.deepcopy(keys_list_remaining)
        
        first_segment = True

        while keys_collected_sim_count < self.total_keys_required:
            if not remaining_keys_sim_list: break
            
            # Greedy: pick closest key by heuristic first
            remaining_keys_sim_list.sort(key=lambda k_pos: self.manhattan_distance(current_pos_sim, k_pos))
            
            if not remaining_keys_sim_list: break 
            
            target_key = remaining_keys_sim_list[0] 
            segment = self.greedy_to_target(current_pos_sim, target_key)

            if segment:
                path_accumulator.extend(segment if first_segment else segment[1:])
                first_segment = False
                current_pos_sim = target_key # segment[-1] is target_key
                remaining_keys_sim_list.remove(target_key)
                keys_collected_sim_count += 1
            else:
                return [] # Cannot reach the "closest" key

        path_to_goal_segment = self.greedy_to_target(current_pos_sim, self.goal_pos)
        if path_to_goal_segment:
            path_accumulator.extend(path_to_goal_segment if first_segment else path_to_goal_segment[1:])
            if not path_accumulator and start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required:
                return [start_pos]
            return path_accumulator
        
        elif not path_accumulator and current_pos_sim == self.goal_pos and keys_collected_sim_count >= self.total_keys_required :
             if start_pos == self.goal_pos: return [start_pos]
        
        return [] if not path_accumulator and not (start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required) else path_accumulator


    def get_path_for_algorithm(self, algorithm_name):
        # Uses current game state (agent_pos, collected_keys, self.keys which are remaining keys)
        # Create deep copies of mutable state (self.keys) to pass to solvers,
        # as they might modify their received list.
        current_remaining_keys = copy.deepcopy(self.keys)
        if algorithm_name == "bfs":
            return self._bfs_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        elif algorithm_name == "dfs":
            return self._dfs_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        elif algorithm_name == "a_star":
            return self._a_star_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        elif algorithm_name == "greedy":
            return self._greedy_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        return []


    def find_optimal_algorithm(self):
        algorithms = ["bfs", "a_star", "greedy"] 
        best_algo = None
        best_moves = float('inf')
        
        original_env_state = self.snapshot_state()

        for algo in algorithms:
            # Pathfinding needs to use the current state of the game for this specific method
            # So, no reset to initial maze state here.
            path = self.get_path_for_algorithm(algo) 
            moves = len(path) -1 if path else float('inf')
            
            self.algorithm_stats[algo] = {"moves": moves if moves != float('inf') else -1, "path_length": moves if moves != float('inf') else -1}
            
            if moves != float('inf') and moves < best_moves:
                best_moves = moves
                best_algo = algo
        
        self.restore_state(original_env_state) 
        return best_algo if best_algo else "a_star" # Default to A* if no path found by any

    def snapshot_state(self):
        return {
            'agent_pos': self.agent_pos,
            'keys': copy.deepcopy(self.keys),
            'collected_keys': self.collected_keys,
            'total_keys_required': self.total_keys_required, # Important for sim
            'powerups': copy.deepcopy(self.powerups),
            'obstacles': copy.deepcopy(self.obstacles),
            'obstacle_direction': copy.deepcopy(self.obstacle_direction),
            'game_finished': self.game_finished,
            'score': self.score,
            'moves_taken': self.moves_taken,
            'wall_breaks': self.wall_breaks,
            'freeze_available': self.freeze_available,
            'reveal_path_available': self.reveal_path_available,
            'frozen_obstacles': self.frozen_obstacles,
            'frozen_time': self.frozen_time,
            'start_time': self.start_time,
            'grid': copy.deepcopy(self.grid),
            # Also snapshot initial configurations if they could change (they shouldn't post-generation)
            'initial_grid_config': copy.deepcopy(self.initial_grid_config),
            'initial_agent_pos': self.initial_agent_pos,
            'initial_goal_pos': self.initial_goal_pos,
            'initial_keys': copy.deepcopy(self.initial_keys),
            'initial_powerups': copy.deepcopy(self.initial_powerups),
            'initial_obstacles': copy.deepcopy(self.initial_obstacles),
            'initial_total_keys_required': self.initial_total_keys_required,
        }

    def restore_state(self, snapshot):
        self.agent_pos = snapshot['agent_pos']
        self.keys = snapshot['keys']
        self.collected_keys = snapshot['collected_keys']
        self.total_keys_required = snapshot['total_keys_required']
        self.powerups = snapshot['powerups']
        self.obstacles = snapshot['obstacles']
        self.obstacle_direction = snapshot['obstacle_direction']
        self.game_finished = snapshot['game_finished']
        self.score = snapshot['score']
        self.moves_taken = snapshot['moves_taken']
        self.wall_breaks = snapshot['wall_breaks']
        self.freeze_available = snapshot['freeze_available']
        self.reveal_path_available = snapshot['reveal_path_available']
        self.frozen_obstacles = snapshot['frozen_obstacles']
        self.frozen_time = snapshot['frozen_time']
        self.start_time = snapshot['start_time']
        self.grid = snapshot['grid']

        # Restore initial configurations as well, in case they were part of the snapshot
        # (e.g. if a comparison reset them temporarily)
        self.initial_grid_config = snapshot.get('initial_grid_config', self.initial_grid_config)
        self.initial_agent_pos = snapshot.get('initial_agent_pos', self.initial_agent_pos)
        self.initial_goal_pos = snapshot.get('initial_goal_pos', self.initial_goal_pos)
        self.initial_keys = snapshot.get('initial_keys', self.initial_keys)
        self.initial_powerups = snapshot.get('initial_powerups', self.initial_powerups)
        self.initial_obstacles = snapshot.get('initial_obstacles', self.initial_obstacles)
        self.initial_total_keys_required = snapshot.get('initial_total_keys_required', self.initial_total_keys_required)


    def simulate_ai(self, algorithm):
        original_state_snapshot = self.snapshot_state()
        
        # Reset relevant parts of state for simulation specific to this call
        # The environment (self) is already set to the desired starting point for this simulation
        # (e.g. absolute start by compare_algorithms_ui, or current player state by use_reveal_path)
        
        # Track sim-specific collections from zero for score calculation
        sim_collected_keys_count = 0
        sim_collected_powerups_count = 0
        
        # Store values from the state *before* this simulation run modified them
        # These are effectively the "starting conditions" for this simulation instance.
        # If compare_algorithms_ui called reset_state_for_replay(), then self.collected_keys starts at 0.
        # If called for "reveal path", it starts from player's current self.collected_keys.
        # The key is that get_path_for_algorithm uses self.collected_keys and self.keys as its starting point.
        
        self.moves_taken = 0 
        self.game_finished = False # Ensure game can be "finished" in sim
        # Score for simulation will be calculated based on actions within sim
        self.score = 0 # Reset score for this simulation run.
        
        # Lifelines are not used by AI in simulation
        # Obstacles are assumed static for the duration of path calculation and this simulation.

        calculated_path = self.get_path_for_algorithm(algorithm) # Uses current self.agent_pos, self.keys, self.collected_keys
        
        sim_path_followed = []

        if calculated_path and len(calculated_path) > 1: # Path has at least one move
            sim_path_followed.append(calculated_path[0]) # Start node
            for i in range(1, len(calculated_path)):
                pos = calculated_path[i]
                self.agent_pos = pos 
                self.moves_taken += 1
                sim_path_followed.append(pos)

                # Simulate pickups based on current state of self.keys and self.powerups
                if pos in self.keys: 
                    self.keys.remove(pos) # Modify temporary copy for this sim
                    self.collected_keys += 1 # This increments from the sim's starting self.collected_keys
                    sim_collected_keys_count +=1
                
                if pos in self.powerups: 
                    self.powerups.remove(pos)
                    # self.collected_powerups +=1 # Not strictly needed for score calc here
                    sim_collected_powerups_count +=1

                if pos == self.goal_pos and self.collected_keys >= self.total_keys_required:
                    # For simulation, don't use real time for finish_game's time bonus
                    self.finish_game(simulated_time_override=0) # Pass 0 to max out time bonus, or a fixed value
                                                                # Or, calculate score manually below
                    break 
        elif calculated_path and len(calculated_path) == 1 and calculated_path[0] == self.goal_pos: # Started at goal
             if self.collected_keys >= self.total_keys_required:
                self.finish_game(simulated_time_override=0)
             sim_path_followed.append(calculated_path[0])

        # Calculate a more representative score for simulation:
        # Score = (keys_score) + (powerups_score) + (win_bonus if won) + (move_bonus if won)
        sim_score_final = 0
        sim_score_final += sim_collected_keys_count * 50
        sim_score_final += sim_collected_powerups_count * 30
        
        if self.game_finished: # if finish_game was called
            sim_score_final += 500 # Base win bonus
            move_bonus = max(0, 500 - self.moves_taken * 5)
            sim_score_final += move_bonus
        
        sim_moves_final = self.moves_taken
        
        self.restore_state(original_state_snapshot) 
        
        return sim_moves_final, sim_score_final, sim_path_followed


class MazeApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("Maze Solver AI Challenge")
        self.root.configure(bg="#101010")  # Black background

        self.env = MazeEnvironment(GRID_WIDTH, GRID_HEIGHT)
        self.ai_animating = False
        self.ai_path_to_animate = []
        self.ai_animation_paused = False
        self.ai_current_path_taken = []
        self.wall_break_mode = False
        self.obstacle_move_interval = 2000
        
        self.compare_window = None
        self.compare_tree = None
        self.compare_update_id = None
                
        self.status_message = tk.StringVar(value="Welcome to the Maze Challenge!")

        # --- Main Layout ---
        self.main_frame = tk.Frame(root_tk, bg="#101010")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = tk.Frame(self.main_frame, bg="#101010")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.right_frame = tk.Frame(self.main_frame, width=500, bg="#181c1b")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.right_frame.pack_propagate(False)
        
        # --- Right Panel: Stats & feature_key ---
        self.right_center_container = tk.Frame(self.right_frame, bg="#181c1b")
        self.right_center_container.pack(expand=True, fill=tk.BOTH)

        # --- Top Controls (Left Frame) ---
        self.controls_frame = tk.LabelFrame(
            self.left_frame, text="AI Controls", bg="#181c1b", fg="#22c55e",
            padx=10, pady=10, font=("Arial", 10, "bold"), relief=tk.GROOVE
        )
        self.controls_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.controls_frame, text="Algorithm:", bg="#181c1b", fg="#22c55e").pack(side=tk.LEFT, padx=(0, 5))
        self.algorithm_var = tk.StringVar(value="a_star")
        self.algorithm_menu = ttk.Combobox(
            self.controls_frame, textvariable=self.algorithm_var,
            values=["bfs", "dfs", "a_star", "greedy"], state="readonly", width=10, font=("Arial", 10)
        )
        self.algorithm_menu.pack(side=tk.LEFT, padx=(0, 10))
        self.algorithm_menu.bind("<<ComboboxSelected>>", lambda e: self.canvas.focus_set())
        self.run_ai_button = tk.Button(
            self.controls_frame, text="Run AI", command=self.run_ai_agent,
            bg="#22c55e", fg="#101010", width=9, font=("Arial", 10, "bold"),
            relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.run_ai_button.pack(side=tk.LEFT, padx=3)

        self.pause_ai_button = tk.Button(
            self.controls_frame, text="Pause AI", command=self.toggle_ai_pause, state=tk.DISABLED,
            bg="#a3e635", fg="#101010", width=9, font=("Arial", 10, "bold"),
            relief=tk.RAISED, borderwidth=2, activebackground="#bef264"
        )
        self.pause_ai_button.pack(side=tk.LEFT, padx=3)

        self.optimal_button = tk.Button(
            self.controls_frame, text="Optimal Solver", command=self.run_optimal_solver,
            bg="#166534", fg="#e5ffe5", width=12, font=("Arial", 10, "bold"),
            relief=tk.RAISED, borderwidth=2, activebackground="#22c55e"
        )
        self.optimal_button.pack(side=tk.LEFT, padx=3)

        

        # --- Canvas (Left Frame) ---
        self.canvas_frame = tk.Frame(self.left_frame, bg="#101010", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=GRID_WIDTH * CELL_SIZE, height=GRID_HEIGHT * CELL_SIZE,
            bg="#181c1b", highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        # --- Lifelines (Left Frame) ---
        self.lifelines_frame = tk.LabelFrame(
            self.left_frame, text="Player Lifelines", bg="#181c1b", fg="#22c55e",
            padx=10, pady=10, font=("Arial", 10, "bold"), relief=tk.GROOVE
        )
        self.lifelines_frame.pack(fill=tk.X, pady=(0, 10))

        btn_font = ("Arial", 10, "bold")
        self.wall_break_button = tk.Button(
            self.lifelines_frame, text="Wall Break (0)", command=self.activate_wall_break_mode,
            bg="#22c55e", fg="#101010", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.wall_break_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.freeze_button = tk.Button(
            self.lifelines_frame, text="Freeze (0)", command=self.use_freeze_ui,
            bg="#22c55e", fg="#101010", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.freeze_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.reveal_path_button = tk.Button(
            self.lifelines_frame, text="Reveal Path (0)", command=self.use_reveal_path_ui,
            bg="#22c55e", fg="#101010", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.reveal_path_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Game/Reset Controls (Left Frame - Bottom) ---
        self.game_controls_frame = tk.Frame(self.left_frame, bg="#101010")
        self.game_controls_frame.pack(fill=tk.X, pady=(0, 5))

        self.replay_button = tk.Button(
            self.game_controls_frame, text="Replay Maze", command=self.replay_maze,
            bg="#166534", fg="#e5ffe5", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#22c55e"
        )
        self.replay_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.reset_button = tk.Button(
            self.game_controls_frame, text="New Maze", command=self.reset_maze_ui,
            bg="#dc2626", fg="#e5ffe5", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#991b1b"
        )
        self.reset_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Status Message (Left Frame - Bottom) ---
        self.status_label = tk.Label(
            self.left_frame, textvariable=self.status_message,
            bg="#181c1b", fg="#22c55e", relief=tk.SUNKEN, anchor="w", padx=5, font=("Arial", 10)
        )
        self.status_label.pack(fill=tk.X, ipady=3)

        

        stat_font = ("Arial", 11)
        self.stats_frame = tk.LabelFrame(
            self.right_center_container, text="Statistics", bg="#181c1b", padx=10, pady=10,
            relief=tk.RIDGE, bd=2, font=("Arial", 10, "bold"), fg="#22c55e"
        )
        self.stats_frame.pack(fill=tk.X, pady=(0, 10), expand=True)

        self.score_label = tk.Label(self.stats_frame, text="Score: 0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.score_label.pack(anchor="w", pady=1)
        self.moves_label = tk.Label(self.stats_frame, text="Moves: 0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.moves_label.pack(anchor="w", pady=1)
        self.keys_label = tk.Label(self.stats_frame, text="Keys: 0/0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.keys_label.pack(anchor="w", pady=1)
        self.powerups_label = tk.Label(self.stats_frame, text="Powerups: 0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.powerups_label.pack(anchor="w", pady=1)
        self.time_label = tk.Label(self.stats_frame, text="Time: 0s", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.time_label.pack(anchor="w", pady=1)

        # Feature Key
        self.feature_key_frame = tk.LabelFrame(
            self.right_center_container, text="Feature Key", bg="#181c1b", padx=10, pady=10,
            relief=tk.RIDGE, bd=2, font=("Arial", 10, "bold"), fg="#22c55e"
        )
        self.feature_key_frame.pack(fill=tk.X, pady=5, expand=True)

        feature_key_items = {AGENT: "Agent", GOAL: "Goal", KEY: "Key", POWERUP: "Power-up",
                        OBSTACLE: "Obstacle", WALL: "Wall", PATH: "Path",
                        VISITED_PATH: "AI Trace", REVEALED_STEP: "Revealed Path"}
        feature_key_font = ("Arial", 10)
        for item, desc in feature_key_items.items():
            item_frame = tk.Frame(self.feature_key_frame, bg="#181c1b")
            item_frame.pack(anchor="w", fill=tk.X, pady=1)
            tk.Label(item_frame, text="  ", bg=COLOR_MAP.get(item, "gray"), relief=tk.SOLID, borderwidth=1).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(item_frame, text=f": {desc}", bg="#181c1b", font=feature_key_font, fg="#e5ffe5").pack(side=tk.LEFT)

        # --- Live Comparison Table (Bottom Right, but vertically centered with others) ---
        compare_frame = tk.LabelFrame(
            self.right_center_container, text="Live Algorithm Comparison", bg="#181c1b", fg="#22c55e",
            padx=8, pady=6, font=("Arial", 10, "bold"), relief=tk.GROOVE
        )
        compare_frame.pack(fill=tk.X, pady=(10, 0), padx=2, expand=True)

        cols = ["Algorithm", "Moves", "Sim. Score", "Powerups", "Time (s)"]
        
        # Treeview Table Style
        style = ttk.Style(self.right_frame)
        style.theme_use('clam')
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#22c55e", foreground="#101010")
        style.configure("Treeview", rowheight=22, font=("Arial", 9), background="#101010", fieldbackground="#101010", foreground="#e5ffe5")

        self.compare_tree = ttk.Treeview(compare_frame, columns=cols, show="headings", selectmode="none", height=5)
        for col in cols:
            self.compare_tree.heading(col, text=col)
            self.compare_tree.column(col, width=90, anchor='center', stretch=tk.YES)
        self.compare_tree.pack(fill=tk.BOTH, expand=True)
        self.compare_tree.tag_configure("highlight", background="#166534", font=("Arial", 9, "bold"), foreground="#e5ffe5")

        self.update_compare_results()  # Start periodic update for embedded table        
        self.root.bind("<KeyPress>", self.handle_keypress)
        self.draw_maze()
        self.update_stats_display()
        self._obstacle_move_id = self.root.after(self.obstacle_move_interval, self.periodic_obstacle_move)
        self._periodic_update_id = self.root.after(100, self.periodic_update)

        self.status_message.set("Maze generated. Use arrow keys or AI to solve!")

    def periodic_update(self):
        if not self.env.game_finished:
            if self.env.start_time:
                elapsed_time = time.time() - self.env.start_time
                self.time_label.config(text=f"Time: {int(elapsed_time)}s")
            
            unfreeze_msg = self.env.update_lifeline_timers()
            if unfreeze_msg:
                self.status_message.set(unfreeze_msg)
                self.draw_maze() 
                self.update_stats_display() # Freeze button state might change
            
            self._periodic_update_id = self.root.after(100, self.periodic_update)

    def draw_maze(self):
        self.canvas.delete("all")
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell_type = PATH 
                if self.env.grid[y][x] == 1: cell_type = WALL
                
                color_to_use = COLOR_MAP[cell_type] # Default
                if cell_type == PATH: # Only draw special paths on actual paths
                    if (x,y) in self.ai_current_path_taken :
                        color_to_use = COLOR_MAP[VISITED_PATH]
                    elif (x,y) in self.env.revealed_path_segments:
                        color_to_use = COLOR_MAP[REVEALED_STEP]

                self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                             fill=color_to_use, outline="#CCCCCC", width=0.5) 
        
        items_to_draw = [
            (self.env.keys, KEY),
            (self.env.powerups, POWERUP),
            (self.env.obstacles, OBSTACLE),
            ([self.env.goal_pos], GOAL), 
            ([self.env.agent_pos], AGENT),
        ]
        item_margin = CELL_SIZE * 0.15 # Slightly larger items
        for item_list, item_type in items_to_draw:
            for pos_x, pos_y in item_list:
                # Ensure item is on a path, or is agent/obstacle (can be on paths by definition)
                if self.env.grid[pos_y][pos_x] == 0 or item_type == AGENT or item_type == OBSTACLE : 
                    self.canvas.create_rectangle(pos_x * CELL_SIZE + item_margin, 
                                                 pos_y * CELL_SIZE + item_margin,
                                                 (pos_x + 1) * CELL_SIZE - item_margin, 
                                                 (pos_y + 1) * CELL_SIZE - item_margin,
                                                 fill=COLOR_MAP[item_type], outline=COLOR_MAP[item_type])
        
        if self.env.game_finished:
            center_x = self.env.width * CELL_SIZE / 2
            center_y = self.env.height * CELL_SIZE / 2
            rect_width = 480
            rect_height = 90
            rect_color = "#166534"   # Deep green for dark mode
            outline_color = "#22c55e"  # Bright green accent
            text_color = "#e5ffe5"     # Light greenish-white

            # Draw rectangle (simulate transparency by using a lighter color)
            self.canvas.create_rectangle(
                center_x - rect_width/2, center_y - rect_height/2,
                center_x + rect_width/2, center_y + rect_height/2,
                fill=rect_color, outline=outline_color, width=6
            )
            # Draw the text on top
            self.canvas.create_text(
                center_x, center_y,
                text="GOAL Reached!",
                font=("Impact", 36, "bold"),
                fill=text_color
            )
        


    def update_stats_display(self):
        self.score_label.config(text=f"Score: {self.env.score}")
        self.moves_label.config(text=f"Moves: {self.env.moves_taken}")
        self.keys_label.config(text=f"Keys: {self.env.collected_keys}/{self.env.total_keys_required}")
        self.powerups_label.config(text=f"Powerups: {self.env.collected_powerups}")
        
        can_use_lifelines = not self.ai_animating and not self.env.game_finished

        self.wall_break_button.config(text=f"Wall Break ({self.env.wall_breaks})", 
                                      state=tk.NORMAL if self.env.wall_breaks > 0 and can_use_lifelines else tk.DISABLED)
        
        if self.env.frozen_obstacles:
            remaining_freeze_time = 10 - (time.time() - self.env.frozen_time)
            self.freeze_button.config(text=f"Frozen ({int(max(0,remaining_freeze_time))}s)", state=tk.DISABLED)
        else:
            self.freeze_button.config(text=f"Freeze ({self.env.freeze_available})",
                                      state=tk.NORMAL if self.env.freeze_available > 0 and can_use_lifelines else tk.DISABLED)
        
        self.reveal_path_button.config(text=f"Reveal Path ({self.env.reveal_path_available})",
                                       state=tk.NORMAL if self.env.reveal_path_available > 0 and can_use_lifelines else tk.DISABLED)
        
    def handle_keypress(self, event):
        if self.env.game_finished or self.ai_animating or self.wall_break_mode: # Disable kbd if wall_break active
            return

        dx, dy = 0, 0
        if event.keysym == "Up": dy = -1
        elif event.keysym == "Down": dy = 1
        elif event.keysym == "Left": dx = -1
        elif event.keysym == "Right": dx = 1
        else: return

        ax, ay = self.env.agent_pos
        new_pos = (ax + dx, ay + dy)

        if self.env.is_valid_move(new_pos):
            message = self.env.move_agent(new_pos)
            if message: self.status_message.set(message)
            else: self.status_message.set(f"Moved to ({new_pos[0]}, {new_pos[1]})")
            
            self.draw_maze()
            self.update_stats_display()

            if self.env.game_finished:
                self.handle_game_end()
        else:
            self.status_message.set("Invalid move or blocked.")

    def handle_canvas_click(self, event):
        if not self.wall_break_mode or self.ai_animating or self.env.game_finished:
            return

        grid_x, grid_y = event.x // CELL_SIZE, event.y // CELL_SIZE
        
        success, message = self.env.break_wall((grid_x, grid_y))
        self.status_message.set(message)
        
        if success:
            self.draw_maze()
        
        self.wall_break_mode = False 
        self.canvas.config(cursor="")
        self.wall_break_button.config(relief=tk.RAISED)
        self.update_stats_display() # Reflect wall break usage

    def activate_wall_break_mode(self):
        if self.env.wall_breaks > 0 and not self.ai_animating and not self.env.game_finished:
            self.wall_break_mode = not self.wall_break_mode # Toggle
            if self.wall_break_mode:
                self.status_message.set("WALL BREAK: Click on a wall to break. Press key to cancel.")
                self.canvas.config(cursor="hand2")
                self.wall_break_button.config(relief=tk.SUNKEN)
            else:
                self.status_message.set("Wall break mode deactivated.")
                self.canvas.config(cursor="")
                self.wall_break_button.config(relief=tk.RAISED)
        else:
            self.status_message.set("Cannot use Wall Break now.")
            
    def use_freeze_ui(self):
        if self.ai_animating or self.env.game_finished: return
        success, message = self.env.use_freeze_obstacles()
        self.status_message.set(message)
        if success:
            self.update_stats_display()
            self.draw_maze() 

    def use_reveal_path_ui(self):
        if self.ai_animating or self.env.game_finished: return
        selected_algo = self.algorithm_var.get()
        success, message = self.env.use_reveal_path(selected_algo)
        self.status_message.set(message)
        if success:
            self.update_stats_display()
            self.draw_maze()

    def reset_maze_ui(self, show_prompt=True):
        if self.ai_animating:
            self.status_message.set("AI is running. Cannot reset now.")
            return

        if show_prompt:
            if not messagebox.askyesno("Reset Maze", "Start a new random maze? Your progress will be lost."):
                return
        
        self.env.generate_maze() # This creates new initial config and resets state
        self.common_reset_ui_actions("New maze generated. Good luck!")

       
    def replay_maze(self):
        if self.ai_animating:
            self.status_message.set("AI is running. Cannot replay now.")
            return
        if self.env.initial_grid_config is None:
            messagebox.showinfo("Replay Maze", "No maze has been generated yet to replay.")
            return
        if not messagebox.askyesno("Replay Maze", "Restart the current maze? Your progress will be lost."):
            return

        self.env.reset_state_for_replay() # Resets to the current maze's initial state
        self.common_reset_ui_actions("Maze replayed. Try again!")

    def common_reset_ui_actions(self, status_msg):
        self.ai_current_path_taken = []
        # self.env.revealed_path_segments are reset by env.reset_state_for_replay() or _initialize_game_state()
        self.status_message.set(status_msg)
        self.draw_maze()
        self.update_stats_display() # Also updates time to 0s initially
        self.time_label.config(text="Time: 0s") # Explicitly reset time label
        self.enable_controls() # Ensure all controls are correctly enabled/disabled

        # Cancel and restart periodic updates to ensure they use fresh state
        if hasattr(self, '_periodic_update_id') and self._periodic_update_id:
            self.root.after_cancel(self._periodic_update_id)
        self._periodic_update_id = self.root.after(100, self.periodic_update)
        
        if hasattr(self, '_obstacle_move_id') and self._obstacle_move_id:
            self.root.after_cancel(self._obstacle_move_id)
        self._obstacle_move_id = self.root.after(self.obstacle_move_interval, self.periodic_obstacle_move)
        
        self.wall_break_mode = False # Ensure wall break mode is off
        self.canvas.config(cursor="")
        self.wall_break_button.config(relief=tk.RAISED)


    def run_ai_agent(self, algorithm_name=None):
        if self.ai_animating or self.env.game_finished:
            self.status_message.set("Cannot run AI now (already running or game over).")
            return

        if algorithm_name is None:
            algorithm_name = self.algorithm_var.get()
        
        self.status_message.set(f"AI ({algorithm_name.upper()}) is calculating path...")
        self.root.update_idletasks() 

        # Path is calculated from current player state
        path = self.env.get_path_for_algorithm(algorithm_name)
        
        if path and len(path) > 1: # Path must involve at least one move
            self.ai_path_to_animate = path 
            self.ai_animating = True
            self.ai_animation_paused = False
            self.ai_current_path_taken = [self.env.agent_pos] 
            self.disable_controls_for_ai()
            self.status_message.set(f"AI ({algorithm_name.upper()}) running. Path moves: {len(path)-1}")
            self.animate_ai_step()
        elif path and len(path) == 1 and path[0] == self.env.agent_pos: # AI is already at goal / done
            self.status_message.set(f"AI ({algorithm_name.upper()}): Already at destination or no moves needed.")
        else:
            self.status_message.set(f"AI ({algorithm_name.upper()}) could not find a path from current state.")

    def animate_ai_step(self):
        if not self.ai_animating: return # AI animation was stopped/completed
        if self.ai_animation_paused:
            self.root.after(100, self.animate_ai_step) 
            return

        if not self.ai_path_to_animate: 
            self.ai_animating = False
            is_game_won = self.env.agent_pos == self.env.goal_pos and self.env.collected_keys >= self.env.total_keys_required
            self.status_message.set(f"AI finished. Moves: {self.env.moves_taken}. {'Goal Reached!' if is_game_won else ''}")
            self.enable_controls()
            if self.env.game_finished: self.handle_game_end()
            return

        # Handle the first node of ai_path_to_animate which is the start pos for this segment
        if self.ai_path_to_animate[0] == self.env.agent_pos:
            self.ai_path_to_animate.pop(0) # Remove current agent pos if it's head of list
            if not self.ai_path_to_animate: # Path was just [current_pos]
                self.animate_ai_step() # Re-check conditions, should lead to completion
                return
        
        next_pos = self.ai_path_to_animate.pop(0)
        
        # Path was pre-calculated. If an obstacle moved into path, AI might fail.
        # For this version, AI pathing assumes static obstacles during its single path computation.
        # A more advanced AI would re-plan or have dynamic obstacle avoidance.
        if not self.env.is_valid_move(next_pos):
            self.status_message.set(f"AI stopped: Path blocked at {next_pos}. Re-planning needed (not implemented).")
            self.ai_animating = False
            self.enable_controls()
            # Optionally, could try to re-run AI from current pos: self.run_ai_agent(self.algorithm_var.get())
            return

        message = self.env.move_agent(next_pos) 
        self.ai_current_path_taken.append(next_pos)
        
        if message: self.status_message.set(f"AI: {message.split('!')[0]}") # Shorten message
        
        self.draw_maze()
        self.update_stats_display()

        if self.env.game_finished:
            self.ai_animating = False # Ensure it's set before handle_game_end
            self.handle_game_end()
            return

        self.root.after(180, self.animate_ai_step) # Animation speed (ms per step)

    def toggle_ai_pause(self):
        if not self.ai_animating: return
        self.ai_animation_paused = not self.ai_animation_paused
        if self.ai_animation_paused:
            self.pause_ai_button.config(text="Resume AI", bg="#4CAF50", fg="white") # Green for resume
            self.status_message.set("AI Paused. Obstacles may still move.")
        else:
            self.pause_ai_button.config(text="Pause AI", bg="#FFC107", fg="black") # Amber for pause
            self.status_message.set("AI Resumed.")
            # self.animate_ai_step() # Animation loop will pick it up via the periodic check

    def run_optimal_solver(self):
        if self.ai_animating or self.env.game_finished: return
        self.status_message.set("Finding optimal algorithm for current state...")
        self.root.update_idletasks()
        
        # find_optimal_algorithm uses current game state
        optimal_algo = self.env.find_optimal_algorithm()
        if optimal_algo:
            self.status_message.set(f"Optimal from here: {optimal_algo.upper()}. Running...")
            self.algorithm_var.set(optimal_algo)
            self.run_ai_agent(optimal_algo)
        else:
            self.status_message.set("Could not determine optimal algorithm or no path found.")

   

    
    def update_compare_results(self):
        results = []
        algorithms_to_compare = ["bfs", "a_star", "greedy", "dfs"]
        current_player_state_snapshot = self.env.snapshot_state()
        best_idx = -1
        best_moves = float('inf')
        best_score = float('-inf')
        best_time = float('inf')

        for idx, algo_name in enumerate(algorithms_to_compare):
            start_sim_time = time.perf_counter()
            moves, sim_score, sim_path = self.env.simulate_ai(algo_name)
            sim_duration = time.perf_counter() - start_sim_time
            powerups_collected = 0
            if sim_path:
                powerups_collected = len([pos for pos in sim_path if pos in self.env.initial_powerups])
            results.append({
                "Algorithm": algo_name.upper(),
                "Moves": moves if moves >= 0 else "N/A",
                "Sim. Score": sim_score if moves >= 0 else "N/A",
                "Powerups": powerups_collected if moves >= 0 else "N/A",
                "Time (s)": f"{sim_duration:.3f}" if moves >= 0 else "N/A"
            })
            if moves >= 0:
                if (moves < best_moves or
                    (moves == best_moves and sim_score > best_score) or
                    (moves == best_moves and sim_score == best_score and sim_duration < best_time)):
                    best_moves = moves
                    best_score = sim_score
                    best_time = sim_duration
                    best_idx = idx
        self.env.restore_state(current_player_state_snapshot)

        # Update treeview
        for i in self.compare_tree.get_children():
            self.compare_tree.delete(i)
        for idx, res_item in enumerate(results):
            tags = ()
            if best_idx is not None and idx == best_idx:
                tags = ("highlight",)
            self.compare_tree.insert("", "end", values=tuple(res_item[c] for c in ["Algorithm", "Moves", "Sim. Score", "Powerups", "Time (s)"]), tags=tags)

        # Schedule next update
        self.compare_update_id = self.root.after(500, self.update_compare_results)
    def close_compare_window(self):
        if self.compare_update_id:
            self.root.after_cancel(self.compare_update_id)
            self.compare_update_id = None
        if self.compare_window:
            self.compare_window.destroy()
        self.compare_window = None
    
    def display_comparison_results(self, results, highlight_idx=None):
        top = tk.Toplevel(self.root)
        top.title("Algorithm Comparison Results (Current State)")
        top.geometry("520x250")
        top.configure(bg="#E0E0E0")

        tk.Label(top, text="Comparison based on solving from the agent's current position and state.",
                font=("Arial", 9), bg="#E0E0E0", fg="#333333").pack(pady=(5,0))

        cols = ["Algorithm", "Moves", "Sim. Score", "Powerups", "Time (s)"]
        tree_frame = tk.Frame(top)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        style = ttk.Style(top)
        style.theme_use('clam')
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#3da9fc", foreground="white")
        style.configure("Treeview", rowheight=25, font=("Arial", 9), background="#f7f7ff", fieldbackground="#f7f7ff")
        style.map("Highlight.Treeview", background=[('selected', '#f6c90e')])

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="none")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center', stretch=tk.YES)

        for idx, res_item in enumerate(results):
            tags = ()
            if highlight_idx is not None and idx == highlight_idx:
                tags = ("highlight",)
            tree.insert("", "end", values=tuple(res_item[c] for c in cols), tags=tags)

        tree.tag_configure("highlight", background="#B2FFB2", font=("Arial", 9, "bold"))

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Button(top, text="Close", command=top.destroy, bg="#D3D3D3", font=("Arial",9,"bold"), relief=tk.RAISED, borderwidth=2).pack(pady=10)
        top.transient(self.root)
        top.grab_set()


    def periodic_obstacle_move(self):
        # Obstacles move even if AI is paused, but not if AI is actively animating its pre-calculated path step.
        # Or, allow them to move if AI is paused but not if AI is running a step.
        # Current logic: move if not game_finished AND (not ai_animating OR ai_animation_paused)
        # For simplicity, let's stick to: obstacles move if player is in control or AI is paused.
        # If AI is actively taking a step (self.ai_animating is True and self.ai_animation_paused is False),
        # obstacles should ideally not move in that exact moment to respect AI's pre-calculated path.
        # However, they *can* move between AI steps if the animation delay is long.
        # The current obstacle_move_interval is independent of AI animation step.
        
        # Let's allow obstacles to move unless AI is in the middle of its non-paused animation sequence.
        # This means if AI is paused, obstacles CAN move.
        can_obstacles_move = not self.env.game_finished and \
                             (not self.ai_animating or self.ai_animation_paused)

        if can_obstacles_move:
            collision = self.env.move_obstacles()
            if collision: 
                self.status_message.set(f"Ouch! Obstacle collision! Score -100. Score: {self.env.score}")
                # Could end game: self.env.finish_game(); self.handle_game_end("Hit by obstacle!")
            self.draw_maze()
            self.update_stats_display() # In case score changed
        
        # Keep scheduling obstacle moves as long as game is not finished.
        # This ensures obstacles resume if AI finishes or game is reset.
        if not self.env.game_finished : 
             self._obstacle_move_id = self.root.after(self.obstacle_move_interval, self.periodic_obstacle_move)


    def handle_game_end(self, custom_message=None):
        # This function is called when game ends, either by player or AI.
        self.ai_animating = False # Stop any AI animation
        self.ai_animation_paused = False # Reset pause state
        self.ai_path_to_animate = [] # Clear path
        
        # Ensure all controls are re-enabled, respecting current game item counts for lifelines
        self.enable_controls() 
        
        final_message = f"Game Over! Final Score: {self.env.score}, Moves: {self.env.moves_taken}."
        if self.env.start_time:
            final_time = int(time.time() - self.env.start_time)
            final_message += f" Time: {final_time}s."

        if custom_message: final_message = custom_message + "\n" + final_message
        
        self.status_message.set(final_message.split('\n')[0]) 
        self.draw_maze() # Show "GAME OVER" text on canvas
        self.update_stats_display() # Update final stats on labels

        # Stop periodic updates that run during active gameplay
        if hasattr(self, '_periodic_update_id') and self._periodic_update_id:
            self.root.after_cancel(self._periodic_update_id)
            self._periodic_update_id = None # Clear the ID
        if hasattr(self, '_obstacle_move_id') and self._obstacle_move_id:
            self.root.after_cancel(self._obstacle_move_id)
            self._obstacle_move_id = None # Clear the ID

        action = messagebox.askquestion("Game Over", f"{final_message}\n\nReplay this maze?", 
                                        icon='info', type=messagebox.YESNOCANCEL, parent=self.root)
        if action == 'yes':
            self.replay_maze()
        elif action == 'no':
            self.reset_maze_ui(show_prompt=False) 
        else: # Cancel or closed dialog
            self.status_message.set("Game ended. Choose Replay or New Maze to continue.")
            # UI remains interactive for looking at final state or manual reset/replay.

    def disable_controls_for_ai(self):
        self.run_ai_button.config(state=tk.DISABLED)
        self.optimal_button.config(state=tk.DISABLED)
        # self.compare_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        
        # Disable lifelines as AI path is pre-computed and shouldn't be interfered with
        self.wall_break_button.config(state=tk.DISABLED)
        self.freeze_button.config(state=tk.DISABLED)
        self.reveal_path_button.config(state=tk.DISABLED)
        
        # If wall break mode was active, deactivate it
        if self.wall_break_mode:
            self.wall_break_mode = False
            self.canvas.config(cursor="")
            self.wall_break_button.config(relief=tk.RAISED) # Reset visual state

        self.algorithm_menu.config(state=tk.DISABLED)
        self.pause_ai_button.config(state=tk.NORMAL, text="Pause AI", bg="#FFC107", fg="black")
        
    def enable_controls(self):
        self.run_ai_button.config(state=tk.NORMAL)
        self.optimal_button.config(state=tk.NORMAL)
        # self.compare_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.replay_button.config(state=tk.NORMAL)
        self.algorithm_menu.config(state="readonly")
        self.pause_ai_button.config(state=tk.DISABLED, text="Pause AI", bg="#FFC107", fg="black") # Default state for pause
        
        # Lifeline button states are handled by update_stats_display based on availability
        self.update_stats_display() 


if __name__ == "__main__":
    main_root = tk.Tk()
    # Basic theming for ttk widgets if available
    try:
        s = ttk.Style()
        s.theme_use('clam') # 'clam', 'alt', 'default', 'classic' are common
    except tk.TclError:
        print("Ttk themes not available or 'clam' theme failed. Using default.")
        
    app = MazeApp(main_root)
    main_root.mainloop()
