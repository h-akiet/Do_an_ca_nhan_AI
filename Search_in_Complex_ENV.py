import numpy as np
import itertools
import pygame
import logging
from copy import copy
from queue import PriorityQueue
from random import choice, randint
from typing import Tuple, Union, Set, Dict, List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FPS = 60

try:
    pygame.init()
    logger.info("Pygame initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Pygame: {e}")
    raise

# Display settings
PUZZLES_PER_ROW = 8
PUZZLES_PER_COL = 4
TILE_SIZE = 50
PUZZLE_WIDTH = TILE_SIZE * 3
PUZZLE_HEIGHT = TILE_SIZE * 3
PADDING = 10
WIDTH = PUZZLES_PER_ROW * (PUZZLE_WIDTH + PADDING) + PADDING
HEIGHT = PUZZLES_PER_COL * (PUZZLE_HEIGHT + PADDING) + PADDING + 100
try:
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle-Nguyễn Hoàng Anh Kiệt-23110247")
    font = pygame.font.SysFont('arial', 24)
    title_font = pygame.font.SysFont('arial', 30, bold=True)
    logger.info("Pygame display and font set up successfully.")
except Exception as e:
    logger.error(f"Error setting up Pygame display: {e}")
    raise

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHT_GRAY = (200, 200, 200)
DARK_GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)

move_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

class EightPuzzle:
    def __init__(self, initial_state):
        self.state = np.array(initial_state).reshape(3, 3)
        self.goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape(3, 3)
        self.actions = ['up', 'down', 'left', 'right']
        self._manhattan_cache = None
        self._misplaced_cache = None
        if sorted(self.state.flatten().tolist()) != list(range(9)):
            logger.error(f"Invalid initial state: {self.state.flatten()}")
            raise ValueError(f"Invalid initial state: {self.state.flatten()}")

    def get_blank_pos(self):
        blank = np.argwhere(self.state == 0)
        if blank.size == 0:
            logger.error("No blank tile (0) found in state")
            raise ValueError("No blank tile (0) found")
        return tuple(blank[0])

    def is_valid_action(self, action, blank_pos):
        r, c = blank_pos
        valid = not (
            (action == 'up' and r == 0) or
            (action == 'down' and r == 2) or
            (action == 'left' and c == 0) or
            (action == 'right' and c == 2)
        )
        return valid

    def move(self, action):
        new_state = self.state.copy()
        r, c = self.get_blank_pos()
        if not self.is_valid_action(action, (r, c)):
            logger.warning(f"Invalid action {action} at blank_pos ({r}, {c})")
            return new_state
        if action == 'up':
            new_state[r, c], new_state[r-1, c] = new_state[r-1, c], new_state[r, c]
        elif action == 'down':
            new_state[r, c], new_state[r+1, c] = new_state[r+1, c], new_state[r, c]
        elif action == 'left':
            new_state[r, c], new_state[r, c-1] = new_state[r, c-1], new_state[r, c]
        elif action == 'right':
            new_state[r, c], new_state[r, c+1] = new_state[r, c+1], new_state[r, c]
        self._manhattan_cache = None
        self._misplaced_cache = None
        logger.debug(f"Applied action {action}: {new_state.flatten()}")
        return new_state

    def get_observation(self):
        r, c = self.get_blank_pos()
        obs = [-1] * 4  
        if r > 0: obs[0] = self.state[r-1, c]
        if r < 2: obs[1] = self.state[r+1, c]
        if c > 0: obs[2] = self.state[r, c-1]
        if c < 2: obs[3] = self.state[r, c+1]
        return ((r, c), tuple(obs))

    def manhattan_distance(self):
        if self._manhattan_cache is not None:
            return self._manhattan_cache
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.state[i, j] != 0:
                    val = self.state[i, j]
                    goal_r, goal_c = divmod(val-1, 3) if val != 0 else (2, 2)
                    distance += abs(i - goal_r) + abs(j - goal_c)
        self._manhattan_cache = distance
        return distance

    def misplaced_tiles(self):
        if self._misplaced_cache is not None:
            return self._misplaced_cache
        count = np.sum((self.state != self.goal_state) & (self.state != 0))
        self._misplaced_cache = count
        return count

    def is_goal(self):
        return np.array_equal(self.state, self.goal_state)

    def is_solvable(self):
        flat_state = self.state.flatten()
        inversions = 0
        for i in range(9):
            if flat_state[i] == 0:
                continue
            for j in range(i + 1, 9):
                if flat_state[j] == 0:
                    continue
                if flat_state[i] > flat_state[j]:
                    inversions += 1
        return inversions % 2 == 0

    @staticmethod
    def move_tuple(state: Tuple[int, ...], action: str) -> Union[Tuple[int, ...], None]:
        if 0 not in state:
            logger.error(f"Invalid state, no blank tile: {state}")
            return None
        idx = state.index(0)
        row, col = divmod(idx, 3)
        dr, dc = move_map[action]
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state = list(state)
            new_state[idx], new_state[new_idx] = new_state[new_idx], new_state[new_idx]
            return tuple(new_state)
        return None

    @staticmethod
    def result_fn(states: Set[Tuple[int]], action: str) -> Set[Tuple[int]]:
        next_states = set()
        for s in states:
            res = EightPuzzle.move_tuple(s, action)
            if res:
                next_states.add(res)
        return next_states

def and_or_search(initial_belief: Set[Tuple[int]], max_depth: int = 30) -> Union[List[str], None]:
    GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def or_search(belief: Set[Tuple[int]], path: List[frozenset]) -> Union[List[str], str]:
        logger.debug(f"OR-SEARCH: belief size = {len(belief)}")
        for s in belief:
            logger.debug(f"  Checking if state {s} is goal: {s == GOAL_STATE}")
        if any(s == GOAL_STATE for s in belief):
            return []

        if frozenset(belief) in path:
            logger.debug("  Cycle detected in OR-SEARCH, skipping")
            return 'failure'

        if len(path) >= max_depth:
            logger.debug("  Max depth reached in OR-SEARCH")
            return 'failure'

        action_scores = []
        for action in move_map:
            next_belief = EightPuzzle.result_fn(belief, action)
            if not next_belief:
                logger.debug(f"    No resulting states for action {action}")
                continue
            heuristic = 0
            for s in next_belief:
                puzzle = EightPuzzle(np.array(s).reshape(3, 3))
                heuristic += 0.7 * puzzle.manhattan_distance() + 0.3 * puzzle.misplaced_tiles()
            heuristic /= len(next_belief) if next_belief else 1
            action_scores.append((action, heuristic))

        action_scores.sort(key=lambda x: x[1])

        for action, _ in action_scores:
            next_belief = EightPuzzle.result_fn(belief, action)
            logger.debug(f"  Trying action '{action}' → next belief size: {len(next_belief)}")
            for s in next_belief:
                if 0 not in s or sorted(s) != list(range(9)):
                    logger.error(f"Invalid state in next_belief: {s}")
                    return 'failure'
            plan = and_search(next_belief, path + [frozenset(belief)])
            if plan != 'failure':
                return [action] + plan

        logger.debug("  No valid action found in OR-SEARCH")
        return 'failure'

    def and_search(belief: Set[Tuple[int]], path: List[frozenset]) -> Union[List[str], str]:
        logger.debug(f"AND-SEARCH: belief size = {len(belief)}")
        successful_plans = []

        for s in belief:
            plan = or_search({s}, path)
            if plan == 'failure':
                logger.debug(f"    State {s} failed in OR-SEARCH")
                continue
            successful_plans.append(plan)

        if not successful_plans:
            if len(belief) == 1:
                logger.debug("  Single state failed — trying fallback attempt on lone state")
                s = next(iter(belief))
                return or_search({s}, path)
            return 'failure'

        if len(successful_plans) > 1:
            successful_plans.sort(key=len)
        first = successful_plans[0]
        if all(p == first for p in successful_plans):
            return first

        logger.debug("  Multiple differing plans found, selecting shortest one")
        return first

    logger.info(f"Starting AND-OR Search with {len(initial_belief)} belief states")
    plan = or_search(initial_belief, [])
    if plan == 'failure':
        logger.warning("AND-OR Search failed to find a conformant plan")
        return None
    logger.info(f"AND-OR Search succeeded with plan: {plan}")
    return plan

def non_observation_search(initial_belief: Set[Tuple[int]], max_depth: int = 30) -> Union[List[str], None]:
    logger.info(f"Starting Non-Observation search with {len(initial_belief)} belief states")
    GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    actions = ['up', 'down', 'left', 'right']
    visited = set() 
    pq = PriorityQueue()

    initial_heuristic = 0
    for state in initial_belief:
        puzzle = EightPuzzle(np.array(state).reshape(3, 3))
        initial_heuristic += 0.7 * puzzle.manhattan_distance() + 0.3 * puzzle.misplaced_tiles()
    initial_heuristic /= len(initial_belief) if initial_belief else 1
    pq.put((initial_heuristic, 0, initial_belief, []))   
    visited.add(frozenset(initial_belief))

    while not pq.empty():
        f_score, cost, belief, action_sequence = pq.get()
        logger.debug(f"Exploring belief with {len(belief)} states, cost={cost}, f_score={f_score}")

        if all(state == GOAL_STATE for state in belief):
            logger.info(f"Non-Observation search succeeded with plan: {action_sequence}")
            return action_sequence

        if cost >= max_depth:
            logger.debug(f"Reached max depth {max_depth}, skipping")
            continue

        for action in actions:
            next_belief = EightPuzzle.result_fn(belief, action)
            if not next_belief:
                logger.debug(f"No resulting states for action {action}")
                continue

            valid = True
            for s in next_belief:
                if 0 not in s or sorted(s) != list(range(9)):
                    logger.error(f"Invalid state in next_belief: {s}")
                    valid = False
                    break
            if not valid:
                continue

            belief_key = frozenset(next_belief)
            if belief_key in visited:
                logger.debug(f"Belief {belief_key} already visited, skipping")
                continue

            heuristic = 0
            for state in next_belief:
                puzzle = EightPuzzle(np.array(state).reshape(3, 3))
                heuristic += 0.7 * puzzle.manhattan_distance() + 0.3 * puzzle.misplaced_tiles()
            heuristic /= len(next_belief) if next_belief else 1

           
            next_cost = cost + 1
            next_f_score = next_cost + heuristic

            pq.put((next_f_score, next_cost, next_belief, action_sequence + [action]))
            visited.add(belief_key)
            logger.debug(f"Added action '{action}' to queue, next_belief size={len(next_belief)}, f_score={next_f_score}")

    logger.warning("Non-Observation search failed to find a conformant plan")
    return None

def generate_possible_states(partial_state, num_states, include_state=None, observation=None):
    logger.info(f"Generating {num_states} possible states for partial_state: {partial_state}")
    partial_state = np.array(partial_state).flatten()
    
    known_values = [(i, val) for i, val in enumerate(partial_state) if val != -1]
    known_positions = [i for i, _ in known_values]
    known_vals = [val for _, val in known_values]

    states = []

    if include_state is not None:
        include_state_flat = np.array(include_state).flatten()
        if sorted(include_state_flat.tolist()) == list(range(9)):
            puzzle = EightPuzzle(include_state_flat.reshape(3, 3))
            if puzzle.is_solvable():
                states.append(include_state_flat.reshape(3, 3))
                logger.info(f"Added include_state: {include_state_flat}")
        else:
            logger.warning(f"include_state is invalid: {include_state_flat}")

    max_attempts = 10000
    attempts = 0

    while len(states) < num_states and attempts < max_attempts:
        if not known_values:
            new_state = np.random.permutation(9).tolist()
        else:
            new_state = partial_state.copy()
            remaining_positions = [i for i in range(9) if i not in known_positions]
            used_values = set(val for val in new_state if val != -1)
            available_values = list(set(range(9)) - used_values)

            if len(available_values) != len(remaining_positions):
                logger.debug(f"Mismatch in available values: {available_values}, positions: {remaining_positions}")
                attempts += 1
                continue

            np.random.shuffle(available_values)
            for pos, val in zip(remaining_positions, available_values):
                new_state[pos] = val

        
        if isinstance(new_state, np.ndarray):
            new_state = new_state.flatten().tolist()

        
        if sorted(new_state) != list(range(9)):
            logger.debug(f"Invalid state generated: {new_state}")
            attempts += 1
            continue

        state_array = np.array(new_state).reshape(3, 3)

        try:
            puzzle = EightPuzzle(state_array)
            if not puzzle.is_solvable():
                logger.debug(f"Unsolvable state skipped: {new_state}")
                attempts += 1
                continue

            if observation is not None and puzzle.get_observation() != observation:
                logger.debug(f"Observation mismatch, skipping state: {new_state}")
                attempts += 1
                continue

            
            if not any(np.array_equal(state_array, s) for s in states):
                states.append(state_array)
                logger.info(f"Added state: {new_state}")

        except ValueError as e:
            logger.debug(f"State rejected due to error: {e}")
            attempts += 1
            continue

        attempts += 1

    if len(states) < num_states:
        logger.error(f"Only generated {len(states)} states after {max_attempts} attempts")
        raise ValueError(f"Could only generate {len(states)} valid states")

    logger.info(f"Successfully generated {len(states)} unique valid states")
    return states


def generate_random_init_states(num_states):
    logger.info(f"Generating {num_states} random initial states")
    states = []
    available_positions = [3, 4, 5, 6, 7, 8]
    remaining_values = [4, 5, 6, 7, 8]
    max_attempts = 1000
    attempts = 0
    while len(states) < num_states and attempts < max_attempts:
        new_state = [1, 2, 3, -1, -1, -1, -1, -1, -1]
        zero_pos = np.random.choice(available_positions)
        new_state[zero_pos] = 0
        remaining_positions = [i for i in available_positions if zero_pos != i]
        random_values = np.random.permutation(remaining_values).tolist()
        for i, val in enumerate(random_values):
            new_state[remaining_positions[i]] = val
        if sorted(new_state) != list(range(9)):
            attempts += 1
            logger.debug(f"Invalid initial state: {new_state}")
            continue
        new_state_array = np.array(new_state).reshape(3, 3)
        try:
            puzzle = EightPuzzle(new_state_array)
            if not puzzle.is_solvable():
                attempts += 1
                continue
            if not any(np.array_equal(new_state_array, s) for s in states):
                states.append(new_state_array)
                logger.info(f"Added random init state: {new_state}")
        except ValueError as e:
            logger.debug(f"Invalid initial state skipped: {new_state}, error: {e}")
            attempts += 1
        attempts += 1
    if len(states) < num_states:
        logger.error(f"Could not generate {num_states} solvable states after {max_attempts} attempts")
        raise ValueError(f"Could not generate {num_states} solvable states")
    logger.info(f"Successfully generated {len(states)} random initial states")
    return states

def generate_near_goal_belief(num_states=9, goal_state=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
    logger.info(f"Generating {num_states} belief states with increased difficulty near goal state: {goal_state}")
    belief_states = set()
    goal_array = np.array(goal_state).reshape(3, 3)
    actions = ['up', 'down', 'left', 'right']
    max_attempts = 1000
    attempts = 0

    while len(belief_states) < num_states and attempts < max_attempts:
        current_state = goal_array.copy()
        puzzle = EightPuzzle(current_state)

        num_moves = randint(5, 10)
        for _ in range(num_moves):
            valid_actions = []
            blank_pos = puzzle.get_blank_pos()
            for action in actions:
                if puzzle.is_valid_action(action, blank_pos):
                    valid_actions.append(action)
            if not valid_actions:
                break
            action = choice(valid_actions)
            current_state = puzzle.move(action)
            puzzle = EightPuzzle(current_state)

        state_tuple = tuple(current_state.flatten())
        if state_tuple == goal_state or state_tuple in belief_states:
            attempts += 1
            logger.debug(f"Duplicate or goal state skipped: {state_tuple}")
            continue

        try:
            puzzle = EightPuzzle(current_state)
            if not puzzle.is_solvable():
                attempts += 1
                logger.debug(f"Unsolvable state skipped: {state_tuple}")
                continue
            belief_states.add(state_tuple)
            logger.info(f"Added state after {num_moves} random moves: {state_tuple}")
        except ValueError as e:
            attempts += 1
            logger.debug(f"Invalid state skipped due to error: {e}")
            continue

        attempts += 1

    while len(belief_states) < num_states and attempts < max_attempts:
        current_state = goal_array.copy()
        num_swaps = randint(3, 5)
        positions = [(i, j) for i in range(3) for j in range(3) if current_state[i, j] != 0]
        for _ in range(num_swaps):
            if len(positions) < 2:
                break
            pos1, pos2 = np.random.choice(len(positions), 2, replace=False)
            (i1, j1), (i2, j2) = positions[pos1], positions[pos2]
            current_state[i1, j1], current_state[i2, j2] = current_state[i2, j2], current_state[i1, j1]
            positions = [(i, j) for i in range(3) for j in range(3) if current_state[i, j] != 0]

        state_tuple = tuple(current_state.flatten())
        if state_tuple == goal_state or state_tuple in belief_states:
            attempts += 1
            logger.debug(f"Duplicate or goal state skipped: {state_tuple}")
            continue

        try:
            puzzle = EightPuzzle(current_state)
            if not puzzle.is_solvable():
                attempts += 1
                logger.debug(f"Unsolvable state skipped: {state_tuple}")
                continue
            belief_states.add(state_tuple)
            logger.info(f"Added state after {num_swaps} random swaps: {state_tuple}")
        except ValueError as e:
            attempts += 1
            logger.debug(f"Invalid state skipped due to error: {e}")
            continue

        attempts += 1

    if len(belief_states) < num_states:
        logger.warning(f"Only generated {len(belief_states)} states, filling with random valid states")
        while len(belief_states) < num_states and attempts < max_attempts:
            state = np.random.permutation(9).tolist()
            state_tuple = tuple(state)
            if state_tuple == goal_state or state_tuple in belief_states:
                attempts += 1
                continue
            try:
                puzzle = EightPuzzle(np.array(state).reshape(3, 3))
                if puzzle.is_solvable():
                    belief_states.add(state_tuple)
                    logger.info(f"Filled with random state: {state_tuple}")
                else:
                    attempts += 1
                    logger.debug(f"Unsolvable random state skipped: {state_tuple}")
            except ValueError as e:
                attempts += 1
                logger.debug(f"Invalid random state skipped: {e}")
            attempts += 1

    if len(belief_states) < num_states:
        logger.error(f"Could not generate {num_states} states after {max_attempts} attempts")
        raise ValueError(f"Could not generate {num_states} states")

    logger.info(f"Generated {len(belief_states)} belief states: {belief_states}")
    return belief_states
class ParticleFilter:
    def __init__(self, num_particles, partial_state, actual_state=None, use_non_observation=False):
        logger.info(f"Initializing ParticleFilter with {num_particles} particles")
        self.num_particles = num_particles
        self.partial_state = np.array(partial_state).reshape(3, 3)
        self.permanently_invalid_particles = []
        self.invalid_action_particles = []
        if use_non_observation:
            initial_belief = generate_near_goal_belief(num_particles)
            self.particles = [np.array(state).reshape(3, 3) for state in initial_belief]
        else:
            include_state = actual_state.flatten() if actual_state is not None else None
            try:
                self.particles = [s.copy() for s in generate_possible_states(
                    partial_state, num_particles, include_state)]
                for i, particle in enumerate(self.particles):
                    flat_particle = particle.flatten().tolist()
                    if sorted(flat_particle) != list(range(9)):
                        logger.error(f"Invalid particle {i}: {flat_particle}")
                        raise ValueError(f"Invalid particle {i}: {flat_particle}")
                    try:
                        puzzle = EightPuzzle(particle)
                        if not puzzle.is_solvable():
                            logger.error(f"Unsolvable particle {i}: {flat_particle}")
                            raise ValueError(f"Unsolvable particle {i}: {flat_particle}")
                    except ValueError as e:
                        logger.error(f"Error validating particle {i}: {e}")
                        raise ValueError(f"Error validating particle {i}: {e}")
            except ValueError as e:
                logger.error(f"Error generating particles: {e}")
                raise RuntimeError(f"Failed to initialize ParticleFilter: {e}")
        logger.info("ParticleFilter initialized successfully")

    def get_particle_weights(self):
        logger.info("Calculating particle weights")
        weights = np.zeros(len(self.particles))
        for i, particle in enumerate(self.particles):
            if i in self.permanently_invalid_particles:
                weights[i] = 0.0
                logger.debug(f"Particle {i} is permanently invalid, weight set to 0")
                continue
            try:
                puzzle = EightPuzzle(particle)
                manhattan = puzzle.manhattan_distance()
                misplaced = puzzle.misplaced_tiles()
                weights[i] = 1.0 / (0.7 * manhattan + 0.3 * misplaced + 1)
            except ValueError as e:
                logger.error(f"Error calculating weight for particle {i}: {e}")
                weights[i] = 0.0
                self.permanently_invalid_particles.append(i)
        weights /= (weights.sum() + 1e-10) if weights.sum() > 0 else 1.0
        logger.info("Particle weights calculated successfully")
        return weights

    def transition(self, action):
        logger.info(f"Applying transition with action: {action}")
        new_particles = []
        self.invalid_action_particles = []
        for i, particle in enumerate(self.particles):
            if i in self.permanently_invalid_particles:
                new_particles.append(particle.copy())
                self.invalid_action_particles.append(i)
                logger.debug(f"Particle {i} is permanently invalid, keeping unchanged")
                continue
            try:
                puzzle = EightPuzzle(particle)
                blank_pos = puzzle.get_blank_pos()
                if not puzzle.is_valid_action(action, blank_pos):
                    new_particles.append(particle.copy())
                    self.invalid_action_particles.append(i)
                    self.permanently_invalid_particles.append(i)
                    logger.debug(f"Particle {i} invalid for action {action}, marked permanently invalid")
                    continue
                new_state = puzzle.move(action)
                new_particles.append(new_state)
                logger.debug(f"Particle {i} updated: {new_state.flatten()}")
            except ValueError as e:
                logger.error(f"Error in transition for particle {i}: {e}")
                new_particles.append(particle.copy())
                self.invalid_action_particles.append(i)
                self.permanently_invalid_particles.append(i)
        self.particles = new_particles
        logger.info(f"Number of particles after transition: {len(self.particles)}")

    def filter_particles(self, observation):
        logger.info(f"No filtering applied, keeping all {len(self.particles)} particles")
        return self.particles.copy()

    def update(self, action, observation):
        logger.info(f"Updating particles with action: {action}, observation: {observation}")
        self.transition(action)
        self.particles = self.filter_particles(observation)
        logger.info(f"Number of particles after update: {len(self.particles)}")

    def low_variance_resample(self, weights):
        pass

    def regenerate_particles(self):
        logger.info("Regenerating particles due to lack of solvable states")
        self.permanently_invalid_particles = []
        self.invalid_action_particles = []
        self.particles = [s.copy() for s in generate_possible_states(
            self.partial_state.flatten(), self.num_particles, include_state=self.particles[0])]
        logger.info(f"Regenerated {len(self.particles)} particles")

class POMDPAgent:
    def __init__(self, partial_state, actual_state=None, num_particles=32, use_non_observation=False):
        logger.info("Initializing POMDPAgent")
        self.use_non_observation = use_non_observation  
        self.particle_filter = ParticleFilter(num_particles, partial_state, actual_state, use_non_observation=self.use_non_observation)
        self.actions = ['up', 'down', 'left', 'right']
        self.last_action = None
        self.action_history = []
        self.step_count = 0
        self.use_and_or = False
        self.puzzle = EightPuzzle(actual_state if actual_state is not None else 
                                 generate_possible_states(partial_state, 1)[0])
        if not self.puzzle.is_solvable():
            logger.warning(f"Initial state is not solvable: {self.puzzle.state.flatten()}")
            if not self.switch_to_next_solvable_state():
                logger.error("No solvable state found in belief set")
                self.particle_filter.regenerate_particles()
                if not self.switch_to_next_solvable_state():
                    raise ValueError("No solvable state found after regeneration")
        logger.info(f"POMDPAgent initialized with state: {self.puzzle.state.flatten()}")
    def switch_to_next_solvable_state(self):
        logger.info("Searching for next solvable state in belief set")
        current_state = self.puzzle.state.copy()
        for i, particle in enumerate(self.particle_filter.particles):
            if np.array_equal(particle, current_state) or i in self.particle_filter.permanently_invalid_particles:
                continue
            try:
                puzzle = EightPuzzle(particle)
                if puzzle.is_solvable():
                    self.puzzle.state = particle.copy()
                    logger.info(f"Switched to solvable state: {self.puzzle.state.flatten()}")
                    return True
            except ValueError:
                continue
        logger.warning("No solvable state found in belief set")
        return False

    def select_action(self):
        logger.info(f"Selecting action for state: {self.puzzle.state.flatten()}")
        
        particles = self.particle_filter.particles
        if not particles:
            logger.warning("No particles available to evaluate actions")
            return None

        if self.use_and_or:
            belief_states = set()
            for p in particles:
                if any(np.array_equal(p, self.particle_filter.particles[i]) for i in self.particle_filter.permanently_invalid_particles):
                    continue
                flat_p = p.flatten().tolist()
                if sorted(flat_p) != list(range(9)):
                    logger.error(f"Invalid particle in belief_states: {flat_p}")
                    continue
                try:
                    puzzle = EightPuzzle(p)
                    if puzzle.is_solvable():
                        belief_states.add(tuple(flat_p))
                except ValueError as e:
                    logger.error(f"Error adding particle to belief_states: {e}")
                    continue
            if not belief_states:
                logger.warning("No valid solvable belief states for AND-OR search")
                return None
            logger.info(f"Running AND-OR search with {len(belief_states)} belief states")
            plan = and_or_search(belief_states, max_depth=30)
            if plan and len(plan) > 0:
                best_action = plan[0]
                logger.info(f"AND-OR search selected action: {best_action}")
                self.last_action = best_action
                return best_action
            else:
                logger.warning("AND-OR search failed to find a plan, falling back to random valid action")
                valid_actions = set()
                for particle in particles:
                    try:
                        puzzle = EightPuzzle(particle)
                        blank_pos = puzzle.get_blank_pos()
                        for action in self.actions:
                            if puzzle.is_valid_action(action, blank_pos):
                                valid_actions.add(action)
                    except ValueError:
                        continue
                if valid_actions:
                    best_action = choice(list(valid_actions))
                    logger.info(f"Fallback random action: {best_action}")
                    self.last_action = best_action
                    return best_action
                logger.warning("No valid actions available")
                return None

        if self.use_non_observation:
            belief_states = set()
            for p in particles:
                if any(np.array_equal(p, self.particle_filter.particles[i]) for i in self.particle_filter.permanently_invalid_particles):
                    continue
                flat_p = p.flatten().tolist()
                if sorted(flat_p) != list(range(9)):
                    logger.error(f"Invalid particle in belief_states: {flat_p}")
                    continue
                try:
                    puzzle = EightPuzzle(p)
                    if puzzle.is_solvable():
                        belief_states.add(tuple(flat_p))
                except ValueError as e:
                    logger.error(f"Error adding particle to belief_states: {e}")
                    continue
            if not belief_states:
                logger.warning("No valid solvable belief states for Non-Observation search")
                return None
            logger.info(f"Running Non-Observation search with {len(belief_states)} belief states")
            plan = non_observation_search(belief_states, max_depth=30)
            if plan and len(plan) > 0:
                best_action = plan[0]
                logger.info(f"Non-Observation search selected action: {best_action}")
                self.last_action = best_action
                return best_action
            else:
                logger.warning("Non-Observation search failed to find a plan, falling back to random valid action")
                valid_actions = set()
                for particle in particles:
                    try:
                        puzzle = EightPuzzle(particle)
                        blank_pos = puzzle.get_blank_pos()
                        for action in self.actions:
                            if puzzle.is_valid_action(action, blank_pos):
                                valid_actions.add(action)
                    except ValueError:
                        continue
                if valid_actions:
                    best_action = choice(list(valid_actions))
                    logger.info(f"Fallback random action: {best_action}")
                    self.last_action = best_action
                    return best_action
                logger.warning("No valid actions available")
                return None

        weights = self.particle_filter.get_particle_weights()
        best_action = None
        best_heuristic_score = float('inf')
        valid_actions = set()

        for particle in particles:
            try:
                puzzle = EightPuzzle(particle)
                blank_pos = puzzle.get_blank_pos()
                for action in self.actions:
                    if puzzle.is_valid_action(action, blank_pos):
                        valid_actions.add(action)
            except ValueError:
                continue

        if not valid_actions:
            logger.warning("No valid actions available across all belief states")
            return None

        for action in valid_actions:
            total_heuristic = 0.0
            total_weight = 0.0
            for i, particle in enumerate(particles):
                if i in self.particle_filter.permanently_invalid_particles or weights[i] == 0:
                    continue
                try:
                    puzzle = EightPuzzle(particle)
                    blank_pos = puzzle.get_blank_pos()
                    if not puzzle.is_valid_action(action, blank_pos):
                        continue
                    new_state = puzzle.move(action)
                    temp_puzzle = EightPuzzle(new_state)
                    heuristic = 0.7 * temp_puzzle.manhattan_distance() + 0.3 * temp_puzzle.misplaced_tiles()
                    total_heuristic += heuristic * weights[i]
                    total_weight += weights[i]
                except ValueError:
                    continue

            if total_weight > 0:
                avg_heuristic = total_heuristic / total_weight
                logger.debug(f"Action {action}: Average heuristic = {avg_heuristic}")
                if avg_heuristic < best_heuristic_score:
                    best_heuristic_score = avg_heuristic
                    best_action = action

        if best_action is None:
            best_action = choice(list(valid_actions))
            logger.warning(f"No better action found, choosing randomly: {best_action}")

        self.last_action = best_action
        logger.info(f"Selected action: {best_action} with average heuristic score: {best_heuristic_score}")
        return best_action

    def step(self):
        logger.info(f"Starting step {self.step_count + 1}")
        action = self.select_action()
        if action is None:
            logger.warning("Step terminated: No valid action")
            return None, None
        old_state = self.puzzle.state.copy()
        new_state = self.puzzle.move(action)
        self.puzzle.state = new_state
        observation = self.puzzle.get_observation()
        self.particle_filter.update(action, observation)
        self.action_history.append(action)
        self.step_count += 1
        logger.info(f"Step {self.step_count}: Action = {action}, Old state = {old_state.flatten()}, "
                    f"New state = {self.puzzle.state.flatten()}, Particles = {len(self.particle_filter.particles)}")
        return action, observation

    def is_belief_goal(self):
        if not self.particle_filter.particles:
            return False
        return any(EightPuzzle(p).is_goal() for p in self.particle_filter.particles)

def draw_belief_states(particles, actual_state, title_text, invalid_action_particles=None, action=None, use_and_or=False, use_non_observation=False):
    logger.info("Drawing belief states")
    screen.fill(WHITE)
    
    title = title_font.render(title_text, True, BLACK)
    title_rect = title.get_rect(center=(WIDTH // 2, 25))
    screen.blit(title, title_rect)

    button_width = 100
    button_height = 40
    button_y = 60
    astar_button_x = WIDTH // 2 - button_width * 3 // 2 - 50
    andor_button_x = WIDTH // 2 - button_width // 2 - 10
    nonobs_button_x = WIDTH // 2 + button_width // 2 + 10

    astar_color = DARK_GREEN if not use_and_or and not use_non_observation else LIGHT_GRAY
    pygame.draw.rect(screen, astar_color, (astar_button_x, button_y, button_width + 20, button_height))
    pygame.draw.rect(screen, BLACK, (astar_button_x, button_y, button_width +20, button_height), 2)
    astar_text = font.render("Partial-Obs", True, BLACK)
    astar_text_rect = astar_text.get_rect(center=(astar_button_x + button_width // 2 + 10, button_y + button_height // 2))
    screen.blit(astar_text, astar_text_rect)
    
    andor_color = DARK_GREEN if use_and_or else LIGHT_GRAY
    pygame.draw.rect(screen, andor_color, (andor_button_x, button_y, button_width, button_height))
    pygame.draw.rect(screen, BLACK, (andor_button_x, button_y, button_width, button_height), 2)
    andor_text = font.render("AND-OR", True, BLACK)
    andor_text_rect = andor_text.get_rect(center=(andor_button_x + button_width // 2, button_y + button_height // 2))
    screen.blit(andor_text, andor_text_rect)

    nonobs_color = DARK_GREEN if use_non_observation else LIGHT_GRAY
    pygame.draw.rect(screen, nonobs_color, (nonobs_button_x, button_y, button_width, button_height))
    pygame.draw.rect(screen, BLACK, (nonobs_button_x, button_y, button_width, button_height), 2)
    nonobs_text = font.render("Non-Obs", True, BLACK)
    nonobs_text_rect = nonobs_text.get_rect(center=(nonobs_button_x + button_width // 2, button_y + button_height // 2))
    screen.blit(nonobs_text, nonobs_text_rect)

    for idx, particle in enumerate(particles):
        try:
            puzzle = EightPuzzle(particle)
            row = (idx // PUZZLES_PER_ROW)
            col = idx % PUZZLES_PER_ROW
            offset_x = col * (PUZZLE_WIDTH + PADDING) + PADDING
            offset_y = row * (PUZZLE_HEIGHT + PADDING) + PADDING + 100

            for i in range(3):
                for j in range(3):
                    x = offset_x + j * TILE_SIZE
                    y = offset_y + i * TILE_SIZE
                    value = puzzle.state[i, j]
                    color = YELLOW if puzzle.is_goal() else (GRAY if invalid_action_particles and idx in invalid_action_particles else GREEN)
                    pygame.draw.rect(screen, color, (x, y, TILE_SIZE, TILE_SIZE))
                    pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 2)
                    if value != 0:
                        text = font.render(str(value), True, BLACK)
                        text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                        screen.blit(text, text_rect)
        except ValueError as e:
            logger.error(f"Error drawing particle {idx}: {e}")
            continue

    pygame.display.flip()
    logger.info("Finished drawing belief states")
    return (astar_button_x, button_y, button_width, button_height), (andor_button_x, button_y, button_width, button_height), (nonobs_button_x, button_y, button_width, button_height)

def main():
    partial_state = [1, 2, 3, -1, -1, -1, -1, -1, -1]
    use_and_or = False
    use_non_observation = False
    num_particles = 32
    try:
        init_states = generate_random_init_states(1)
        actual_state = init_states[0]
    except Exception as e:
        logger.error(f"Error generating initial state: {e}")
        pygame.quit()
        return

    try:
        agent = POMDPAgent(partial_state, actual_state=actual_state, num_particles=num_particles, use_non_observation=use_non_observation)
        agent.use_and_or = use_and_or
    except Exception as e:
        logger.error(f"Error initializing POMDPAgent: {e}")
        pygame.quit()
        return

    clock = pygame.time.Clock()
    running = True
    max_steps = 200

    astar_button, andor_button, nonobs_button = draw_belief_states(
        agent.particle_filter.particles, agent.puzzle.state,
        "Initial Belief States (SPACE to step, Click buttons to switch algorithm, ESC to quit)",
        invalid_action_particles=agent.particle_filter.invalid_action_particles,
        use_and_or=use_and_or,
        use_non_observation=use_non_observation
    )

    while running:
        if agent.step_count >= max_steps or agent.puzzle.is_goal() or agent.is_belief_goal():
            if agent.puzzle.is_goal():
                astar_button, andor_button, nonobs_button = draw_belief_states(
                    agent.particle_filter.particles, agent.puzzle.state,
                    "Actual State Reached Goal (Press SPACE to quit)",
                    invalid_action_particles=agent.particle_filter.invalid_action_particles,
                    use_and_or=use_and_or,
                    use_non_observation=use_non_observation
                )
            elif agent.is_belief_goal():
                astar_button, andor_button, nonobs_button = draw_belief_states(
                    agent.particle_filter.particles, agent.puzzle.state,
                    "Goal State Reached for Belief Set (Press SPACE to quit)",
                    invalid_action_particles=agent.particle_filter.invalid_action_particles,
                    use_and_or=use_and_or,
                    use_non_observation=use_non_observation
                )
            elif agent.step_count >= max_steps:
                astar_button, andor_button, nonobs_button = draw_belief_states(
                    agent.particle_filter.particles, agent.puzzle.state,
                    "Maximum Steps Reached (Press SPACE to quit)",
                    invalid_action_particles=agent.particle_filter.invalid_action_particles,
                    use_and_or=use_and_or,
                    use_non_observation=use_non_observation
                )
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action, observation = agent.step()
                if action is None and observation is None:
                    logger.warning("No valid action available, attempting to regenerate particles")
                    agent.particle_filter.regenerate_particles()
                    if not agent.switch_to_next_solvable_state():
                        logger.error("No solvable state found after regeneration")
                        astar_button, andor_button, nonobs_button = draw_belief_states(
                            agent.particle_filter.particles, agent.puzzle.state,
                            "No Solvable State Available (Press SPACE to quit)",
                            invalid_action_particles=agent.particle_filter.invalid_action_particles,
                            use_and_or=use_and_or,
                            use_non_observation=use_non_observation
                        )
                        running = False
                        break
                    astar_button, andor_button, nonobs_button = draw_belief_states(
                        agent.particle_filter.particles, agent.puzzle.state,
                        f"Regenerated Belief States (SPACE to step, Click buttons to switch algorithm, ESC to quit)",
                        invalid_action_particles=agent.particle_filter.invalid_action_particles,
                        use_and_or=use_and_or,
                        use_non_observation=use_non_observation
                    )
                else:
                    algo = "Non-Obs" if use_non_observation else ("AND-OR" if use_and_or else "Partial Obs")
                    astar_button, andor_button, nonobs_button = draw_belief_states(
                        agent.particle_filter.particles, agent.puzzle.state,
                        f"Step {agent.step_count} after '{action}' (Algorithm: {algo}, SPACE to step, Click buttons to switch, ESC to quit)",
                        invalid_action_particles=agent.particle_filter.invalid_action_particles,
                        action=action,
                        use_and_or=use_and_or,
                        use_non_observation=use_non_observation
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                astar_x, astar_y, astar_w, astar_h = astar_button
                andor_x, andor_y, andor_w, andor_h = andor_button
                nonobs_x, nonobs_y, nonobs_w, nonobs_h = nonobs_button
                if astar_x <= mouse_pos[0] <= astar_x + astar_w and astar_y <= mouse_pos[1] <= astar_y + astar_h:
                    use_and_or = False
                    use_non_observation = False
                    num_particles = 32
                    agent = POMDPAgent(partial_state, actual_state=actual_state, num_particles=num_particles, use_non_observation=use_non_observation)
                    agent.use_and_or = use_and_or
                    logger.info("Switched to algorithm: Partial Obs")
                elif andor_x <= mouse_pos[0] <= andor_x + andor_w and andor_y <= mouse_pos[1] <= andor_y + andor_h:
                    use_and_or = True
                    use_non_observation = False
                    num_particles = 32
                    agent = POMDPAgent(partial_state, actual_state=actual_state, num_particles=num_particles, use_non_observation=use_non_observation)
                    agent.use_and_or = use_and_or 
                    logger.info("Switched to algorithm: AND-OR")
                elif nonobs_x <= mouse_pos[0] <= nonobs_x + nonobs_w and nonobs_y <= mouse_pos[1] <= nonobs_y + nonobs_h:
                    use_and_or = False
                    use_non_observation = True
                    num_particles = 32
                    agent = POMDPAgent(partial_state, actual_state=actual_state, num_particles=num_particles, use_non_observation=use_non_observation)
                    agent.use_and_or = use_and_or
                    logger.info("Switched to algorithm: Non-Observation")
                astar_button, andor_button, nonobs_button = draw_belief_states(
                    agent.particle_filter.particles, agent.puzzle.state,
                    f"Switched to {'Non-Obs' if use_non_observation else ('AND-OR' if use_and_or else 'A*')} (SPACE to step, Click buttons to switch, ESC to quit)",
                    invalid_action_particles=agent.particle_filter.invalid_action_particles,
                    use_and_or=use_and_or,
                    use_non_observation=use_non_observation
                )

        clock.tick(FPS)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                running = False
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        pygame.quit()