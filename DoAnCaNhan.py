import pygame
import sys
from collections import deque
import copy
import heapq
import time
import random
import math
import numpy as np

pygame.init()

# Window and Grid Settings
WIDTH, HEIGHT = 1200, 800
GRID_SIZE = 3
CELL_SIZE = 100
INPUT_CELL_SIZE = 50
FPS = 60
GRID_OFFSET_X = 300
GRID_OFFSET_Y = 100
INITIAL_OFFSET_X, INITIAL_OFFSET_Y = 50, 450
GOAL_OFFSET_X, GOAL_OFFSET_Y = 600, 450
TEXTBOX_X, TEXTBOX_Y = 900, 100
TEXTBOX_WIDTH, TEXTBOX_HEIGHT = 250, 500
TEXTBOX_LINE_HEIGHT = 150
SCROLLBAR_WIDTH = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GRAY = (200, 200, 200)
GREEN = (0, 200, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (150, 150, 150)

# ListBox Class
class ListBox:
    def __init__(self, x, y, width, height, items, font, visible_items=5):
        self.rect = pygame.Rect(x, y, width, height)
        self.items = items
        self.font = font
        self.visible_items = visible_items
        self.scroll_offset = 0
        self.selected_index = 0
        self.item_height = height // visible_items
        self.line_spacing = 40
        self.scrollbar_width = 10
        self.max_scroll = max(0, len(items) - visible_items)

    def draw(self, screen, active_algorithm, is_solvable_state):
        pygame.draw.rect(screen, LIGHT_GRAY, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        for i in range(self.scroll_offset, min(self.scroll_offset + self.visible_items, len(self.items))):
            item_y = self.rect.y + (i - self.scroll_offset) * self.item_height
            item_rect = pygame.Rect(self.rect.x, item_y, self.rect.width - self.scrollbar_width, self.item_height)
            if self is path_listbox:
                color = LIGHT_GRAY
                if algorithm == "IDS":
                    self.item_height = self.line_spacing * 5
                else:
                    self.item_height = self.line_spacing * 4
            else:
                color = RED if not is_solvable_state else (GREEN if solving and algorithm == self.items[i].upper() else BLUE if algorithm == self.items[i].upper() else GRAY)
                if i == self.selected_index:
                    pygame.draw.rect(screen, color, item_rect)
            lines = self.items[i].split('\n')
            total_text_height = len(lines) * self.line_spacing
            start_y = item_y + (self.item_height - total_text_height) // 2
            for line_idx, line in enumerate(lines):
                text = self.font.render(line, True, WHITE if i == self.selected_index else BLACK)
                line_y = item_y + line_idx * self.line_spacing
                if line_idx == 0:
                    text_rect = text.get_rect(midleft=(self.rect.x + 10, line_y + self.line_spacing // 2))
                else:
                    text_rect = text.get_rect(center=(item_rect.centerx, line_y + self.line_spacing // 2))
                screen.blit(text, text_rect)
        if self.max_scroll > 0:
            scrollbar_height = self.rect.height * (self.visible_items / len(self.items))
            scrollbar_y = self.rect.y + (self.scroll_offset / self.max_scroll) * (self.rect.height - scrollbar_height)
            pygame.draw.rect(screen, DARK_GRAY,
                            (self.rect.x + self.rect.width - self.scrollbar_width,
                             scrollbar_y, self.scrollbar_width, scrollbar_height))

    def clear(self):
        self.items.clear()
        self.scroll_offset = 0
        self.selected_index = 0
        self.max_scroll = 0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                relative_y = event.pos[1] - self.rect.y
                clicked_index = self.scroll_offset + relative_y // self.item_height
                if clicked_index < len(self.items):
                    self.selected_index = clicked_index
                    return self.items[self.selected_index]
        elif event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll_offset = max(0, min(self.scroll_offset - event.y, self.max_scroll))
        return None

    def get_selected(self):
        return self.items[self.selected_index]

# Pygame Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver- Nguyễn Hoàng Anh Kiệt -23110247")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 40)
button_font = pygame.font.Font(None, 32)
input_font = pygame.font.Font(None, 28)
info_font = pygame.font.Font(None, 30)
textbox_font = pygame.font.Font(None, 35)
title_font = pygame.font.Font(None, 50)

base_delay = 500
speed_multiplier = 1
speed_options = ["1x", "2x", "4x", "10x"]
speed_listbox = ListBox(450, 650, 100, 110, speed_options, button_font, visible_items=3)
path_listbox = ListBox(TEXTBOX_X, TEXTBOX_Y, TEXTBOX_WIDTH + 25, TEXTBOX_HEIGHT+150, [], textbox_font, visible_items=4)
algorithms = ["BFS", "DFS", "UCS", "IDS", "GREEDY", "A*", "IDA*", "SIMPLE_HILL_CLIMBING", "STEEPEST_HILL", "STOCHASTIC_HILL", "SIMULATED_ANNEALING", "BEAM_SEARCH", "GENETIC","AND-OR", "BACKTRACKING_CSP","GENERATE & TEST","AC3","Q_LEARNING"]
algorithm_listbox = ListBox(50, 650, 350, 110, algorithms, button_font, visible_items=4)
solve_button = pygame.Rect(650, 650, 100, 40)
reload_button = pygame.Rect(650, 700, 100, 40)

initial_input_boxes = [[pygame.Rect(INITIAL_OFFSET_X + j * (INPUT_CELL_SIZE + 10),
                                    INITIAL_OFFSET_Y + i * (INPUT_CELL_SIZE + 10),
                                    INPUT_CELL_SIZE, INPUT_CELL_SIZE)
                       for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
goal_input_boxes = [[pygame.Rect(GOAL_OFFSET_X + j * (INPUT_CELL_SIZE + 10),
                                 GOAL_OFFSET_Y + i * (INPUT_CELL_SIZE + 10),
                                 INPUT_CELL_SIZE, INPUT_CELL_SIZE)
                    for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]

solving = False
solution = []
solution_states = []
algorithm = "BFS"
default_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
default_goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
matrix = copy.deepcopy(default_matrix)
goal_state = copy.deepcopy(default_goal)
initial_input = copy.deepcopy(default_matrix)
goal_input = copy.deepcopy(default_goal)
active_input = None
steps = 0
cost = 0
mouse_over_button = None
solve_time = 0.0
last_belief_size = 0

def manhattan_distance_1d(state_1d, goal_1d=None):
    if goal_1d is None:
        goal_1d = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    distance = 0
    for i, num in enumerate(state_1d):
        if num != 0:
            target_pos = goal_1d.index(num)
            current_x, current_y = divmod(i, 3)
            target_x, target_y = divmod(target_pos, 3)
            distance += abs(current_x - target_x) + abs(current_y - target_y)
    return distance

def draw_grid():
    global steps, cost, mouse_over_button
    screen.fill(WHITE)
    title_text = title_font.render("8-Puzzle Solver", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(GRID_OFFSET_X + col * CELL_SIZE,
                              GRID_OFFSET_Y + row * CELL_SIZE,
                              CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLUE if matrix[row][col] != 0 else LIGHT_GRAY, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
            if matrix[row][col] != 0:
                text = font.render(str(matrix[row][col]), True, WHITE)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pygame.draw.rect(screen, GRAY, initial_input_boxes[i][j])
            pygame.draw.rect(screen, BLACK, initial_input_boxes[i][j], 2)
            if initial_input[i][j] != 0:
                text = input_font.render(str(initial_input[i][j]), True, BLACK)
                text_rect = text.get_rect(center=initial_input_boxes[i][j].center)
                screen.blit(text, text_rect)
            pygame.draw.rect(screen, GRAY, goal_input_boxes[i][j])
            pygame.draw.rect(screen, BLACK, goal_input_boxes[i][j], 2)
            if goal_input[i][j] != 0:
                text = input_font.render(str(goal_input[i][j]), True, BLACK)
                text_rect = text.get_rect(center=goal_input_boxes[i][j].center)
                screen.blit(text, text_rect)
            if active_input:
                input_type, r, c = active_input
                if input_type == "initial" and i == r and j == c:
                    pygame.draw.rect(screen, RED, initial_input_boxes[i][j], 3)
                elif input_type == "goal" and i == r and j == c:
                    pygame.draw.rect(screen, RED, goal_input_boxes[i][j], 3)
    screen.blit(input_font.render("Initial State", True, BLACK), (INITIAL_OFFSET_X, INITIAL_OFFSET_Y - 40))
    screen.blit(input_font.render("Goal State", True, BLACK), (GOAL_OFFSET_X, GOAL_OFFSET_Y - 40))
    algorithm_listbox.draw(screen, algorithm, is_solvable(matrix))
    speed_listbox.draw(screen, f"{speed_multiplier}x", True)
    solve_color = GREEN if is_solvable(matrix) else RED
    if solve_button == mouse_over_button:
        solve_color = tuple(min(c + 50, 255) for c in solve_color)
    pygame.draw.rect(screen, solve_color, solve_button)
    pygame.draw.rect(screen, BLACK, solve_button, 1)
    solve_text = button_font.render("Solve", True, BLACK)
    solve_text_rect = solve_text.get_rect(center=solve_button.center)
    screen.blit(solve_text, solve_text_rect)
    reload_color = YELLOW
    if reload_button == mouse_over_button:
        reload_color = tuple(min(c + 50, 255) for c in reload_color)
    pygame.draw.rect(screen, reload_color, reload_button)
    pygame.draw.rect(screen, BLACK, reload_button, 1)
    reload_text = button_font.render("Reload", True, BLACK)
    reload_text_rect = reload_text.get_rect(center=reload_button.center)
    screen.blit(reload_text, reload_text_rect)
    screen.blit(info_font.render("Solution Path", True, BLACK), (TEXTBOX_X + 10, TEXTBOX_Y - 30))
    path_listbox.draw(screen, None, True)
    info_text = info_font.render(f"Steps: {steps} | Cost: {cost}", True, BLACK)
    screen.blit(info_text, (WIDTH // 3 - info_text.get_width() // 2, 500))
    time_text = info_font.render(f"Time: {solve_time:.6f}", True, BLACK)
    screen.blit(time_text, (WIDTH // 3 - time_text.get_width() // 2, 530))
    if algorithm == "PARTIAL_OBSERVATION":
        belief_size = last_belief_size
        belief_text = info_font.render(f"Belief States: {belief_size}", True, BLACK)
        screen.blit(belief_text, (WIDTH // 3 - belief_text.get_width() // 2, 560))
    pygame.display.flip()

def set_speed(multiplier_str):
    global speed_multiplier, base_delay
    speed_multiplier = int(multiplier_str.replace("x", ""))
    base_delay = 500 // speed_multiplier

def is_valid_state_1d(state_1d):
    return len(state_1d) == len(set(state_1d)) and all(0 <= num <= 8 for num in state_1d)

def is_solvable(state_2d):
    flat = [num for row in state_2d for num in row if num != 0]
    inversions = sum(1 for i in range(len(flat)) for j in range(i + 1, len(flat)) if flat[i] > flat[j])
    return inversions % 2 == 0


def to_1d(state_2d):
    return [state_2d[i][j] for i in range(3) for j in range(3)]

def to_2d(state_1d):
    return [[state_1d[i * 3 + j] for j in range(3)] for i in range(3)]

def get_zero_pos_1d(state_1d):
    return state_1d.index(0)

def get_move_direction(x1, y1, x2, y2):
    if x1 == x2:
        if y2 == y1 + 1:
            return "Move right"
        elif y2 == y1 - 1:
            return "Move left"
    elif y1 == y2:
        if x2 == x1 + 1:
            return "Move down"
        elif x2 == x1 - 1:
            return "Move up"
    return "Unknown move"

moves_1d = [(0, 1), (1, 0), (0, -1), (-1, 0)]



def get_neighbors_1d(state_1d):
    zero_idx = get_zero_pos_1d(state_1d)
    row, col = divmod(zero_idx, 3)
    neighbors = []
    for dx, dy in moves_1d:
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state_1d = state_1d.copy()
            new_state_1d[zero_idx], new_state_1d[new_idx] = new_state_1d[new_idx], new_state_1d[zero_idx]
            neighbors.append((new_state_1d, (row, col, new_row, new_col)))
    return neighbors

def bfs_solve(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    queue = deque([(start_1d, [])])
    visited = set()
    while queue:
        current_1d, path = queue.popleft()
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for new_state_1d, move in get_neighbors_1d(current_1d):
            new_state_tuple = tuple(new_state_1d)
            if new_state_tuple not in visited:
                queue.append((new_state_1d, path + [move]))
    return None

def dfs_solve(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    stack = [(start_1d, [])]
    visited = set()
    while stack:
        current_1d, path = stack.pop()
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for new_state_1d, move in reversed(get_neighbors_1d(current_1d)):
            new_state_tuple = tuple(new_state_1d)
            if new_state_tuple not in visited:
                stack.append((new_state_1d, path + [move]))
    return None


def ucs_solve(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    pq = [(0, start_1d, [])]
    visited = set()
    while pq:
        cost, current_1d, path = heapq.heappop(pq)
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for new_state_1d, move in get_neighbors_1d(current_1d):
            new_state_tuple = tuple(new_state_1d)
            if new_state_tuple not in visited:
                new_cost = cost + 1
                heapq.heappush(pq, (new_cost, new_state_1d, path + [move]))
    return None

def ids_solve(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    depth = 0
    max_depth = 100
    while depth < max_depth:
        stack = [(start_1d, [], 0)]
        visited = set()
        while stack:
            current_1d, path, current_depth = stack.pop()
            if current_1d == goal_1d:
                return path, depth
            if current_depth >= depth:
                continue
            state_tuple = tuple(current_1d)
            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            for new_state_1d, move in reversed(get_neighbors_1d(current_1d)):
                new_state_tuple = tuple(new_state_1d)
                if new_state_tuple not in visited:
                    stack.append((new_state_1d, path + [move], current_depth + 1))
        depth += 1
    return None, None

def greedy_solve(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    pq = [(manhattan_distance_1d(start_1d, goal_1d), start_1d, [])]
    visited = set()
    while pq:
        _, current_1d, path = heapq.heappop(pq)
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for new_state_1d, move in get_neighbors_1d(current_1d):
            new_state_tuple = tuple(new_state_1d)
            if new_state_tuple not in visited:
                new_cost = manhattan_distance_1d(new_state_1d, goal_1d)
                heapq.heappush(pq, (new_cost, new_state_1d, path + [move]))
    return None


def a_star_search(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    pq = [(0, 0, start_1d, [])]
    visited = set()
    while pq:
        f_cost, g_cost, current_1d, path = heapq.heappop(pq)
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for new_state_1d, move in get_neighbors_1d(current_1d):
            new_state_tuple = tuple(new_state_1d)
            if new_state_tuple not in visited:
                g_new = g_cost + 1
                h_new = manhattan_distance_1d(new_state_1d, goal_1d)
                f_new = g_new + h_new
                heapq.heappush(pq, (f_new, g_new, new_state_1d, path + [move]))
    return None

def ida_star(start_2d, goal_2d):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    def search(node_1d, g, bound, path_1d):
        f = g + manhattan_distance_1d(node_1d, goal_1d)
        if f > bound:
            return f, None
        if node_1d == goal_1d:
            return None, path_1d
        min_threshold = float('inf')
        for new_state_1d, move in get_neighbors_1d(node_1d):
            new_state_tuple = tuple(new_state_1d)
            if new_state_tuple not in [tuple(p) for p in path_1d]:
                new_path = path_1d + [new_state_1d]
                temp_bound, result = search(new_state_1d, g + 1, bound, new_path)
                if result is not None:
                    return None, result
                min_threshold = min(min_threshold, temp_bound)
        return min_threshold, None
    bound = manhattan_distance_1d(start_1d, goal_1d)
    path = [start_1d]
    while True:
        temp_bound, result = search(start_1d, 0, bound, path)
        if result is not None:
            moves_list = []
            for i in range(len(result) - 1):
                curr_state_1d = result[i]
                next_state_1d = result[i + 1]
                zero_idx = get_zero_pos_1d(curr_state_1d)
                next_zero_idx = get_zero_pos_1d(next_state_1d)
                curr_row, curr_col = divmod(zero_idx, 3)
                next_row, next_col = divmod(next_zero_idx, 3)
                moves_list.append((curr_row, curr_col, next_row, next_col))
            return moves_list
        if temp_bound == float('inf'):
            return None
        bound = temp_bound

def simple_hill_climbing(start_2d, goal_2d):
    current_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    path = []
    visited = set()
    while True:
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        current_heuristic = manhattan_distance_1d(current_1d, goal_1d)
        best_neighbor_1d = None
        best_heuristic = float('inf')
        best_move = None
        for new_state_1d, move in get_neighbors_1d(current_1d):
            new_heuristic = manhattan_distance_1d(new_state_1d, goal_1d)
            if new_heuristic < best_heuristic:
                best_heuristic = new_heuristic
                best_neighbor_1d = new_state_1d
                best_move = move
        if best_neighbor_1d is None or best_heuristic >= current_heuristic:
            return None
        current_1d = best_neighbor_1d
        path.append(best_move)
    return None

def steepest_hill_climbing(start_2d, goal_2d):
    current_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    path = []
    visited = set()
    while True:
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        current_heuristic = manhattan_distance_1d(current_1d, goal_1d)
        best_neighbor_1d = None
        best_heuristic = current_heuristic
        best_move = None
        for new_state_1d, move in get_neighbors_1d(current_1d):
            new_heuristic = manhattan_distance_1d(new_state_1d, goal_1d)
            if new_heuristic < best_heuristic:
                best_heuristic = new_heuristic
                best_neighbor_1d = new_state_1d
                best_move = move
        if best_neighbor_1d is None:
            return None
        current_1d = best_neighbor_1d
        path.append(best_move)
    return None

def stochastic_hill_climbing(start_2d, goal_2d):
    current_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    path = []
    visited = set()
    while True:
        if current_1d == goal_1d:
            return path
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)
        current_heuristic = manhattan_distance_1d(current_1d, goal_1d)
        neighbors = get_neighbors_1d(current_1d)
        if not neighbors:
            return None
        better_neighbors = [(state, h, move) for state, move in neighbors if (h := manhattan_distance_1d(state, goal_1d)) < current_heuristic]
        if better_neighbors:
            next_state_1d, _, move = random.choice(better_neighbors)
        else:
            return None
        current_1d = next_state_1d
        path.append(move)
    return None

def simulated_annealing(start_2d, goal_2d, max_iterations=100000, initial_temp=100000, max_restarts=100):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    if not is_valid_state_1d(start_1d) or not is_valid_state_1d(goal_1d):
        return []
    best_path = None
    best_distance = float('inf')
    for _ in range(max_restarts):
        current_1d = start_1d.copy()
        path = []
        temperature = initial_temp
        cooling_rate = 0.99
        min_temperature = 0.01
        iteration = 0
        visited_states = {tuple(current_1d)}
        while temperature > min_temperature and iteration < max_iterations:
            if current_1d == goal_1d:
                return path
            neighbors = get_neighbors_1d(current_1d)
            valid_neighbors = [(n, m) for n, m in neighbors if tuple(n) not in visited_states]
            if not valid_neighbors:
                break
            valid_neighbors.sort(key=lambda x: manhattan_distance_1d(x[0], goal_1d))
            next_state_1d, move = random.choice(valid_neighbors[:3]) if len(valid_neighbors) >= 3 else random.choice(valid_neighbors)
            current_value = manhattan_distance_1d(current_1d, goal_1d)
            next_value = manhattan_distance_1d(next_state_1d, goal_1d)
            delta_e = current_value - next_value
            if delta_e > 0 or random.uniform(0, 1) < math.exp(delta_e / temperature):
                current_1d = next_state_1d.copy()
                path.append(move)
                visited_states.add(tuple(current_1d))
            temperature *= cooling_rate
            iteration += 1
        final_distance = manhattan_distance_1d(current_1d, goal_1d)
        if final_distance < best_distance:
            best_distance = final_distance
            best_path = path.copy()
        if current_1d == goal_1d:
            return path
    return best_path if best_path else None

def beam_search(start_2d, goal_2d, beam_width=5):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    current_beam = [(start_1d, [])]
    visited = set()
    while current_beam:
        next_beam = []
        for state_1d, path in current_beam:
            if state_1d == goal_1d:
                return path
            state_tuple = tuple(state_1d)
            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            neighbors = get_neighbors_1d(state_1d)
            for neighbor_1d, move in neighbors:
                next_beam.append((neighbor_1d, path + [move]))
        next_beam.sort(key=lambda x: manhattan_distance_1d(x[0], goal_1d))
        current_beam = next_beam[:beam_width]
    return None

def genetic_algorithm_solve(start_2d, goal_2d, population_size=50, generations=200, mutation_rate=0.1):
    goal_1d = to_1d(goal_2d)
    def fitness(state_1d):
        return -manhattan_distance_1d(state_1d, goal_1d)
    def random_state_1d():
        nums = list(range(9))
        while True:
            random.shuffle(nums)
            state_2d = to_2d(nums)
            if is_solvable(state_2d):
                return nums
    def crossover(parent1_1d, parent2_1d):
        point = random.randint(1, 7)
        new_flat = parent1_1d[:point] + [x for x in parent2_1d if x not in parent1_1d[:point]]
        return new_flat
    def mutate(state_1d):
        zero_idx = get_zero_pos_1d(state_1d)
        row, col = divmod(zero_idx, 3)
        dx, dy = random.choice(moves_1d)
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state_1d = state_1d.copy()
            new_state_1d[zero_idx], new_state_1d[new_idx] = new_state_1d[new_idx], new_state_1d[zero_idx]
            return new_state_1d
        return state_1d
    population = [random_state_1d() for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=fitness, reverse=True)
        if population[0] == goal_1d:
            break
        next_generation = population[:population_size // 2]
        while len(next_generation) < population_size:
            p1, p2 = random.sample(next_generation, 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            next_generation.append(child)
        population = next_generation
    best_1d = population[0]
    if best_1d != goal_1d:
        return None
    return bfs_solve(start_2d, to_2d(best_1d))

def and_or_graph_search_8puzzle(start_2d, goal_2d, max_steps=50, max_possible_states=2):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    
    def is_goal(state_1d):
        return state_1d == goal_1d
    
    def get_successors(state_1d, step):
        successors = []
        for new_state_1d, move in get_neighbors_1d(state_1d):
            if len(move) != 4:
                raise ValueError(f"Invalid move tuple: {move}, expected 4 elements")
            states = [(new_state_1d, False)]  
            for _ in range(max_possible_states - 1):
                perturbed_state = perturb_state(new_state_1d)
                if perturbed_state:
                    states.append((perturbed_state, False))
            successors.append((move, states, step))
        return successors
    def perturb_state(state_1d):
        state = state_1d.copy()
        non_zero_indices = [i for i, val in enumerate(state) if val != 0]
        if len(non_zero_indices) < 2:
            return None
        idx1, idx2 = random.sample(non_zero_indices, 2)
        state[idx1], state[idx2] = state[idx2], state[idx1]
        return state
    
   
    queue = deque([(start_1d, set(), 0, [], [])])  
    while queue:
        state_1d, visited_states, step, path, possible_states = queue.popleft()
        
        if step >= max_steps:
            continue
        if is_goal(state_1d):
           
            updated_possible_states = []
            for step_idx, states in possible_states:
                new_states = []
                for state, chosen in states:
                    new_chosen = chosen or (tuple(state) == tuple(state_1d))
                    new_states.append((state, new_chosen))
                updated_possible_states.append((step_idx, new_states))
            return path, updated_possible_states  
        
        state_tuple = tuple(state_1d)
        if state_tuple in visited_states:
            continue
        visited_states.add(state_tuple)
        
        # OR Search
        successors = get_successors(state_1d, step)
        if not successors:
            continue
        
        # AND Search
        for action, result_states, step_idx in successors:
            if not result_states: 
                continue
            
            new_possible_states = possible_states + [(step_idx, result_states)]
            for next_state, _ in result_states:
                new_visited = visited_states.copy()
                new_path = path + [action]
                queue.append((next_state, new_visited, step + 1, new_path, new_possible_states))
                break
    
    return [], []

def backtracking_csp_solve():
    np.random.seed(42)

    def is_complete(state):
        return np.count_nonzero(state) == 8

    def is_consistent(state, row, col, value):
        if value != 0:
            if value in state:
                return False
            if col > 0 and state[row, col - 1] != 0 and value != state[row, col - 1] + 1:
                return False
            if row > 0 and state[row - 1, col] != 0 and value != state[row - 1, col] + 3:
                return False
        else:  
            return np.count_nonzero(state == 0) <= 1
        return True

    def find_unassigned(state):
        zeros = np.argwhere(state == 0)
        return zeros[0] if zeros.size > 0 else (None, None)

    def recursive_backtracking(state, all_states):
        if is_complete(state):
            all_states.append(state.copy())  
            return True

        row, col = find_unassigned(state)
        if row is None:
            return False

        values = np.arange(1, 9).tolist() + [0]
        np.random.shuffle(values)

        for value in values:
            if is_consistent(state, row, col, value):
                state[row, col] = value
                all_states.append(state.copy())  

                if recursive_backtracking(state, all_states):
                    return True

                state[row, col] = 0
                all_states.append(state.copy())  # Lưu trạng thái sau backtrack

        return False
   
    all_states = []
    recursive_backtracking(np.zeros((3, 3),dtype=int), all_states)
    return all_states
  

# kiểm thử
def check_constraints(state, row, col, value):
    values = [state[i][j] for i in range(3) for j in range(3) 
              if state[i][j] is not None and (i, j) != (row, col)]
    if value in values:
        return False

    if row < 2:
        if col == 0:
            if state[row][1] is not None and state[row][1] != value + 1:
                return False
            if state[row][2] is not None and state[row][2] != value + 2:
                return False
        elif col == 1:
            if state[row][0] is not None and state[row][0] != value - 1:
                return False
            if state[row][2] is not None and state[row][2] != value + 1:
                return False
        elif col == 2:
            if state[row][1] is not None and state[row][1] != value - 1:
                return False
            if state[row][0] is not None and state[row][0] != value - 2:
                return False

    if col < 2:
        if row == 0:
            if state[1][col] is not None and state[1][col] != value + 3:
                return False
            if state[2][col] is not None and state[2][col] != value + 6:
                return False
        elif row == 1:
            if state[0][col] is not None and state[0][col] != value - 3:
                return False
            if state[2][col] is not None and state[2][col] != value + 3:
                return False
        elif row == 2:
            if state[1][col] is not None and state[1][col] != value - 3:
                return False
            if state[0][col] is not None and state[0][col] != value - 6:
                return False

    return True

def calculate_constraints():
    constraints_count = {
        (0, 0): 2, (0, 1): 3, (0, 2): 2,
        (1, 0): 3, (1, 1): 4, (1, 2): 3,
        (2, 0): 2, (2, 1): 3, (2, 2): 0
    }
    return sorted(constraints_count.keys(), key=lambda k: -constraints_count[k])

def generate_and_test():
    state = [[None for _ in range(3)] for _ in range(3)]
    all_states = []
    state_count = [0]
    order = calculate_constraints()
    other_positions = [pos for pos in order if pos != (1, 1) and pos != (2, 2)]
    center_values = random.sample(range(1, 9), 8)

    def backtrack(index):
        all_states.append(np.array([[0 if x is None else x for x in row] for row in state]))
        state_count[0] += 1

        if index == len(other_positions) + 1:
            state[2][2] = 0
            all_states.append(np.array([[0 if x is None else x for x in row] for row in state]))
            state_count[0] += 1
            return True

        if index == 0:
            row, col = (1, 1)
            values = center_values
        else:
            row, col = other_positions[index - 1]
            values = [v for v in range(1, 9) 
                      if v not in [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None]]

        for value in values:
            if check_constraints(state, row, col, value):
                state[row][col] = value
                if backtrack(index + 1):
                    return True
                state[row][col] = None

        return False

    backtrack(0)
    return all_states
# AC3
np.random.seed(52)

DOMAIN = list(range(1, 9)) + [0]

def is_complete(state):
    return np.count_nonzero(state) == 8

def is_consistent(state, row, col, value):
    if value != 0:
        if value in state:
            return False
        if col > 0 and state[row, col - 1] != 0 and value != state[row, col - 1] + 1:
            return False
        if row > 0 and state[row - 1, col] != 0 and value != state[row - 1, col] + 3:
            return False
    else:  
        return np.count_nonzero(state == 0) <= 1
    return True

def find_unassigned(state):
    zeros = np.argwhere(state == 0)
    return zeros[0] if zeros.size > 0 else (None, None)

def revise(domains, xi, xj):
    revised = False
    for x in domains[xi][:]:
        if all(x == y for y in domains[xj]):
            domains[xi].remove(x)
            revised = True
    return revised

def backtracking_with_ac3(state, all_states):
    if is_complete(state):
        all_states.append(state.copy())
        return True

    row, col = find_unassigned(state)
    if row is None:
        return False

    values = DOMAIN.copy()
    np.random.shuffle(values)

    for value in values:
        if is_consistent(state, row, col, value):
            state[row, col] = value
            all_states.append(state.copy()) 

            if backtracking_with_ac3(state, all_states):
                return True

            state[row, col] = 0
            all_states.append(state.copy()) 
    return False

def get_all_states_csp_ac3():
    start_state = np.zeros((3, 3),dtype=int)
    all_states = []
    backtracking_with_ac3(start_state, all_states)
    return all_states

#Q learning
def q_learning_solve(start_2d, goal_2d, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    start_1d = to_1d(start_2d)
    goal_1d = to_1d(goal_2d)
    Q = {}
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  

    def get_action(state_1d, epsilon):
        state_tuple = tuple(state_1d)
        if state_tuple not in Q:
            Q[state_tuple] = {a: 0.0 for a in actions}
        valid_actions = get_valid_actions(state_1d)
        if not valid_actions:
            return None
        if random.uniform(0, 1) < epsilon:
            return random.choice(valid_actions)
        else:
            return max(Q[state_tuple], key=lambda a: Q[state_tuple][a] if a in valid_actions else float('-inf'))

    def get_valid_actions(state_1d):
        zero_idx = get_zero_pos_1d(state_1d)
        row, col = divmod(zero_idx, 3)
        valid = []
        for dx, dy in actions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                valid.append((dx, dy))
        return valid

    # Training
    for _ in range(episodes):
        current_1d = start_1d.copy()
        step_count = 0
        max_steps = 1000
        while tuple(current_1d) != tuple(goal_1d) and step_count < max_steps:
            state_tuple = tuple(current_1d)
            action = get_action(current_1d, epsilon)
            if action is None:
                break
            zero_idx = get_zero_pos_1d(current_1d)
            row, col = divmod(zero_idx, 3)
            new_row, new_col = row + action[0], col + action[1]
            new_idx = new_row * 3 + new_col
            next_state_1d = current_1d.copy()
            next_state_1d[zero_idx], next_state_1d[new_idx] = next_state_1d[new_idx], next_state_1d[zero_idx]
            reward = -1  
            if tuple(next_state_1d) == tuple(goal_1d):
                reward = 100 
            next_state_tuple = tuple(next_state_1d)
            if next_state_tuple not in Q:
                Q[next_state_tuple] = {a: 0.0 for a in actions}
            next_valid_actions = get_valid_actions(next_state_1d)
            max_future_q = max(Q[next_state_tuple][a] for a in next_valid_actions) if next_valid_actions else 0
            Q[state_tuple][action] += alpha * (reward + gamma * max_future_q - Q[state_tuple][action])
            current_1d = next_state_1d
            step_count += 1

    path = []
    current_1d = start_1d.copy()
    visited = set()
    max_steps = 1000
    step = 0
    while tuple(current_1d) != tuple(goal_1d) and step < max_steps:
        state_tuple = tuple(current_1d)
        if state_tuple in visited:
            return None  
        visited.add(state_tuple)
        action = get_action(current_1d, 0)  
        if action is None:
            return None
        zero_idx = get_zero_pos_1d(current_1d)
        row, col = divmod(zero_idx, 3)
        new_row, new_col = row + action[0], col + action[1]
        new_idx = new_row * 3 + new_col
        next_state_1d = current_1d.copy()
        next_state_1d[zero_idx], next_state_1d[new_idx] = next_state_1d[new_idx], next_state_1d[zero_idx]
        path.append((row, col, new_row, new_col))
        current_1d = next_state_1d
        step += 1
    if tuple(current_1d) != tuple(goal_1d):
        return None
    return path

def reload_puzzle():
    global matrix, goal_state, initial_input, goal_input, steps, cost,solve_time, solving, solution, solution_states
    matrix = copy.deepcopy(default_matrix)
    goal_state = copy.deepcopy(default_goal)
    initial_input = copy.deepcopy(default_matrix)
    goal_input = copy.deepcopy(default_goal)
    steps = 0
    cost = 0
    solve_time = 0
    solving = False
    solution = []
    solution_states = []
    path_listbox.clear()
    draw_grid()

def solve_puzzle(selected_algorithm):
    global solving, solution, algorithm, matrix, goal_state, steps, cost, solution_states, solve_time, last_belief_size
    algorithm = selected_algorithm
    matrix = copy.deepcopy(initial_input)
    goal_state = copy.deepcopy(goal_input)
    matrix_1d = to_1d(matrix)
    goal_1d = to_1d(goal_state)
    
    if algorithm != "BACKTRACKING_CSP" and algorithm != "GENERATE & TEST" and algorithm != "AC3":
        if not is_valid_state_1d(matrix_1d) or not is_valid_state_1d(goal_1d):
            path_listbox.items = ["Invalid state: 0-8 must appear once!"]
            draw_grid()
            return
        if not is_solvable(matrix):
            path_listbox.items = ["Puzzle is not solvable!"]
            draw_grid()
            return
    solving = True
    steps = 0
    cost = 0
    current_depth = 0
    solution_states = [copy.deepcopy(matrix)] 
    path_listbox.items = []
    path_listbox.visible_items = 4 if algorithm != "IDS" else 3
    initial_state_str = f"Step 0:\n" + "\n".join(" ".join(f"{num:>2}" for num in row) for row in matrix)
    path_listbox.items.append(initial_state_str)
    draw_grid()
    start = time.time()
    possible_states = []
    if algorithm == "BFS":
        solution = bfs_solve(matrix, goal_state)
    elif algorithm == "DFS":
        solution = dfs_solve(matrix, goal_state)
    elif algorithm == "UCS":
        solution = ucs_solve(matrix, goal_state)
    elif algorithm == "IDS":
        solution, current_depth = ids_solve(matrix, goal_state)
        if solution is None:
            solution = []
    elif algorithm == "GREEDY":
        solution = greedy_solve(matrix, goal_state)
    elif algorithm == "A*":
        solution = a_star_search(matrix, goal_state)
    elif algorithm == "IDA*":
        solution = ida_star(matrix, goal_state)
    elif algorithm == "SIMPLE_HILL_CLIMBING":
        solution = simple_hill_climbing(matrix, goal_state)
    elif algorithm == "STEEPEST_HILL":
        solution = steepest_hill_climbing(matrix, goal_state)
    elif algorithm == "STOCHASTIC_HILL":
        solution = stochastic_hill_climbing(matrix, goal_state)
    elif algorithm == "SIMULATED_ANNEALING":
        solution = simulated_annealing(matrix, goal_state)
    elif algorithm == "BEAM_SEARCH":
        solution = beam_search(matrix, goal_state)
    elif algorithm == "AND-OR":
        solution, possible_states = and_or_graph_search_8puzzle(matrix, goal_state, max_possible_states=2)
    elif algorithm == "GENETIC":
        solution = genetic_algorithm_solve(matrix, goal_state)
    
    elif algorithm == "BACKTRACKING_CSP":
        solution = backtracking_csp_solve()
    elif algorithm == "GENERATE & TEST":
        solution = generate_and_test()
    elif algorithm == "AC3":
         solution = get_all_states_csp_ac3()
    elif algorithm == "Q_LEARNING":
        solution = q_learning_solve(matrix, goal_state)
   
    end = time.time()
    solve_time = end - start
    if solution:
        if algorithm == "BACKTRACKING_CSP" or algorithm == "GENERATE & TEST" or algorithm == "AC3":
            for idx, state in enumerate(solution):
                matrix = state.tolist()  
                steps = idx
                cost = idx
                solution_states.append(copy.deepcopy(matrix))
                draw_grid()
                pygame.time.wait(base_delay)
        else:
            rep_state = to_1d(matrix)
            for idx, move in enumerate(solution, 1):
                steps += 1
                if algorithm == "GREEDY":
                   cost = manhattan_distance_1d(to_1d(matrix),to_1d(goal_input))
                 
                elif algorithm == "A*" or algorithm == "IDA*":
                    cost = steps + manhattan_distance_1d(to_1d(matrix),to_1d(goal_input))
                    
                else:
                    cost += 1
                x1, y1, x2, y2 = move
                
                matrix[x1][y1], matrix[x2][y2] = matrix[x2][y2], matrix[x1][y1]
                solution_states.append(copy.deepcopy(matrix))
                
                move_direction = get_move_direction(x1, y1, x2, y2)
                state_str = f"Step {idx}: {move_direction}\n" + ("(Depth: %d)\n" % current_depth if algorithm == "IDS" else "") + "\n".join(" ".join(f"{num:>2}" for num in row) for row in matrix)
                path_listbox.items.append(state_str)
                
                if algorithm == "AND-OR" and possible_states:
                    for step_idx, states in possible_states:
                        if step_idx == idx - 1:
                            for i, (state_1d, chosen) in enumerate(states, 1):
                                state_2d = [state_1d[i:i+3] for i in range(0, 9, 3)]
                                label = "Chosen State" if chosen else "Possible State"
                                state_str = f"{label} {i}\n" + "\n".join(" ".join(f"{num:>2}" for num in row) for row in state_2d)
                                path_listbox.items.append(state_str)
                                path_listbox.max_scroll = max(0, len(path_listbox.items) - path_listbox.visible_items)
                if solving:
                    path_listbox.scroll_offset = path_listbox.max_scroll
                draw_grid()
                pygame.time.wait(base_delay)
    else:
        path_listbox.items = ["No solution found!"]
        steps = 0
        cost = 0
        solution_states = []
    solving = False
    draw_grid()
def handle_click(pos):
    global solving, active_input
    if not solving:
        selected = algorithm_listbox.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos))
        if selected:
            algorithm_listbox.get_selected()
        selected_speed = speed_listbox.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos))
        if selected_speed:
            set_speed(selected_speed)
        if solve_button.collidepoint(pos):
            active_input = None
            solve_puzzle(algorithm_listbox.get_selected().upper())
        if reload_button.collidepoint(pos):
            active_input = None
            reload_puzzle()
        if not solving:
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if initial_input_boxes[i][j].collidepoint(pos):
                        active_input = ("initial", i, j)
                    elif goal_input_boxes[i][j].collidepoint(pos):
                        active_input = ("goal", i, j)

def handle_input(key):
    global active_input, initial_input, goal_input, matrix
    if active_input:
        input_type, i, j = active_input
        if key in range(pygame.K_0, pygame.K_9 + 1):
            num = key - pygame.K_0
            if num <= 8:
                if input_type == "initial":
                    initial_input[i][j] = num
                    matrix = copy.deepcopy(initial_input)
                else:
                    goal_input[i][j] = num
        elif key == pygame.K_BACKSPACE:
            if input_type == "initial":
                initial_input[i][j] = 0
                matrix = copy.deepcopy(initial_input)
            else:
                goal_input[i][j] = 0
                goal_state = copy.deepcopy(goal_input)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_click(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            else:
                handle_input(event.key)
        elif event.type == pygame.MOUSEWHEEL:
            algorithm_listbox.handle_event(event)
            speed_listbox.handle_event(event)
            if not solving:
                path_listbox.handle_event(event)
    mouse_pos = pygame.mouse.get_pos()
    mouse_over_button = solve_button if solve_button.collidepoint(mouse_pos) else reload_button if reload_button.collidepoint(mouse_pos) else None
    draw_grid()
    clock.tick(FPS)

pygame.quit()
sys.exit()