# (kontynuacja pliku arc_agi_system.py)
import json
import os
# arc_agi_system.py — system operacyjny dla rozwiązywania zadań ARC
import numpy as np
import json
import os

### === GRID === ###
class Grid:
    def __init__(self, pixels):
        self.pixels = np.array(pixels, dtype=int)

    def shape(self):
        return self.pixels.shape

    def copy(self):
        return Grid(self.pixels.copy())

    def flip(self, axis):
        if axis == 'x': return Grid(np.flipud(self.pixels))
        elif axis == 'y': return Grid(np.fliplr(self.pixels))
        else: raise ValueError("Invalid axis")

    def rotate(self, k=1):
        return Grid(np.rot90(self.pixels, k))

    def recolor(self, from_color, to_color):
        new = self.pixels.copy()
        new[new == from_color] = to_color
        return Grid(new)

    def crop(self, y1, y2, x1, x2):
        return Grid(self.pixels[y1:y2, x1:x2])

    def pad(self, pad_width, value=0):
        return Grid(np.pad(self.pixels, pad_width, constant_values=value))

### === OBJECT === ###
class GridObject:
    def __init__(self, pixels, mask=None):
        self.grid = Grid(pixels)
        self.mask = np.array(mask, dtype=bool) if mask is not None else (self.grid.pixels > 0)
        self.bbox = self.compute_bbox()
        self.color_hist = self.color_distribution()

    def compute_bbox(self):
        ys, xs = np.where(self.mask)
        if ys.size == 0: return (0, 0, 0, 0)
        return (ys.min(), ys.max()+1, xs.min(), xs.max()+1)

    def color_distribution(self):
        vals, counts = np.unique(self.grid.pixels[self.mask], return_counts=True)
        return dict(zip(map(int, vals), map(int, counts)))

    def extract_patch(self):
        y1, y2, x1, x2 = self.bbox
        return self.grid.crop(y1, y2, x1, x2)

    def area(self): return np.sum(self.mask)
    def centroid(self):
        ys, xs = np.where(self.mask)
        if len(ys) == 0: return (0, 0)
        return (float(np.mean(ys)), float(np.mean(xs)))

### === TRANSFORM === ###
class Transform:
    def __init__(self, ops):
        self.ops = ops if isinstance(ops, list) else [ops]

    def apply(self, obj: GridObject) -> GridObject:
        result = obj.grid.copy()
        for op in self.ops:
            if op == 'flip_x': result = result.flip('x')
            elif op == 'flip_y': result = result.flip('y')
            elif op == 'rotate_90': result = result.rotate(1)
            elif op == 'rotate_180': result = result.rotate(2)
            elif op == 'rotate_270': result = result.rotate(3)
            elif op == 'orig': continue
            else: raise ValueError(f"Unknown transform: {op}")
        return GridObject(result.pixels)

def detect_transformation(input_grid: Grid, target_grid: Grid):
    transformations = ['orig', 'flip_y', 'flip_x', 'rotate_90', 'rotate_180', 'rotate_270']
    obj = GridObject(input_grid.pixels)
    for t in transformations:
        if np.array_equal(Transform(t).apply(obj).grid.pixels, target_grid.pixels):
            return t
    for t1 in transformations:
        for t2 in transformations:
            tf = Transform([t1, t2])
            if np.array_equal(tf.apply(obj).grid.pixels, target_grid.pixels):
                return [t1, t2]
    return None

### === PATTERN2D === ###
class Pattern2D:
    def __init__(self, base: GridObject, transform_matrix):
        self.base = base
        self.transform_matrix = transform_matrix

    def expand(self):
        reps_y = len(self.transform_matrix)
        reps_x = len(self.transform_matrix[0])
        row_grids = []
        for y in range(reps_y):
            row_tiles = []
            for x in range(reps_x):
                tf = Transform(self.transform_matrix[y][x])
                obj = tf.apply(self.base)
                row_tiles.append(obj.extract_patch().pixels)
            row_grids.append(np.hstack(row_tiles))
        return Grid(np.vstack(row_grids))

### === FALLBACK TILE === ###
def expand_pattern_2d(mask_grid: Grid, tile_grid: Grid, mode='copy_if_1') -> Grid:
    reps_y, reps_x = mask_grid.shape()
    tile_h, tile_w = tile_grid.shape()
    result = np.zeros((reps_y * tile_h, reps_x * tile_w), dtype=int)
    flipped_tile = np.flip(tile_grid.pixels, axis=(0, 1))
    for y in range(reps_y):
        for x in range(reps_x):
            v = mask_grid.pixels[y, x]
            iy, ix = y * tile_h, x * tile_w
            if mode == 'copy_if_1' and v == 1:
                result[iy:iy+tile_h, ix:ix+tile_w] = tile_grid.pixels
            elif mode == 'copy_if_0' and v == 0:
                result[iy:iy+tile_h, ix:ix+tile_w] = tile_grid.pixels
            elif mode == 'copy_flipped_if_1' and v == 1:
                result[iy:iy+tile_h, ix:ix+tile_w] = flipped_tile
            elif mode == 'copy_flipped_if_0' and v == 0:
                result[iy:iy+tile_h, ix:ix+tile_w] = flipped_tile
    return Grid(result)

def detect_fallback_mode(input_grid: Grid, output_grid: Grid):
    mask = (input_grid.pixels > 0).astype(int)
    tile_area = input_grid.shape()[0] * input_grid.shape()[1]
    out_count = np.count_nonzero(output_grid.pixels)
    mask_area = np.count_nonzero(mask)
    if out_count == mask_area * mask_area: return 'copy_if_1'
    if out_count == (tile_area - mask_area) * mask_area: return 'copy_if_0'
    if out_count == mask_area * (tile_area - mask_area): return 'copy_flipped_if_1'
    if out_count == (tile_area - mask_area) * (tile_area - mask_area): return 'copy_flipped_if_0'
    return 'none'

### === DSL OPERATIONS & INTERPRETER === ###
class Operation:
    def apply(self, obj: GridObject) -> GridObject:
        raise NotImplementedError

class FlipX(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        return GridObject(obj.grid.flip('x').pixels)

class FlipY(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        return GridObject(obj.grid.flip('y').pixels)

class Rotate90(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        return GridObject(obj.grid.rotate(1).pixels)

class Rotate180(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        return GridObject(obj.grid.rotate(2).pixels)

class Recolor(Operation):
    def __init__(self, from_color, to_color):
        self.from_color = from_color
        self.to_color = to_color

    def apply(self, obj: GridObject) -> GridObject:
        return GridObject(obj.grid.recolor(self.from_color, self.to_color).pixels)

class Sequence(Operation):
    def __init__(self, steps):
        self.steps = steps

    def apply(self, obj: GridObject) -> GridObject:
        for op in self.steps:
            obj = op.apply(obj)
        return obj

class Identity(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        return obj

def run_program(program: Operation, grid: Grid) -> Grid:
    obj = GridObject(grid.pixels)
    return program.apply(obj).grid


### === HEURYSTYKA: suggest_grid_filling_strategy === ###
def suggest_grid_filling_strategy(input_grid, output_grid):
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
    if oh % ih != 0 or ow % iw != 0:
        return None

    reps_y = oh // ih
    reps_x = ow // iw

    tile_transforms = ['orig', 'flip_x', 'flip_y', 'rotate_90', 'rotate_180', 'rotate_270']
    mask_types = {
        'nonzero': (input_grid.pixels != 0).astype(int),
        'zero': (input_grid.pixels == 0).astype(int),
    }

    best_match = None
    best_score = -1

    for mask_name, mask_array in mask_types.items():
        for tf in tile_transforms:
            match_count = 0
            total_count = reps_y * reps_x
            tile = Transform(tf).apply(GridObject(input_grid.pixels)).grid

            for y in range(reps_y):
                for x in range(reps_x):
                    iy, ix = y * ih, x * iw
                    out_patch = output_grid.pixels[iy:iy+ih, ix:ix+iw]
                    condition = mask_array[y % ih, x % iw]
                    expected = tile.pixels if condition else np.zeros_like(tile.pixels)
                    if np.array_equal(out_patch, expected):
                        match_count += 1

            score = match_count / total_count
            if score > best_score:
                best_score = score
                best_match = {
                    'mode': 'mask_and_tile',
                    'mask_type': mask_name,
                    'tile_op': tf,
                    'score': round(score, 3)
                }

    if best_score > 0.8:
        return best_match
    return None

### === DEBUGGER / TESTER === ###
def generate_candidate_programs(input_grid: Grid, output_grid: Grid):
    candidates = []
    explanations = []

    # Heurystyka 1: identyczny rozmiar + histogram kolorów
    if input_grid.shape() == output_grid.shape():
        in_hist = np.bincount(input_grid.pixels.flatten(), minlength=10)
        out_hist = np.bincount(output_grid.pixels.flatten(), minlength=10)
        if np.all(in_hist == out_hist):
            for tf in ['orig', 'flip_x', 'flip_y', 'rotate_90', 'rotate_180', 'rotate_270']:
                obj = GridObject(input_grid.pixels)
                result = Transform(tf).apply(obj).grid
                candidates.append((Sequence([]) if tf == 'orig' else Transform(tf), f"global_transform: {tf}"))

    # Heurystyka 2: pattern2D
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
    if oh % ih == 0 and ow % iw == 0:
        scale = (oh // ih) * (ow // iw)
        in_hist = np.bincount(input_grid.pixels.flatten(), minlength=10)
        out_hist = np.bincount(output_grid.pixels.flatten(), minlength=10)
        if np.all((in_hist * scale >= out_hist) | (in_hist == 0)):
            reps_y, reps_x = oh // ih, ow // iw
            matrix = [[detect_transformation(input_grid, Grid(output_grid.pixels[y*ih:(y+1)*ih, x*iw:(x+1)*iw]))
                       or 'orig' for x in range(reps_x)] for y in range(reps_y)]
            candidates.append((Pattern2D(GridObject(input_grid.pixels), matrix), "pattern2d from grid tiling"))

    # Heurystyka 3: maska + kafel
    suggestion = suggest_grid_filling_strategy(input_grid, output_grid)
    if suggestion:
        mask_type = suggestion['mask_type']
        tile_op = suggestion['tile_op']
        mask = (input_grid.pixels != 0).astype(int) if mask_type == 'nonzero' else (input_grid.pixels == 0).astype(int)
        tile = Transform(tile_op).apply(GridObject(input_grid.pixels)).grid
        mode = f"copy_if_{1 if mask_type == 'nonzero' else 0}"
        result = expand_pattern_2d(Grid(mask), tile, mode)
        candidates.append((result, f"mask_and_tile: {mask_type} + {tile_op}"))

    return candidates


def debug_task(task_id, dataset_path="./"):
    challenges_file = os.path.join(dataset_path, "arc-agi_training_challenges.json")
    solutions_file = os.path.join(dataset_path, "arc-agi_training_solutions.json")
    with open(challenges_file) as f:
        challenges = json.load(f)
    with open(solutions_file) as f:
        solutions = json.load(f)

    input_data = challenges[task_id]["train"][0]["input"]
    output_data = challenges[task_id]["train"][0]["output"]
    input_grid = Grid(input_data)
    output_grid = Grid(output_data)

    candidates = generate_candidate_programs(input_grid, output_grid)
    for program, explanation in candidates:
        try:
            if isinstance(program, Grid):  # fallback
                result = program
            else:
                result = run_program(program, input_grid)
            if np.array_equal(result.pixels, output_grid.pixels):
                return True, explanation
        except Exception as e:
            continue
    return False, "no match"


def debug_many_tasks(task_ids, dataset_path="./"):
    success, failure, report_lines = [], [], []
    for task_id in task_ids:
        print(f"===== {task_id} =====")
        try:
            ok, why = debug_task(task_id, dataset_path)
            if ok:
                print(f"✅ {task_id} passed via {why}")
                success.append(task_id)
                report_lines.append(f"✅ {task_id}: {why}")
            else:
                print(f"❌ {task_id} failed")
                failure.append(task_id)
                report_lines.append(f"❌ {task_id}: {why}")
        except Exception as e:
            print(f"💥 {task_id} crashed: {e}")
            failure.append(task_id)
            report_lines.append(f"💥 {task_id}: crash")

    print("\n=== PODSUMOWANIE ===")
    print(f"✅ Udało się: {len(success)} / {len(task_ids)}")
    print(f"❌ Nieudane: {len(failure)}")

    with open("debug_success.txt", "w") as f:
        for tid in success: f.write(tid + "\n")

    with open("debug_failure.txt", "w") as f:
        for tid in failure: f.write(tid + "\n")

    with open("debug_report.txt", "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    TASK_IDS = [
        "00576224", "0c786b71", "3af2c5a8", "3c9b0459", "4c4377d9", "59341089",
        "6150a2bd", "62c24649", "67a3c6ac", "67e8384a", "68b16354", "6d0aefbc",
        "6fa7a44f", "74dd1130", "7953d61e", "7fe24cdd", "833dafe3", "8be77c9e",
        "8d5021e8", "963e52fc", "9dfd6313", "a416b8f3", "bc4146bd", "c48954c1",
        "c9e6f938", "cf5fd0ad", "ed98d772"
    ]
    debug_many_tasks(TASK_IDS, dataset_path="dane/")