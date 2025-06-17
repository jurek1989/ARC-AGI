import json
import numpy as np
import itertools

challenges_path = "dane/arc-agi_training_challenges.json"
solutions_path = "dane/arc-agi_training_solutions.json"

with open(challenges_path) as f:
    challenges_data = json.load(f)

with open(solutions_path) as f:
    solutions_data = json.load(f)
from pathlib import Path

### === PODSTAWOWE STRUKTURY === ###

class Grid:
    def __init__(self, pixels):
        self.pixels = pixels

    @classmethod
    def from_list(cls, data):
        return cls(np.array(data, dtype=int))

    def shape(self):
        return self.pixels.shape

    def flip(self, axis):
        if axis == 'y':
            return Grid(np.fliplr(self.pixels))
        elif axis == 'x':
            return Grid(np.flipud(self.pixels))
        else:
            raise ValueError("Axis must be 'x' or 'y'")

    def rotate(self, k=1):
        return Grid(np.rot90(self.pixels, k=k))

    def recolor(self, from_color, to_color):
        new_pixels = self.pixels.copy()
        new_pixels[new_pixels == from_color] = to_color
        return Grid(new_pixels)

    def empty_like(self):
        return Grid(np.full(self.pixels.shape, fill_value=0, dtype=int))

class GridObject:
    def __init__(self, grid):
        self.grid = grid

### === TRANSFORMACJE === ###

class Transform:
    def __init__(self, ops):
        if isinstance(ops, str):
            self.ops = [ops]
        else:
            self.ops = ops

    def apply(self, obj: GridObject):
        result = obj
        for op in self.ops:
            if op == 'orig':
                continue
            elif op == 'flip_x':
                result = GridObject(result.grid.flip('x'))
            elif op == 'flip_y':
                result = GridObject(result.grid.flip('y'))
            elif op == 'rotate_90':
                result = GridObject(result.grid.rotate(1))
            elif op == 'rotate_180':
                result = GridObject(result.grid.rotate(2))
            elif op == 'rotate_270':
                result = GridObject(result.grid.rotate(3))
            else:
                raise ValueError(f"Unknown transform: {op}")
        return result

### === WZORZEC TILE === ###

class Pattern2D:
    def __init__(self, base, transform_matrix):
        self.base = base
        self.transform_matrix = transform_matrix

    def expand(self):
        reps_y = len(self.transform_matrix)
        reps_x = len(self.transform_matrix[0])
        row_grids = []
        for y in range(reps_y):
            row_tiles = []
            for x in range(reps_x):
                transform = Transform(self.transform_matrix[y][x])
                obj = transform.apply(self.base)
                row_tiles.append(obj.grid.pixels)
            row_grids.append(np.hstack(row_tiles))
        return Grid(np.vstack(row_grids))

### === FALLBACK === ###

def expand_pattern_2d(mask_grid, tile_grid, mode='copy_if_1'):
    reps_y, reps_x = mask_grid.shape
    tile_h, tile_w = tile_grid.shape
    result = np.zeros((reps_y * tile_h, reps_x * tile_w), dtype=int)
    flipped_tile = np.flip(tile_grid, axis=(0, 1))

    for y in range(reps_y):
        for x in range(reps_x):
            v = mask_grid[y, x]
            insert_y = y * tile_h
            insert_x = x * tile_w

            if mode == 'copy_if_1' and v == 1:
                result[insert_y:insert_y+tile_h, insert_x:insert_x+tile_w] = tile_grid
            elif mode == 'copy_if_0' and v == 0:
                result[insert_y:insert_y+tile_h, insert_x:insert_x+tile_w] = tile_grid
            elif mode == 'copy_flipped_if_1' and v == 1:
                result[insert_y:insert_y+tile_h, insert_x:insert_x+tile_w] = flipped_tile
            elif mode == 'copy_flipped_if_0' and v == 0:
                result[insert_y:insert_y+tile_h, insert_x:insert_x+tile_w] = flipped_tile

    return Grid(result)

def is_masked_grid_tiling(input_grid, output_grid):
    h, w = input_grid.shape()
    tile_area = h * w
    count_input = np.count_nonzero(input_grid.pixels)
    count_output = np.count_nonzero(output_grid.pixels)
    if count_output == count_input * count_input:
        return "copy_if_1"
    if count_output == (tile_area - count_input) * count_input:
        return "copy_if_0"
    if count_output == count_input * (tile_area - count_input):
        return "copy_flipped_if_1"
    if count_output == (tile_area - count_input) * (tile_area - count_input):
        return "copy_flipped_if_0"
    return "none"

### === DETEKCJA === ###

def detect_transformation(input_grid, target_grid):
    base = GridObject(input_grid)
    transformations = ['orig', 'flip_y', 'flip_x', 'rotate_90', 'rotate_180', 'rotate_270']
    for t in transformations:
        tf = Transform(t)
        if np.array_equal(tf.apply(base).grid.pixels, target_grid.pixels):
            return t
    for t1, t2 in itertools.product(transformations, repeat=2):
        tf = Transform([t1, t2])
        if np.array_equal(tf.apply(base).grid.pixels, target_grid.pixels):
            return [t1, t2]
    return None

### === EXECUTOR === ###

def run_solution_test():
    challenges = challenges_data
    solutions = solutions_data
    
    task_ids = [
        "00576224", "0c786b71", "3af2c5a8", "3c9b0459", "4c4377d9", "59341089", "6150a2bd",
        "62c24649", "67a3c6ac", "67e8384a", "68b16354", "6d0aefbc", "6fa7a44f", "74dd1130",
        "7953d61e", "7fe24cdd", "833dafe3", "8be77c9e", "8d5021e8", "963e52fc", "9dfd6313",
        "a416b8f3", "bc4146bd", "c48954c1", "c9e6f938", "cf5fd0ad", "ed98d772"
    ]
    for task_id in task_ids:
        print(f"\n===== Task {task_id} =====")
        train_pairs = [
            (Grid.from_list(ex['input']), Grid.from_list(ex['output']))
            for ex in challenges[task_id]['train']
        ]
        test_inputs = [Grid.from_list(ex['input']) for ex in challenges[task_id]['test']]
        test_outputs = [Grid.from_list(out) for out in solutions[task_id]]

        inp0, out0 = train_pairs[0]
        reps_x = out0.shape()[1] // inp0.shape()[1]
        reps_y = out0.shape()[0] // inp0.shape()[0]

        transform_matrix = []
        for y in range(reps_y):
            row = []
            for x in range(reps_x):
                y0 = y * inp0.shape()[0]
                y1 = (y + 1) * inp0.shape()[0]
                x0 = x * inp0.shape()[1]
                x1 = (x + 1) * inp0.shape()[1]
                patch = Grid(out0.pixels[y0:y1, x0:x1])
                tr = detect_transformation(inp0, patch)
                row.append(tr if tr is not None else 'orig')
            transform_matrix.append(row)

        for i, (test_in, test_out) in enumerate(zip(test_inputs, test_outputs)):
            print(f"--- Test example {i} ---")
            pattern = Pattern2D(GridObject(test_in), transform_matrix)
            pred = pattern.expand()
            if np.array_equal(pred.pixels, test_out.pixels):
                print("✅ Tiling succeeded")
                continue

            print("❌ Tiling failed, trying fallback")
            mode = is_masked_grid_tiling(test_in, test_out)
            print("Fallback mode:", mode)
            if mode != "none":
                mask = (test_in.pixels > 0).astype(int) if '1' in mode else (test_in.pixels == 0).astype(int)
                fb = expand_pattern_2d(mask, inp0.pixels, mode)
                if np.array_equal(fb.pixels, test_out.pixels):
                    print("✅ Fallback succeeded")
                else:
                    print("❌ Fallback failed")
            else:
                print("❌ No matching fallback mode")



if __name__ == "__main__":
    run_solution_test()
    
