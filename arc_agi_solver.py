"""
# ARC Prize 2025 – Konkurs AGI: celem jest rozwiązywanie zadań wymagających abstrakcyjnego rozumowania.
# Zadania polegają na przekształcaniu siatek (gridów) złożonych z liczb 0–9 na podstawie przykładowych par (input/output).
# Każde zadanie ma strukturę:
# {
#   "train": [{"input": [[...]], "output": [[...]]}, ...],
#   "test": [{"input": [[...]]}, ...]
# }
# Przykład:
# input:  [[1, 0],      output: [[0, 1],
#          [0, 1]]                [1, 0]]
# -> symetria względem przekątnej

# Celem jest wygenerowanie poprawnego outputu dla każdego test inputu.
# Tylko **dokładne** dopasowanie do prawidłowego rozwiązania (perfect match) jest punktowane.
# Skuteczność modelu to odsetek trafionych outputów spośród wszystkich testów.

# Wskazówki:
# - Zadania nie są losowe – większość opiera się na geometrycznych transformacjach, kolorach, strukturach bloków.
# - Nie działają proste LLM-y, CNN-y czy brute-force (setki transformacji) – zadania są zbyt zróżnicowane.
# - Przykładowe podejścia: reguły symboliczne, kompozycje prostych operacji, programy generatywne.
# - Model musi być zdolny do generalizacji – nie można "uczyć się" konkretnych tasków testowych.

# Wymagania:
# - Czas działania notebooka: max 12h (CPU/GPU), bez Internetu.
# - submission.json musi zawierać DWIE próby dla każdego test inputu (nie można pomijać attempt_2).
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

"""
ARC AGI Solver — Wersja obiektowa i DSL

Moduł zawiera komponenty służące do reprezentacji, analizy i transformacji siatek ARC w kontekście generalizacji i semantycznego rozumowania.
Zawiera podstawowe klasy do pracy z siatkami (`Grid`), obiektami (`GridObject`), transformacjami (`Transform`), operacjami DSL oraz heurystykami kaflowymi.

Sekcje:
- GRID: reprezentacja siatki i operacje podstawowe
- OBJECT: wykryte obiekty i ich właściwości
- TRANSFORM: rotacje i flipy, wykrywanie przekształceń
- DSL: semantyczne operacje (obiektowe)
- PATTERN2D: wzorce kaflowe i ich rozszerzanie
- MASK + TILE ENGINE: stare heurystyki oparte na maskach
- HEURYSTYKA: porównanie input/output, fallbacki
"""

### === GRID === ###
# podstawowe operacje na siatkach
class Grid:
    def __init__(self, pixels):
        # Reprezentuje pełną siatkę ARC (np. input lub output)
        # pixels: 2D numpy array (lub konwertowalny), typu int, kolory jako liczby
        self.pixels = np.array(pixels, dtype=int)

    def shape(self):
        # Zwraca krotkę (wysokość, szerokość)
        return self.pixels.shape

    def copy(self):
        # Tworzy nową kopię grida (deep copy)
        return Grid(self.pixels.copy())

    def flip(self, axis):
        # Zwraca nowy Grid po odbiciu:
        # axis = 'x' → flip w pionie (góra-dół),
        # axis = 'y' → flip w poziomie (lewo-prawo)
        if axis == 'x': return Grid(np.flipud(self.pixels))
        elif axis == 'y': return Grid(np.fliplr(self.pixels))
        else: raise ValueError("Invalid axis")

    def rotate(self, k=1):
        # Rotacja o 90 stopni * k razy (domyślnie jedna rotacja w lewo)
        return Grid(np.rot90(self.pixels, k))

    def recolor(self, from_color, to_color):
        # Zwraca grid, gdzie każdy piksel from_color jest zamieniony na to_color
        new = self.pixels.copy()
        new[new == from_color] = to_color
        return Grid(new)

    def crop(self, y1, y2, x1, x2):
        # Zwraca prostokątny wycinek grida (subgrid)
        # Współrzędne odpowiadają indeksowaniu NumPy: [y1:y2, x1:x2]
        return Grid(self.pixels[y1:y2, x1:x2])

    def pad(self, pad_width, value=0):
        # Dodaje padding (ramkę) dookoła siatki, domyślnie wypełnioną zerami
        return Grid(np.pad(self.pixels, pad_width, constant_values=value))

### === OBJECT === ###
# reprezentacja wyodrębnionych obiektów, bbox, maska, centroid, kolory
class GridObject:
    def __init__(self, pixels, mask=None):
        # Obiekt wyodrębniony z grida (np. pojedynczy kształt, tło, symbol)
        # pixels: pełny patch grida (wycinek siatki z bboxa), typowo Grid.pixels[y1:y2, x1:x2]
        # mask: binarna maska 2D (bool), lokalna względem patcha (czyli mask.shape == pixels.shape)
        # jeśli maska niepodana, przyjmujemy maskę: pixels > 0
        self.grid = Grid(pixels)
        self.mask = np.array(mask, dtype=bool) if mask is not None else (self.grid.pixels > 0)

        # Bounding box (relatywny do oryginalnego grida): (top, bottom, left, right)
        self.bbox = self.compute_bbox()

        # Histogram kolorów w masce: {kolor: liczba wystąpień}
        self.color_hist = self.color_distribution()

        # Czy ten obiekt jest kandydatem na tło (inicjalnie False — do ustawienia ręcznego)
        self.is_background = False

    def compute_bbox(self):
        # Oblicza najmniejszy prostokąt zawierający maskę (na podstawie maski)
        ys, xs = np.where(self.mask)
        if ys.size == 0: return (0, 0, 0, 0)  # pusty obiekt
        return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)

    def color_distribution(self):
        # Zwraca histogram kolorów dla pikseli w masce
        vals, counts = np.unique(self.grid.pixels[self.mask], return_counts=True)
        return dict(zip(map(int, vals), map(int, counts)))

    def extract_patch(self):
        # Zwraca patch grida z wyciętym bboxem (bez maski)
        y1, y2, x1, x2 = self.bbox
        return self.grid.crop(y1, y2, x1, x2)

    def area(self):
        # Liczba aktywnych pikseli w masce (czyli powierzchnia obiektu)
        return np.sum(self.mask)

    def centroid(self):
        # Geometryczny środek masy obiektu (uśrednione współrzędne w obrębie maski)
        ys, xs = np.where(self.mask)
        if len(ys) == 0: return (0, 0)
        return (float(np.mean(ys)), float(np.mean(xs)))

    def __repr__(self):
        # Tekstowa reprezentacja obiektu: bbox, area, kolory
        y1, y2, x1, x2 = self.bbox
        return (f"<GridObject bbox=({y1},{y2},{x1},{x2}) "
                f"area={self.area()} "
                f"colors={self.color_hist} "
                f"{'BACKGROUND' if self.is_background else ''}>")
    
    def features(self) -> dict:
        h, w = self.mask.shape
        bbox_area = h * w
        filled_area = np.count_nonzero(self.mask)

        # Kształt geometryczny
        is_rect = filled_area == bbox_area
        is_hline = h == 1 and filled_area == w
        is_vline = w == 1 and filled_area == h
        is_square = is_rect and h == w

        # Priorytet kształtu: linie > kwadrat > prostokąt > inne
        if is_hline:
            shape_type = "line_h"
        elif is_vline:
            shape_type = "line_v"
        elif is_square:
            shape_type = "square"
        elif is_rect:
            shape_type = "rectangle"
        else:
            shape_type = "other"

        # Kolory
        is_uniform = len(self.color_hist) == 1
        color_vector = [self.color_hist.get(i, 0) for i in range(10)]
        main_color = max(self.color_hist.items(), key=lambda kv: kv[1])[0] if self.color_hist else None

        # Czy dotyka granic siatki
        touches = (
            self.bbox[0] == 0 or
            self.bbox[1] == self.grid.shape()[0] or
            self.bbox[2] == 0 or
            self.bbox[3] == self.grid.shape()[1]
        )

        return {
            "area": filled_area,
            "height": h,
            "width": w,
            "aspect_ratio": round(w / h, 2) if h != 0 else None,
            "shape_type": shape_type,
            "is_uniform_color": is_uniform,
            "color_count": len(self.color_hist),
            "main_color": main_color,
            "color_vector": color_vector,
            "touches_border": touches,
        }
        



from scipy.ndimage import label
from collections import Counter

def extract_all_object_views(grid: Grid) -> dict[str, List[GridObject]]:
    """
    Zwraca 4 różne widoki obiektów:
    - 'conn4_color': komponenty spójne tego samego koloru (4-sąsiedztwo)
    - 'conn8_color': j.w., ale 8-sąsiedztwo
    - 'conn4_multicolor': komponenty wszystkich pikseli ≠ 0 (4-sąsiedztwo)
    - 'conn8_multicolor': j.w., ale 8-sąsiedztwo

    Każdy GridObject ma ustawiony bbox, mask, color_hist, is_background
    """
    views = {}
    bg_color = Counter(grid.pixels.flatten()).most_common(1)[0][0]
    H, W = grid.shape()

    for conn in [4, 8]:
        structure = None if conn == 4 else np.ones((3, 3), dtype=int)

        # --- color-based ---
        color_objs = []
        for color in np.unique(grid.pixels):
            color_mask = (grid.pixels == color)
            labeled, num = label(color_mask, structure=structure)
            for i in range(1, num + 1):
                mask = (labeled == i)
                if np.count_nonzero(mask) == 0:
                    continue
                y1, y2 = np.where(mask)[0].min(), np.where(mask)[0].max() + 1
                x1, x2 = np.where(mask)[1].min(), np.where(mask)[1].max() + 1
                patch = grid.pixels[y1:y2, x1:x2]
                local_mask = mask[y1:y2, x1:x2]
                obj = GridObject(patch, mask=local_mask)
                color_objs.append(obj)
        _mark_background(color_objs, bg_color, H, W)
        views[f"conn{conn}_color"] = color_objs

        # --- multicolor (nonzero) ---
        multicolor_objs = []
        mask = (grid.pixels != 0)
        labeled, num = label(mask, structure=structure)
        for i in range(1, num + 1):
            component = (labeled == i)
            if np.count_nonzero(component) == 0:
                continue
            y1, y2 = np.where(component)[0].min(), np.where(component)[0].max() + 1
            x1, x2 = np.where(component)[1].min(), np.where(component)[1].max() + 1
            patch = grid.pixels[y1:y2, x1:x2]
            local_mask = component[y1:y2, x1:x2]
            obj = GridObject(patch, mask=local_mask)
            multicolor_objs.append(obj)
        _mark_background(multicolor_objs, bg_color, H, W)
        views[f"conn{conn}_multicolor"] = multicolor_objs

    return views


def _mark_background(objs, bg_color, grid_h, grid_w):
    # Prosta reguła: jeśli obiekt ma tylko kolor tła i dotyka krawędzi → uznaj za tło
    for obj in objs:
        if obj.color_hist.get(bg_color, 0) == obj.area():
            y1, y2, x1, x2 = obj.bbox
            if y1 == 0 or y2 == grid_h or x1 == 0 or x2 == grid_w:
                obj.is_background = True

def visualize_objects(grid: Grid, objects: List[GridObject]):
    import matplotlib.pyplot as plt
    import numpy as np

    pixels = grid.pixels
    h, w = pixels.shape

    fig, ax = plt.subplots(figsize=(w / 2, h / 2))

    # Wyświetl grid z oryginalnymi kolorami
    cmap = plt.get_cmap("tab10")  # 10 kolorów, zgodnie z wartościami 0–9
    ax.imshow(pixels, cmap=cmap, vmin=0, vmax=9)

    # Ustaw siatkę
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Opcjonalnie: obramowanie całej siatki (np. czerwone tło jeśli coś nie pasuje)
    ax.set_title("Grid wejściowy z oryginalnymi kolorami i siatką")

    plt.tight_layout()
    plt.show()

    # Tekstowo: wypisz obiekty
    print("Legenda:")
    for i, obj in enumerate(objects, 1):
        main_color = max(obj.color_hist.items(), key=lambda kv: kv[1])[0] if obj.color_hist else "∅"
        tag = "TŁO" if obj.is_background else "OBIEKT"
        print(f" {i}: kolor={main_color}, area={obj.area()}, bbox={obj.bbox}, {tag}")

### === MATCHING === ###
def compute_match_cost(obj1: GridObject, obj2: GridObject) -> float:
    """
    Oblicza koszt dopasowania między dwoma obiektami.
    Im mniejszy koszt, tym lepsze dopasowanie.
    """
    f1 = obj1.features()
    f2 = obj2.features()

    # Odległość centroidów (ważna przy przesunięciach)
    c1y, c1x = obj1.centroid()
    c2y, c2x = obj2.centroid()
    centroid_dist = ((c1y - c2y) ** 2 + (c1x - c2x) ** 2) ** 0.5

    # Różnica w area
    area_diff = abs(f1["area"] - f2["area"]) / max(f1["area"], f2["area"], 1)

    # Shape penalty: 0 jeśli identyczny, 1 jeśli inny
    shape_penalty = 0 if f1["shape_type"] == f2["shape_type"] else 1

    # Różnica kolorów (L1 norm)
    color_diff = sum(abs(a - b) for a, b in zip(f1["color_vector"], f2["color_vector"]))

    # Można ważyć składniki, np. centroid najważniejszy
    return (
        1.0 * centroid_dist +
        1.0 * area_diff +
        2.0 * shape_penalty +
        0.5 * color_diff
    )

def diff_features(obj1: GridObject, obj2: GridObject) -> dict:
    """
    Zwraca słownik: które cechy się różnią między obj1 a obj2
    """
    f1 = obj1.features()
    f2 = obj2.features()
    diffs = {}

    for key in f1:
        if f1[key] != f2.get(key):
            diffs[key] = (f1[key], f2.get(key))

    return diffs

from scipy.optimize import linear_sum_assignment

def match_objects(input_objs: List[GridObject], output_objs: List[GridObject]) -> List[Tuple[GridObject, GridObject, dict]]:
    """
    Dopasowuje obiekty z inputu do obiektów z outputu na podstawie kosztu dopasowania.
    Zwraca listę par (input_obj, output_obj, diff_dict)
    """
    if not input_objs or not output_objs:
        return []

    n, m = len(input_objs), len(output_objs)
    cost_matrix = np.zeros((n, m))

    # Oblicz macierz kosztów
    for i, obj_in in enumerate(input_objs):
        for j, obj_out in enumerate(output_objs):
            cost_matrix[i, j] = compute_match_cost(obj_in, obj_out)

    # Znajdź optymalne dopasowanie
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        obj_in = input_objs[i]
        obj_out = output_objs[j]
        diff = diff_features(obj_in, obj_out)
        matches.append((obj_in, obj_out, diff))

    return matches

def print_matches(matches: List[Tuple[GridObject, GridObject, dict]]):
    """
    Czytelny wydruk dopasowań input ↔ output obiektów z różnicami cech.
    """
    for idx, (obj_in, obj_out, diff) in enumerate(matches, 1):
        print(f"--- Dopasowanie {idx} ---")
        print(f"IN : bbox={obj_in.bbox}, area={obj_in.area()}, main_color={obj_in.features()['main_color']}")
        print(f"OUT: bbox={obj_out.bbox}, area={obj_out.area()}, main_color={obj_out.features()['main_color']}")

        if diff:
            print(" ZMIANY:")
            for k, (v1, v2) in diff.items():
                print(f"  - {k}: {v1} → {v2}")
        else:
            print(" ✅ Brak różnic")
        print()


### === TRANSFORM === ###
# class Transform, apply_grid_transform, detect_transformation
class Transform:
    def __init__(self, ops):
        # Sekwencja operacji na gridzie (np. rotacje, flipy)
        # ops: lista nazw operacji jako stringi, np. ['rotate_90'], ['flip_x', 'flip_y']
        # Obsługiwane: 'flip_x', 'flip_y', 'rotate_90', 'rotate_180', 'rotate_270', 'orig'
        self.ops = ops if isinstance(ops, list) else [ops]

    def apply(self, obj: GridObject) -> GridObject:
        # Zastosuj sekwencję operacji do obiektu (na jego gridzie)
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
    

def apply_grid_transform(grid: Grid, transform):
    # Zastosuj pojedynczą transformację na gridzie (bez GridObject)
    # transform: string, np. 'flip_x', 'rotate_180', 'orig'
    if transform == 'orig':
        return grid.copy()
    elif transform == 'flip_x':
        return grid.flip('x')
    elif transform == 'flip_y':
        return grid.flip('y')
    elif transform == 'rotate_90':
        return grid.rotate(1)
    elif transform == 'rotate_180':
        return grid.rotate(2)
    elif transform == 'rotate_270':
        return grid.rotate(3)
    else:
        raise ValueError(f"Unknown transform: {transform}")


def detect_transformation(input_grid: Grid, target_grid: Grid, verbose=False):
    # Próbuj wykryć transformację (flip / rotacja) która przekształca input → output
    # Zwraca pojedynczy transform lub listę dwóch (jeśli złożone), lub None

    transformations = ['orig', 'flip_x', 'flip_y', 'rotate_90', 'rotate_180', 'rotate_270']

    # Przypadki pojedyncze
    for t in transformations:
        transformed = apply_grid_transform(input_grid, t)
        if np.array_equal(transformed.pixels, target_grid.pixels):
            return t

    # Przypadki złożone (kompozycja dwóch transformacji)
    for t1 in transformations:
        for t2 in transformations:
            g1 = apply_grid_transform(input_grid, t1)
            g2 = apply_grid_transform(g1, t2)
            if np.array_equal(g2.pixels, target_grid.pixels):
                return [t1, t2]

    return None  # nic nie pasuje

### === DSL === ###
# semantyczne operacje obiektowe
class Operation:
    def apply(self, obj: GridObject) -> GridObject:
        # Interfejs operacji DSL — każda operacja działa na GridObject
        raise NotImplementedError


class FlipX(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        # Odbicie obiektu w pionie (oś pozioma)
        return GridObject(obj.grid.flip('x').pixels)


class FlipY(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        # Odbicie obiektu w poziomie (oś pionowa)
        return GridObject(obj.grid.flip('y').pixels)


class Rotate90(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        # Rotacja o 90 stopni w lewo
        return GridObject(obj.grid.rotate(1).pixels)


class Rotate180(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        # Rotacja o 180 stopni
        return GridObject(obj.grid.rotate(2).pixels)


class Recolor(Operation):
    def __init__(self, from_color, to_color):
        # Zmiana koloru from_color → to_color wewnątrz obiektu
        self.from_color = from_color
        self.to_color = to_color

    def apply(self, obj: GridObject) -> GridObject:
        return GridObject(obj.grid.recolor(self.from_color, self.to_color).pixels)


class Sequence(Operation):
    def __init__(self, steps):
        # Kompozycja operacji: stosuj po kolei
        self.steps = steps

    def apply(self, obj: GridObject) -> GridObject:
        for op in self.steps:
            obj = op.apply(obj)
        return obj


class Identity(Operation):
    def apply(self, obj: GridObject) -> GridObject:
        # Operacja tożsamościowa (nic nie zmienia)
        return obj


def run_program(program, grid: Grid) -> Grid:
    # Uruchom program DSL (Operation lub Pattern2D) na całym gridzie
    # Jeśli Pattern2D → generuje nowy grid z powieleniem obiektu bazowego
    # Jeśli Operation → działa na GridObject z pełnym gridem jako "obiektem"
    if isinstance(program, Pattern2D):
        return program.expand()
    obj = GridObject(grid.pixels)
    return program.apply(obj).grid

class DSLOp:
    """
    Bazowa klasa operacji DSL. Wszystkie operacje dziedziczą po niej
    i muszą zaimplementować apply(grid, objects) → (new_grid, new_objects)
    """
    def apply(self, grid: Grid, objects: List[GridObject]) -> Tuple[Grid, List[GridObject]]:
        raise NotImplementedError


class Translate(DSLOp):
    """
    Przesuwa obiekt o (dx, dy) — bez kolizji. Modyfikuje grid i odświeża obiekty.
    """
    def __init__(self, obj_id, dx, dy):
        self.obj_id = obj_id
        self.dx = dx
        self.dy = dy

    def apply(self, grid, objects):
        obj = objects[self.obj_id]
        new_pixels = np.copy(grid.pixels)

        # Wymaż stary obiekt
        y1, y2, x1, x2 = obj.bbox
        new_pixels[y1:y2, x1:x2][obj.mask] = 0

        # Wstaw w nowym miejscu
        for (yy, xx), val in np.ndenumerate(obj.mask):
            if not val:
                continue
            ny = y1 + yy + self.dy
            nx = x1 + xx + self.dx
            if 0 <= ny < grid.pixels.shape[0] and 0 <= nx < grid.pixels.shape[1]:
                new_pixels[ny, nx] = obj.grid.pixels[yy, xx]

        new_grid = Grid(new_pixels)
        new_objs = extract_all_object_views(new_grid)["conn4_color"]  # uproszczona rekonstrukcja
        return new_grid, new_objs


class DropToContact(DSLOp):
    """
    Przesuwa obiekt w pionie lub poziomie aż do kontaktu z innym obiektem lub brzegiem.
    direction = +1 (w dół/prawo), -1 (w górę/lewo)
    """
    def __init__(self, obj_id, axis='y', direction=1):
        self.obj_id = obj_id
        self.axis = axis
        self.direction = direction

    def apply(self, grid, objects):
        obj = objects[self.obj_id]
        mask = obj.mask
        y1, y2, x1, x2 = obj.bbox

        max_steps = grid.pixels.shape[0] if self.axis == 'y' else grid.pixels.shape[1]
        step = 0

        while step < max_steps:
            step += 1
            collision = False
            for (yy, xx), val in np.ndenumerate(mask):
                if not val:
                    continue
                gy = y1 + yy + (step * self.direction if self.axis == 'y' else 0)
                gx = x1 + xx + (step * self.direction if self.axis == 'x' else 0)
                if not (0 <= gy < grid.pixels.shape[0] and 0 <= gx < grid.pixels.shape[1]):
                    collision = True
                    break
                if grid.pixels[gy, gx] != 0:
                    collision = True
                    break
            if collision:
                step -= 1
                break

        final_dx = step * self.direction if self.axis == 'x' else 0
        final_dy = step * self.direction if self.axis == 'y' else 0
        return Translate(self.obj_id, final_dx, final_dy).apply(grid, objects)


def run_program(grid: Grid, objects: List[GridObject], ops: List[DSLOp]) -> Tuple[Grid, List[GridObject]]:
    """
    Wykonuje sekwencję operacji DSL. Po każdej aktualizuje grid i obiekty.
    """
    for op in ops:
        grid, objects = op.apply(grid, objects)
    return grid, objects


### === HEURYSTYKI DETEKCJI OPERACJI === ###
# Heurystyki wykrywające możliwe operacje na podstawie różnic input ↔ output

def detect_translate(obj_in: GridObject, obj_out: GridObject, diff: dict, obj_id: int) -> Optional[DSLOp]:
    """
    Wykrywa przesunięcie obiektu (Translate), jeśli jedyną różnicą jest pozycja.
    """
    keys = set(diff.keys())
    allowed_keys = {"centroid", "bbox"}
    if keys.issubset(allowed_keys):
        dy = round(obj_out.centroid()[0] - obj_in.centroid()[0])
        dx = round(obj_out.centroid()[1] - obj_in.centroid()[1])
        return Translate(obj_id=obj_id, dx=dx, dy=dy)
    return None


class Recolor(DSLOp):
    """
    Zmienia kolor `from_color` na `to_color` w obrębie danego obiektu.
    """
    def __init__(self, obj_id, from_color, to_color):
        self.obj_id = obj_id
        self.from_color = from_color
        self.to_color = to_color

    def apply(self, grid, objects):
        obj = objects[self.obj_id]
        new_pixels = np.copy(grid.pixels)
        y1, y2, x1, x2 = obj.bbox
        mask = obj.mask
        patch = obj.grid.pixels

        for (yy, xx), val in np.ndenumerate(mask):
            if val and patch[yy, xx] == self.from_color:
                new_pixels[y1 + yy, x1 + xx] = self.to_color

        new_grid = Grid(new_pixels)
        new_objs = extract_all_object_views(new_grid)["conn4_color"]
        return new_grid, new_objs


def detect_recolor(obj_in: GridObject, obj_out: GridObject, diff: dict, obj_id: int) -> Optional[DSLOp]:
    """
    Wykrywa zamianę koloru głównego obiektu (Recolor), jeśli to jedyna różnica.
    """
    if set(diff.keys()) == {"main_color"}:
        from_color, to_color = diff["main_color"]
        return Recolor(obj_id=obj_id, from_color=from_color, to_color=to_color)
    return None


### === COMPARISON === ###
# statystyki input/output, detekcja rozmiaru kafla, dominujące kolory
from dataclasses import dataclass

@dataclass
class GridComparison:
    # Struktura przechowująca porównanie input i output grida
    # Używana jako pomoc dla heurystyk mask/tile i analizy transformacji

    input_shape: tuple                # (wysokość, szerokość) inputu
    output_shape: tuple               # (wysokość, szerokość) outputu

    input_hist: dict                  # histogram kolorów inputu {kolor: liczba}
    output_hist: dict                 # histogram kolorów outputu

    input_background_color: int       # kolor tła inputu (najczęstszy)
    input_dominant_color: int         # najczęstszy kolor ≠ tło
    input_rare_color: int             # najrzadszy kolor

    output_background_color: int      # kolor tła outputu (najczęstszy)
    output_dominant_color: int        # najczęstszy kolor ≠ tło

    input_area: int                   # liczba wszystkich pikseli inputu
    output_area: int                  # j.w. dla outputu

    is_tiling_n_to_n_squared: bool    # czy output = input tiled n×n
    tiling_repeat: tuple              # (ile razy pionowo, ile poziomo)
    tile_size: tuple                  # wymiary pojedynczego tile (jeśli zidentyfikowane)


def compare_grids(input_grid: Grid, output_grid: Grid) -> GridComparison:
    # Heurystyczne porównanie input/output — zbiera rozmiary, kolory, tła, wzory tilingu

    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()

    def hist(grid):
        vals, counts = np.unique(grid.pixels, return_counts=True)
        return dict(zip(map(int, vals), map(int, counts)))

    def most_common_except(bg, h):
        # najczęstszy kolor ≠ tło
        return max((c for c in h.items() if c[0] != bg), key=lambda x: x[1], default=(None, 0))[0]

    in_hist = hist(input_grid)
    out_hist = hist(output_grid)

    input_bg = max(in_hist.items(), key=lambda x: x[1])[0]
    output_bg = max(out_hist.items(), key=lambda x: x[1])[0]

    input_dom = most_common_except(input_bg, in_hist)
    input_rare = min(in_hist.items(), key=lambda x: x[1])[0]
    output_dom = most_common_except(output_bg, out_hist)

    # Tiling: czy output = input tiled n×n (np. 3x3)
    is_tiling = (oh == ih * ih and ow == iw * iw)

    return GridComparison(
        input_shape=(ih, iw),
        output_shape=(oh, ow),
        input_hist=in_hist,
        output_hist=out_hist,
        input_background_color=input_bg,
        input_dominant_color=input_dom,
        input_rare_color=input_rare,
        output_background_color=output_bg,
        output_dominant_color=output_dom,
        input_area=ih * iw,
        output_area=oh * ow,
        is_tiling_n_to_n_squared=is_tiling,
        tiling_repeat=(oh // ih, ow // iw) if ih != 0 and iw != 0 else (1, 1),
        tile_size=(ih, iw),
    )




### === MASK + TILE ENGINE === ###
# make_mask, make_tile, try_masked_patterns
# heurystyki oparte na maskach, stare podejście

def invert_grid(grid: Grid):
    # Zwraca grid, w którym kolory są "odwrócone" względem tła:
    # - Piksele ≠ 0 stają się zerem (czyli usunięcie obiektów),
    # - Piksele == 0 przyjmują wartość niezerową (kolor nadpisujący)
    # To operacja w stylu "zamień obiekt z tłem" – # HEUR
    nonzeros = grid.pixels[grid.pixels != 0]
    if len(nonzeros) == 0:
        return grid.copy()
    fill_color = int(nonzeros[0])  # np. 6
    new = np.where(grid.pixels == 0, fill_color, 0)
    return Grid(new)

def make_mask(input_grid: Grid, strategy: str, comparison) -> Grid:
    # Tworzy binarną maskę z grida, zgodnie z wybraną strategią
    px = input_grid.pixels
    if strategy == 'nonzero':
        return Grid((px != 0).astype(int))
    elif strategy == 'zero':
        return Grid((px == 0).astype(int))
    elif strategy == '==dominant':
        color = comparison.input_dominant_color
        return Grid((px == color).astype(int))
    elif strategy == '==rare':
        color = comparison.input_rare_color
        return Grid((px == color).astype(int))
    elif strategy.startswith('=='):
        k = int(strategy[2:])
        return Grid((px == k).astype(int))
    elif strategy.startswith('!='):
        k = int(strategy[2:])
        return Grid((px != k).astype(int))
    elif strategy.startswith('in:'):
        ks = list(map(int, strategy[3:].split(',')))
        mask = np.isin(px, ks).astype(int)
        return Grid(mask)
    else:
        raise ValueError(f"Unknown mask strategy: {strategy}")


def make_tile(input_grid: Grid, strategy: str) -> Grid:
    # Tworzy tile (płytkę) z inputu według podanej transformacji
    if strategy == 'orig':
        return input_grid.copy()
    elif strategy == 'invert':
        return invert_grid(input_grid)
    elif strategy == 'flip_x':
        return input_grid.flip('x')
    elif strategy == 'flip_y':
        return input_grid.flip('y')
    elif strategy == 'rot90':
        return input_grid.rotate(1)
    elif strategy == 'rot180':
        return input_grid.rotate(2)
    elif strategy == 'rot270':
        return input_grid.rotate(3)
    elif strategy.startswith('recolor:'):
        _, from_c, to_c = strategy.split(':')
        return input_grid.recolor(int(from_c), int(to_c))
    else:
        raise ValueError(f"Unknown tile strategy: {strategy}")

def try_masked_patterns(input_grid: Grid, output_grid: Grid, comparison) -> List:
    # Próbuj różnych (mask, tile) par + copy_if_1, by zrekonstruować output
    # Strategia: heurystyczne dopasowanie kaflowe, starego typu
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
    if oh % ih != 0 and ow % iw != 0:
        return []

    reps_y = oh // ih if oh % ih == 0 else 1
    reps_x = ow // iw if ow % iw == 0 else 1

    # Lista strategii maskowania
    MASK_STRATEGIES = ['nonzero', 'zero', '==dominant', '==rare']
    all_colors = list(comparison.input_hist.keys())
    for k in all_colors:
        MASK_STRATEGIES.append(f"=={k}")
        MASK_STRATEGIES.append(f"!={k}")
    if len(all_colors) >= 2:
        for i in range(len(all_colors)):
            for j in range(i + 1, len(all_colors)):
                MASK_STRATEGIES.append(f"in:{all_colors[i]},{all_colors[j]}")

    # Lista strategii tworzenia tile
    TILE_STRATEGIES = ['orig', 'invert', 'flip_x', 'flip_y', 'rot90', 'rot180', 'rot270']
    top_colors = sorted(comparison.input_hist.items(), key=lambda x: -x[1])[:2]
    if len(top_colors) >= 2:
        TILE_STRATEGIES.append(f"recolor:{top_colors[0][0]}:{top_colors[1][0]}")
        TILE_STRATEGIES.append(f"recolor:{top_colors[1][0]}:{top_colors[0][0]}")

    MODES = ['copy_if_1']  # tryb rozkładania (można rozszerzyć)

    valid_masks = []
    for mask_name in MASK_STRATEGIES:
        try:
            mask = make_mask(input_grid, mask_name, comparison)
        except Exception:
            continue

        reps_h, reps_w = mask.shape()
        if reps_h != reps_y or reps_w != reps_x:
            continue

        filled = expand_pattern_2d(mask, Grid(np.ones((ih, iw), dtype=int)), mode='copy_if_1')
        if np.count_nonzero(output_grid.pixels * filled.pixels) > 0:
            valid_masks.append((mask_name, mask))

    candidates = []
    for mask_name, mask in valid_masks:
        for tile_name in TILE_STRATEGIES:
            try:
                tile = make_tile(input_grid, tile_name)
            except Exception:
                continue

            for mode in MODES:
                result = expand_pattern_2d(mask, tile, mode=mode)
                if result.pixels.shape != output_grid.pixels.shape:
                    continue

                if np.array_equal(result.pixels, output_grid.pixels):
                    label = f"mask={mask_name} + tile={tile_name}"
                    candidates.append((result, label))

    return candidates


### === PATTERN2D === ###
# class Pattern2D, expand
# wzorce kaflowe składane z obiektów bazowych
class Pattern2D:
    def __init__(self, base: GridObject, transform_matrix):
        # Reprezentacja wzorca kaflowego 2D opartego na bazowym obiekcie
        # base: obiekt źródłowy (GridObject), który ma być powielany
        # transform_matrix: macierz np. 3x3, gdzie każdy element to lista transformacji (np. ['orig'], ['flip_y'])
        self.base = base
        self.transform_matrix = transform_matrix

    def expand(self):
        # Rozszerza wzorzec kaflowy przez aplikację transformacji z matrixa na base
        reps_y = len(self.transform_matrix)
        reps_x = len(self.transform_matrix[0])
        row_grids = []
        for y in range(reps_y):
            row_tiles = []
            for x in range(reps_x):
                tf = Transform(self.transform_matrix[y][x])
                obj = tf.apply(self.base)
                patch = obj.grid.pixels  # UWAGA: pełna siatka, bez cropowania
                row_tiles.append(patch)
            row = np.hstack(row_tiles)
            row_grids.append(row)
        result = np.vstack(row_grids)
        return Grid(result)


### === FALLBACK + STRATEGY === ###
# expand_pattern_2d, detect_fallback_mode, suggest_grid_filling_strategy
# awaryjne strategie dopasowania wzorca do outputu
def expand_pattern_2d(mask_grid: Grid, tile_grid: Grid, mode='copy_if_1') -> Grid:
    # Rozkłada kafelek (tile) według binarnej maski
    # mask_grid: grid o wymiarach (m,n), wartości 0 lub 1
    # tile_grid: grid kafla, np. (3,3)
    # mode: jeden z:
    #  - copy_if_1: kafel jeśli maska == 1
    #  - copy_if_0: kafel jeśli maska == 0
    #  - copy_flipped_if_1: odwrócony kafel jeśli maska == 1
    #  - copy_flipped_if_0: odwrócony kafel jeśli maska == 0

    reps_y, reps_x = mask_grid.shape()
    tile_h, tile_w = tile_grid.shape()
    result = np.zeros((reps_y * tile_h, reps_x * tile_w), dtype=int)
    flipped_tile = np.flip(tile_grid.pixels, axis=(0, 1))  # flip w obu osiach

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
    # Heurystyka próbująca wykryć tryb rozszerzenia inputu do outputu na podstawie liczby pikseli
    # Zakłada, że input był tiled z jakąś maską (nonzero) i może być kopiowany / odwracany
    # Zwraca jeden z trybów: 'copy_if_1', 'copy_if_0', 'copy_flipped_if_1', 'copy_flipped_if_0'
    # lub 'none' jeśli nie znaleziono nic pasującego – # HEUR, # TODO refaktoryzacja

    mask = (input_grid.pixels > 0).astype(int)
    tile_area = input_grid.shape()[0] * input_grid.shape()[1]
    out_count = np.count_nonzero(output_grid.pixels)
    mask_area = np.count_nonzero(mask)

    if out_count == mask_area * mask_area:
        return 'copy_if_1'
    if out_count == (tile_area - mask_area) * mask_area:
        return 'copy_if_0'
    if out_count == mask_area * (tile_area - mask_area):
        return 'copy_flipped_if_1'
    if out_count == (tile_area - mask_area) * (tile_area - mask_area):
        return 'copy_flipped_if_0'
    return 'none'

### === HEURYSTYKA: suggest_grid_filling_strategy === ###
def suggest_grid_filling_strategy(input_grid, output_grid):
    # Heurystyka: sprawdza, czy output da się wygenerować przez:
    # - nałożenie maski na input (nonzero / zero)
    # - transformację inputu (flip / rotacja)
    # - powielenie w siatce reps_y x reps_x
    # Jeśli tak — zwraca strategię jako dict, np.:
    # {'mode': 'mask_and_tile', 'mask_type': 'nonzero', 'tile_op': 'flip_x', 'score': 0.875}

    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()

    # Sprawdzenie czy output jest dokładnym wielokrotnością inputu
    if oh % ih != 0 or ow % iw != 0:
        return None

    reps_y = oh // ih
    reps_x = ow // iw

    # Możliwe transformacje kafla
    tile_transforms = ['orig', 'flip_x', 'flip_y', 'rotate_90', 'rotate_180', 'rotate_270']

    # Typy masek do sprawdzenia
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

            # Zastosuj transformację do inputu, jako kafel
            tile = Transform(tf).apply(GridObject(input_grid.pixels)).grid

            # Dla każdego miejsca w output sprawdź, czy pasuje spodziewany kafel
            for y in range(reps_y):
                for x in range(reps_x):
                    iy, ix = y * ih, x * iw
                    out_patch = output_grid.pixels[iy:iy+ih, ix:ix+iw]

                    # Jeśli maska == 1 → używamy kafla, inaczej → zera
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

    # Zwracamy najlepszy dopasowany wzorzec, jeśli osiąga sensowny próg
    if best_score > 0.8:
        return best_match

    return None




##FInal

def generate_candidate_programs(input_grid: Grid, output_grid: Grid, verbose=False):
    candidates = []
    comparison = compare_grids(input_grid, output_grid)
    if verbose:
        print("[compare] GridComparison:", comparison)
    if verbose:
        print(f"[generate] input: {input_grid.shape()}, output: {output_grid.shape()}")
    explanations = []

    # Heurystyka 1: globalna transformacja (flip, rotate)
    if comparison.input_shape == comparison.output_shape:
        if comparison.input_hist == comparison.output_hist:
            if verbose:
                print("[generate] Heurystyka 1 aktywna (rozmiar =, histogram =)")
            for tf in ['orig', 'flip_x', 'flip_y', 'rotate_90', 'rotate_180', 'rotate_270']:
                transformed = Transform(tf).apply(GridObject(input_grid.pixels)).grid
                candidates.append(
                    (Sequence([]) if tf == 'orig' else Transform(tf), f"global_transform: {tf}")
                )

    # Heurystyka 2: Pattern2D (powielanie kafli z transformacją)
    ih, iw = comparison.input_shape
    oh, ow = comparison.output_shape
    if oh % ih == 0 and ow % iw == 0:
        reps_y, reps_x = oh // ih, ow // iw
        scale = reps_y * reps_x
        if all((comparison.input_hist.get(k, 0) * scale >= comparison.output_hist.get(k, 0)) for k in comparison.input_hist):
            if verbose:
                print("[generate] Heurystyka 2 aktywna (Pattern2D możliwy):")
            matrix = []
            for y in range(reps_y):
                row = []
                for x in range(reps_x):
                    patch = Grid(output_grid.pixels[y*ih:(y+1)*ih, x*iw:(x+1)*iw])
                    tf = detect_transformation(input_grid, patch, verbose=verbose) or 'orig'
                    row.append(tf)
                matrix.append(row)
            candidates.append((Pattern2D(GridObject(input_grid.pixels), matrix), "pattern2d from grid tiling"))

    # Heurystyka 3: maska + kafel
    suggestion = suggest_grid_filling_strategy(input_grid, output_grid)
    if verbose:
        print(f"[generate] Heurystyka 3 wynik: {suggestion}")
    if suggestion:
        mask_type = suggestion['mask_type']
        tile_op = suggestion['tile_op']
        mask = (input_grid.pixels != 0).astype(int) if mask_type == 'nonzero' else (input_grid.pixels == 0).astype(int)
        tile = Transform(tile_op).apply(GridObject(input_grid.pixels)).grid
        mode = f"copy_if_{1 if mask_type == 'nonzero' else 0}"
        result = expand_pattern_2d(Grid(mask), tile, mode)
        candidates.append((result, f"mask_and_tile: {mask_type} + {tile_op}"))

    # Heurystyka 4: kafel = negacja inputu w miejscach input != 0
    if comparison.is_tiling_n_to_n_squared:
        if verbose:
            print("[generate] Heurystyka 4 (negacja kafla wg maski ≠ 0)")
        ih, iw = comparison.input_shape
        oh, ow = comparison.output_shape
        negated = invert_grid(input_grid).pixels
        empty = np.zeros((ih, iw), dtype=int)
        mask = input_grid.pixels != 0

        result = np.zeros((oh, ow), dtype=int)
        for y in range(ih):
            for x in range(iw):
                iy, ix = y * ih, x * iw
                patch = negated if mask[y, x] else empty
                result[iy:iy+ih, ix:ix+iw] = patch

        if np.array_equal(result, output_grid.pixels):
            candidates.append((Grid(result), "invert_tile_if_nonzero"))
    # Heurystyka 5: kafel wstawiany tam, gdzie input == najczęstszy kolor (bez "tła")
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
    reps_y, reps_x = oh // ih, ow // iw

    if (ih, iw) == (reps_y, reps_x):
        # Użyj najczęstszego koloru wprost
        hist = comparison.input_hist
        most_common_color = max(hist.items(), key=lambda x: x[1])[0]
        print("[generate] używam raw dominant_color =", most_common_color)

        mask = (input_grid.pixels == most_common_color).astype(int)
        tile = input_grid.copy()
        result = expand_pattern_2d(Grid(mask), tile, mode='copy_if_1')

        print("[generate] mask:\n", mask)
        print("[generate] wynik heurystyki 5:\n", result.pixels)
        print("[generate] oczekiwany output:\n", output_grid.pixels)
        print("[generate] równość:", np.array_equal(result.pixels, output_grid.pixels))

        if np.array_equal(result.pixels, output_grid.pixels):
            candidates.append((result, "mask==most_common_color + tile=orig"))

        # Heurystyka 6: kafel wstawiany tam, gdzie input == najrzadszy kolor (raw)
    ih, iw = input_grid.shape()
    oh, ow = output_grid.shape()
    reps_y, reps_x = oh // ih, ow // iw

    if (ih, iw) == (reps_y, reps_x):
        hist = comparison.input_hist
        least_common_color = min(hist.items(), key=lambda x: x[1])[0]
        print("[generate] używam raw rare_color =", least_common_color)

        mask = (input_grid.pixels == least_common_color).astype(int)
        tile = input_grid.copy()
        result = expand_pattern_2d(Grid(mask), tile, mode='copy_if_1')

        print("[generate] mask:\n", mask)
        print("[generate] wynik heurystyki 6:\n", result.pixels)
        print("[generate] oczekiwany output:\n", output_grid.pixels)
        print("[generate] równość:", np.array_equal(result.pixels, output_grid.pixels))

        if np.array_equal(result.pixels, output_grid.pixels):
            candidates.append((result, "mask==least_common_color + tile=orig"))

        # Heurystyka 7: mask+tile engine
    for result, label in try_masked_patterns(input_grid, output_grid, comparison):
        candidates.append((result, f"mask_tile_engine: {label}"))

    return candidates

def debug_task(task_id, dataset_path="./", verbose=False):
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

    candidates = generate_candidate_programs(input_grid, output_grid, verbose=verbose)
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
            ok, why = debug_task(task_id, dataset_path, verbose=(task_id == "zxczxcZXC"))
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

def get_all_task_ids_from_json(dataset_path="./"):
    challenges_file = os.path.join(dataset_path, "arc-agi_training_challenges.json")
    with open(challenges_file) as f:
        challenges = json.load(f)
    return list(challenges.keys())

if __name__ == "__main__":
    # debug_task("0692e18c", dataset_path="dane/", verbose=True)
    TASK_IDS = get_all_task_ids_from_json("dane/")
    debug_many_tasks(TASK_IDS, dataset_path="dane/")
    # TASK_IDS = [
    #     "00576224", "007bbfb7", "0c786b71", "3af2c5a8", "3c9b0459",
    #     "46442a0e", "48131b3c", "4c4377d9", "4e7e0eb9", "59341089",
    #     "5b6cbef5", "6150a2bd", "62c24649", "67a3c6ac", "67e8384a",
    #     "68b16354", "6d0aefbc", "6fa7a44f", "74dd1130", "7953d61e",
    #     "7fe24cdd", "833dafe3", "8be77c9e", "8d5021e8", "90347967",
    #     "9dfd6313", "a416b8f3", "a59b95c0", "bc4146bd", "c48954c1",
    #     "c9e6f938", "ccd554ac", "cf5fd0ad", "ed36ccf7", "ed98d772",
    #     "0692e18c" , "27f8ce4f" , "48f8583b"# ← nowe zadanie z heurystyką negującą kafel
    # ]
    # debug_many_tasks(TASK_IDS, dataset_path="dane/")


# if __name__ == "__main__":
#     # Przykładowy grid
#     raw = [
#         [0, 0, 1, 1, 0],
#         [0, 0, 1, 1, 0],
#         [2, 2, 0, 0, 3],
#         [2, 2, 0, 0, 3],
#         [0, 0, 0, 0, 0]
#     ]
#     grid = Grid(raw)

#     # Testuj ekstrakcję obiektów
#     objs = extract_objects(grid, mode='connectivity')
#     print(f"Wykryto {len(objs)} obiektów:")
#     for i, obj in enumerate(objs):
#         print(f"  {i+1}: {obj}")

#     # Debugowo pokaż
#     visualize_objects(grid, objs)