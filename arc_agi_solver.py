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

"""
🧩 ARC AGI SOLVER - PLAN WYKONANIA I STATUS

🎯 CEL NADRZĘDNY: Generalizacja na zadaniach ARC
System musi rozwiązywać nowe zadania na podstawie analizy wzorców, nie przez dopasowanie.

📊 STAN AKTUALNY (2024-12-19):
✅ GOTOWE:
- Grid, GridObject, extract_all_object_views()
- DSL: CropGrid, ResizeGridTo, Transform, Pattern2D
- Operacje obiektowe: Translate, MoveToBorder, MoveToTouch, Recolor, ResizeTo
- Heurystyki: Pattern2D, mask+tile, cut-out detection, object extraction
- LLM: **W PEŁNI ZINTEGROWANY** ✅
  - Endpoint: https://brighton-t-zen-postcard.trycloudflare.com/generate
  - Model: Qwen3-0.6B na Colab + FastAPI + Cloudflare Tunnel
  - Status: DZIAŁA - przetestowany i gotowy
- Narzędzia: task_viewer, debug_task, testy

❌ DO ZROBIENIA:
- Systematyczne porównywanie obiektów input↔output
- Zaawansowane heurystyki (przesuwanie, dodawanie obiektów)
- Beam search / planista symboliczny
- ~~Konfiguracja LLM API~~ ✅ **GOTOWE**
- Testy regresyjne

🚀 PLAN WYKONANIA:
FAZA 1 (1-2 dni): Dokończenie porównywania obiektów + rozbudowa DSL
FAZA 2 (2-3 dni): Zaawansowane heurystyki (przesuwanie, dodawanie obiektów)
FAZA 3 (GOTOWE): ~~Beam search + integracja LLM API~~
FAZA 4 (1-2 dni): Fallback + optymalizacja

🛠 NAD CZYM PRACUJEMY:
Status: LLM ZINTEGROWANY - Przejście do Fazy 1.1
- [x] Usunięto lokalny model LLM (za ciężki dla laptopa)
- [x] Stworzono API client (llm_api_client.py)
- [x] Zintegrowano z głównym solverem
- [x] **LLM API w pełni funkcjonalne** ✅
- [x] **Endpoint:** https://brighton-t-zen-postcard.trycloudflare.com/generate
- [x] **Model:** Qwen3-0.6B na Colab + FastAPI + Cloudflare Tunnel
- [x] **Testy przechodzą** - LLM odpowiada na zapytania
- [x] Fallback mechanism działa (solver bez LLM)

Następny krok: Faza 1.1 - Dokończenie porównywania obiektów z integracją LLM

📝 ZASADY:
1. Testy regresyjne po każdej zmianie
2. Iteracyjne podejście - małe kroki, częste testy
3. Generalizacja nad dopasowaniem
4. Wydajność (CPU-only, 12h limit Kaggle)

Pełny plan w ROADMAP.md
"""

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


import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import label
from collections import Counter
from scipy.optimize import linear_sum_assignment
import time

# Import LLM module
try:
    from llm_api_client import (
        LLMAPIClient, 
        TaskAnalysis, 
        ObjectDescription, 
        LLMResponse,
        grid_object_to_description,
        create_task_analysis
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM API client not available. LLM functionality will be disabled.")


### === GRID === ###
# podstawowe operacje na siatkach
class Grid:
    def __init__(self, pixels):
        # Reprezentuje pełną siatkę ARC (np. input lub output)
        # pixels: 2D numpy array (lub konwertowalny), typu int, kolory jako liczby
        self.pixels = np.array(pixels, dtype=int, copy=True)  # 🧼 wymuszamy pełną kopię danych

    def shape(self):
        # Zwraca krotkę (wysokość, szerokość)
        return self.pixels.shape

    def copy(self):
        # Tworzy nową kopię grida (deep copy)
        return Grid(np.array(self.pixels, copy=True))  # 🧼 bezpieczna kopia

    def flip(self, axis):
        # Zwraca nowy Grid po odbiciu:
        # axis = 'x' → flip w pionie (góra-dół),
        # axis = 'y' → flip w poziomie (lewo-prawo)
        if axis == 'x': return Grid(np.flipud(self.pixels.copy()))
        elif axis == 'y': return Grid(np.fliplr(self.pixels.copy()))
        else: raise ValueError("Invalid axis")

    def rotate(self, k=1):
        # Rotacja o 90 stopni * k razy (domyślnie jedna rotacja w lewo)
        return Grid(np.rot90(self.pixels.copy(), k))

    def recolor(self, from_color, to_color):
        # Zwraca grid, gdzie każdy piksel from_color jest zamieniony na to_color
        new = np.array(self.pixels, copy=True)
        new[new == from_color] = to_color
        return Grid(new)

    def crop(self, y1, y2, x1, x2):
        # Zwraca prostokątny wycinek grida (subgrid)
        # Współrzędne odpowiadają indeksowaniu NumPy: [y1:y2, x1:x2]
        return Grid(np.array(self.pixels[y1:y2, x1:x2], copy=True))  # 🧼 crop zawsze jako kopia

    def pad(self, pad_width, value=0):
        # Dodaje padding (ramkę) dookoła siatki, domyślnie wypełnioną zerami
        return Grid(np.pad(self.pixels, pad_width, constant_values=value))
    
    def plot(self, title="", task_id="", save=False):
        """
        Wyświetla lub zapisuje grid jako obrazek (PNG), z widoczną siatką komórek.
        """
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('tab20')
        norm = mcolors.BoundaryNorm(boundaries=np.arange(21) - 0.5, ncolors=20)

        # Wyświetlenie samego gridu
        ax.imshow(self.pixels, cmap=cmap, norm=norm)

        # Włączenie siatki (gridlines)
        h, w = self.pixels.shape
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title)

        if save:
            filename = f"task_views/png/{task_id}_{title.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def safe_pixels(self):
        return np.array(self.pixels, copy=True)  # 🧼 getter bez efektów ubocznych

### === OBJECT === ###
# reprezentacja wyodrębnionych obiektów, bbox, maska, centroid, kolory
class GridObject:
    def __init__(self, pixels, mask=None, global_position=(0, 0)):
        """
        Reprezentuje pojedynczy obiekt wykryty w gridzie ARC.
        - pixels: lokalny patch wycięty z grida
        - mask: binarna maska zaznaczająca piksele obiektu (względem patcha)
        - global_position: (y, x) pozycji lewego górnego rogu patcha w oryginalnym gridzie
        """
        self.grid = Grid(pixels)
        self.mask = np.array(mask, dtype=bool) if mask is not None else (self.grid.pixels > 0)
        self.global_position = global_position  # ← pozycja globalna: (top_y, left_x)

        self.bbox = self.compute_bbox()          # ← wyliczany bbox względem całej siatki
        self.color_hist = self.color_distribution()
        self.is_background = False

    def compute_bbox(self):
        """
        Oblicza bbox względem oryginalnego grida na podstawie lokalnej maski
        i globalnej pozycji patcha.
        """
        ys, xs = np.where(self.mask)
        if ys.size == 0:
            return (0, 0, 0, 0)
        top, left = self.global_position
        return (top + ys.min(), top + ys.max() + 1,
                left + xs.min(), left + xs.max() + 1)

    def color_distribution(self):
        """
        Histogram kolorów w obrębie maski.
        """
        vals, counts = np.unique(self.grid.pixels[self.mask], return_counts=True)
        return dict(zip(map(int, vals), map(int, counts)))

    def extract_patch(self):
        """
        Zwraca patch grida odpowiadający tylko obiektowi (pixels ograniczony do maski).
        """
        return self.grid

    def area(self):
        """
        Liczba aktywnych pikseli w masce (powierzchnia obiektu).
        """
        return np.sum(self.mask)

    def centroid(self):
        """
        Geometryczny środek masy obiektu (globalny).
        """
        ys, xs = np.where(self.mask)
        if len(ys) == 0:
            return (0, 0)
        y0, x0 = self.global_position
        return (float(np.mean(ys) + y0), float(np.mean(xs) + x0))

    def __repr__(self):
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
        # Liczba dziur (zera wewnątrz bboxa, poza maską)
        patch = self.grid.pixels
        hole_mask = (patch == 0) & (~self.mask)
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]])  # 4-sąsiedztwo
        num_holes = label(hole_mask.astype(int), structure=structure)[1]

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
            "num_holes": num_holes
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
                patch = grid.crop(y1, y2, x1, x2).pixels
                local_mask = mask[y1:y2, x1:x2]
                obj = GridObject(patch, mask=local_mask, global_position=(y1, x1))
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
            patch = grid.crop(y1, y2, x1, x2).pixels
            local_mask = component[y1:y2, x1:x2]
            obj =  GridObject(patch, mask=local_mask, global_position=(y1, x1))
            multicolor_objs.append(obj)
        _mark_background(multicolor_objs, bg_color, H, W)
        views[f"conn{conn}_multicolor"] = multicolor_objs

    return views

def detect_cut_out_candidate(task: dict) -> List["GridProgram"]:
    """
    Heurystyka wykrywająca czy output jest podsiatką inputu (cut-out).
    Jeśli tak, zwraca GridProgram([Crop(...), opcjonalnie ResizeTo(...)])
    Teraz z priorytetyzacją na podstawie cech obiektów w wycięciu.
    """
    programs = []

    for train_example in task.get("train", []):
        input_grid = Grid(train_example["input"])
        output_grid = Grid(train_example["output"])

        H_in, W_in = input_grid.shape()
        H_out, W_out = output_grid.shape()

        # Jeśli input nie jest większy od output, pomijamy
        if H_in < H_out or W_in < W_out:
            continue

        # Bruteforce: przesuwamy okno outputowego rozmiaru po input
        best_cutout = None
        best_score = -1
        
        for y in range(H_in - H_out + 1):
            for x in range(W_in - W_out + 1):
                subgrid = input_grid.crop(y, y + H_out, x, x + W_out)

                # Sprawdź czy to dokładne dopasowanie
                if np.array_equal(subgrid.pixels, output_grid.pixels):
                    # Oblicz score na podstawie cech obiektów w wycięciu
                    score = score_cutout_by_features((y, y + H_out, x, x + W_out), input_grid, output_grid)
                    
                    if score > best_score:
                        best_score = score
                        best_cutout = (y, y + H_out, x, x + W_out)
                        
                        crop_op = CropGrid(y, y + H_out, x, x + W_out)
                        prog = GridProgram([crop_op])
                        programs.append(prog)

                        # Jeśli rozmiar jest inny — dodajemy resize
                        if output_grid.shape() != (H_out, W_out):
                            resize_op = ResizeGridTo(*output_grid.shape())
                            prog = GridProgram([crop_op, resize_op])
                            programs.append(prog)

    return programs

def score_cutout_by_features(bbox: tuple, input_grid: Grid, output_grid: Grid) -> float:
    """
    Oblicza score wycięcia na podstawie cech obiektów w nim zawartych.
    Wyższy score = lepsze dopasowanie do oczekiwanego wzorca.
    """
    y1, y2, x1, x2 = bbox
    H_in, W_in = input_grid.shape()
    
    # Wyciągnij obiekty z wycięcia
    cutout_grid = input_grid.crop(y1, y2, x1, x2)
    cutout_objects = extract_all_object_views(cutout_grid)
    
    # Połącz wszystkie obiekty z różnych widoków
    all_objects = []
    for view_name, objects in cutout_objects.items():
        all_objects.extend(objects)
    
    if not all_objects:
        return 0.0
    
    scores = {}
    
    # 1. POŁOŻENIE - preferuj obiekty w skrajnych pozycjach
    scores['position'] = score_position_features(bbox, H_in, W_in)
    
    # 2. ROZMIAR - preferuj obiekty o skrajnych rozmiarach
    scores['size'] = score_size_features(all_objects)
    
    # 3. DZIURY - preferuj obiekty o skrajnej liczbie dziur
    scores['holes'] = score_holes_features(all_objects)
    
    # 4. KOLOR - preferuj obiekty o skrajnych cechach kolorystycznych
    scores['color'] = score_color_features(all_objects)
    
    # 5. IZOLACJA - preferuj obiekty oddalone od innych
    scores['isolation'] = score_isolation_features(bbox, input_grid)
    
    # Suma wszystkich score'ów
    total_score = sum(scores.values())
    
    return total_score

def score_position_features(bbox: tuple, grid_h: int, grid_w: int) -> float:
    """Score na podstawie położenia wycięcia w gridzie."""
    y1, y2, x1, x2 = bbox
    
    # Normalizuj pozycje do [0,1]
    center_y = (y1 + y2) / 2 / grid_h
    center_x = (x1 + x2) / 2 / grid_w
    
    # Preferuj skrajne pozycje (blisko 0 lub 1)
    score_y = max(center_y, 1 - center_y)  # Im dalej od środka, tym lepiej
    score_x = max(center_x, 1 - center_x)
    
    # Dodatkowy bonus za pozycje w rogach
    corner_bonus = 0.5 if (center_y > 0.8 or center_y < 0.2) and (center_x > 0.8 or center_x < 0.2) else 0
    
    return score_y + score_x + corner_bonus

def score_size_features(objects: List[GridObject]) -> float:
    """Score na podstawie rozmiaru obiektów."""
    if not objects:
        return 0.0
    
    areas = [obj.area() for obj in objects]
    max_area = max(areas)
    min_area = min(areas)
    avg_area = sum(areas) / len(areas)
    
    # Preferuj obiekty o skrajnych rozmiarach
    size_variance = max_area - min_area
    size_score = size_variance / max(avg_area, 1)
    
    return size_score

def score_holes_features(objects: List[GridObject]) -> float:
    """Score na podstawie liczby dziur w obiektach."""
    if not objects:
        return 0.0
    
    holes_counts = [obj.features()['num_holes'] for obj in objects]
    max_holes = max(holes_counts)
    min_holes = min(holes_counts)
    
    # Preferuj obiekty o skrajnej liczbie dziur
    holes_variance = max_holes - min_holes
    holes_score = holes_variance / max(max_holes, 1)
    
    return holes_score

def score_color_features(objects: List[GridObject]) -> float:
    """Score na podstawie cech kolorystycznych obiektów."""
    if not objects:
        return 0.0
    
    color_scores = []
    for obj in objects:
        features = obj.features()
        
        # Preferuj obiekty o skrajnych cechach kolorystycznych
        color_count = features['color_count']
        is_uniform = features['is_uniform_color']
        
        # Bonus za jednolity kolor lub za dużą różnorodność
        if is_uniform:
            color_scores.append(1.0)  # Jednolity kolor
        elif color_count > 3:
            color_scores.append(0.8)  # Duża różnorodność
        else:
            color_scores.append(0.3)  # Średnia różnorodność
    
    return max(color_scores) if color_scores else 0.0

def score_isolation_features(bbox: tuple, input_grid: Grid) -> float:
    """Score na podstawie izolacji obiektów w wycięciu."""
    y1, y2, x1, x2 = bbox
    H_in, W_in = input_grid.shape()
    
    # Sprawdź czy wycięcie jest oddalone od krawędzi
    margin_y = min(y1, H_in - y2) / H_in
    margin_x = min(x1, W_in - x2) / W_in
    
    # Preferuj wycięcia blisko krawędzi (izolowane)
    isolation_score = (1 - margin_y) + (1 - margin_x)
    
    return isolation_score

def _mark_background(objs, bg_color, grid_h, grid_w):
    # Prosta reguła: jeśli obiekt ma tylko kolor tła i dotyka krawędzi → uznaj za tło
    for obj in objs:
        if obj.color_hist.get(bg_color, 0) == obj.area():
            y1, y2, x1, x2 = obj.bbox
            if y1 == 0 or y2 == grid_h or x1 == 0 or x2 == grid_w:
                obj.is_background = True

def visualize_objects(grid: Grid, objects: List[GridObject]):
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

def match_object_to_input(output_obj: GridObject, input_objects: List[GridObject]) -> Optional[GridObject]:
    """
    Znajdź obiekt wejściowy najbardziej podobny do danego obiektu wyjściowego.
    Porównuje tylko cechy (na razie: area, num_holes, shape_type).
    """
    best_score = float("inf")
    best_match = None

    output_feat = output_obj.features()

    for input_obj in input_objects:
        input_feat = input_obj.features()
        score = 0

        # Kategoryczna cecha: typ kształtu
        if input_feat["shape_type"] != output_feat["shape_type"]:
            continue

        # Porównywalne cechy liczbowe
        score += abs(input_feat["area"] - output_feat["area"])
        score += abs(input_feat["num_holes"] - output_feat["num_holes"])

        if score < best_score:
            best_score = score
            best_match = input_obj

    return best_match


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
    def __init__(self, operations: List[Operation]):
        self.operations = operations

    def __repr__(self):
        return f"Sequence({', '.join(repr(op) for op in self.operations)})"

    def apply(self, obj: GridObject, grid_shape: Tuple[int, int]) -> Optional[GridObject]:
        current_obj = obj
        for op in self.operations:
            if current_obj is None:
                return None
            if isinstance(op, Operation):
                current_obj = op.apply(current_obj, grid_shape)
        return current_obj

class Crop:
    def __init__(self, y1, y2, x1, x2):
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def apply(self, grid: Grid) -> Grid:
        return grid.crop(self.y1, self.y2, self.x1, self.x2)

    def __repr__(self):
        return f"Crop(y1={self.y1}, y2={self.y2}, x1={self.x1}, x2={self.x2})"

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

class Translate(Operation):
    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    def __repr__(self):
        return f"Translate(dx={self.dx}, dy={self.dy})"

    def apply(self, obj: GridObject, grid_shape: Tuple[int, int]) -> Optional[GridObject]:
        H, W = grid_shape
        y1, x1 = obj.global_position
        ys, xs = np.where(obj.mask)

        canvas = np.zeros((H, W), dtype=int)
        mask_canvas = np.zeros((H, W), dtype=bool)

        print(f"\n🧪 Translate: dx={self.dx}, dy={self.dy}")
        for i in range(len(ys)):
            y_src = ys[i]
            x_src = xs[i]
            val = obj.grid.pixels[y_src, x_src]

            # Oblicz nową pozycję w gridzie globalnym
            y_target = y1 + y_src + self.dy
            x_target = x1 + x_src + self.dx

            if 0 <= y_target < H and 0 <= x_target < W:
                canvas[y_target, x_target] = val
                mask_canvas[y_target, x_target] = True
                print(f"  ↪ ({y_src},{x_src}) → ({y_target},{x_target}) = {val}")
            else:
                print(f"  ❌ OUT OF BOUNDS: ({y_target},{x_target})")

        # Wyciągnij patch i maskę
        ys_new, xs_new = np.where(mask_canvas)
        if len(ys_new) == 0:
            return None

        y1_, y2_ = ys_new.min(), ys_new.max() + 1
        x1_, x2_ = xs_new.min(), xs_new.max() + 1
        patch = canvas[y1_:y2_, x1_:x2_]
        local_mask = mask_canvas[y1_:y2_, x1_:x2_]

        return GridObject(patch, mask=local_mask, global_position=(y1_, x1_))
    
class MoveToBorder(Operation):
    def __init__(self, direction: str):
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("Invalid direction")
        self.direction = direction

    def __repr__(self):
        return f"MoveToBorder('{self.direction}')"

    def apply(self, obj: GridObject, grid_shape: Tuple[int, int]) -> Optional[GridObject]:
        """
        Przesuwa obiekt do wskazanej krawędzi gridu (up/down/left/right).
        """
        H, W = grid_shape
        y1, y2, x1, x2 = obj.bbox

        if self.direction == "up":
            return Translate(dx=0, dy=-y1).apply(obj, grid_shape)
        elif self.direction == "down":
            return Translate(dx=0, dy=H - y2).apply(obj, grid_shape)
        elif self.direction == "left":
            return Translate(dx=-x1, dy=0).apply(obj, grid_shape)
        elif self.direction == "right":
            return Translate(dx=W - x2, dy=0).apply(obj, grid_shape)

class MoveToTouch(Operation):
    def __init__(self, target_color: int, direction: str):
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("Invalid direction")
        self.target_color = target_color
        self.direction = direction

    def __repr__(self):
        return f"MoveToTouch(color={self.target_color}, dir='{self.direction}')"

    def apply(self, obj: GridObject, grid_shape: Tuple[int, int], full_grid: Optional[Grid] = None) -> Optional[GridObject]:
        if full_grid is None:
            raise ValueError("MoveToTouch requires access to the full input grid.")

        max_steps = grid_shape[0] + grid_shape[1]
        current_obj = obj

        for _ in range(max_steps):
            if self._touches_color(current_obj, self.target_color, grid_shape, full_grid):
                return current_obj

            dx, dy = {
                "up":    (0, -1),
                "down":  (0, 1),
                "left":  (-1, 0),
                "right": (1, 0),
            }[self.direction]

            current_obj = Translate(dx, dy).apply(current_obj, grid_shape)
            if current_obj is None:
                return None
        return None

    def _touches_color(self, obj: GridObject, target_color: int, grid_shape: Tuple[int, int], full_grid: Grid) -> bool:
        y1, x1 = obj.global_position
        ys, xs = np.where(obj.mask)

        for i in range(len(ys)):
            y_abs = y1 + ys[i]
            x_abs = x1 + xs[i]

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y_abs + dy, x_abs + dx
                if 0 <= ny < grid_shape[0] and 0 <= nx < grid_shape[1]:
                    if full_grid.pixels[ny, nx] == target_color:
                        return True
        return False

class CropFromInput(Operation):
    def __init__(self, y1: int, y2: int, x1: int, x2: int):
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def __repr__(self):
        return f"CropFromInput(y1={self.y1}, y2={self.y2}, x1={self.x1}, x2={self.x2})"

    def apply(self, input_grid: Grid, *_args, **_kwargs) -> Grid:
        """
        Zwraca wycięty fragment input grida jako nowy Grid.
        Nie wymaga GridObject.
        """
        return input_grid.crop(self.y1, self.y2, self.x1, self.x2)
    
class ResizeTo(Operation):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __repr__(self):
        return f"ResizeTo({self.height}, {self.width})"

    def apply(self, obj: GridObject, grid_shape: Tuple[int, int]) -> Optional[GridObject]:
        """
        Skalowanie obiektu (z zachowaniem jego maski) do nowego rozmiaru przez najprostszą interpolację.
        """
        from scipy.ndimage import zoom

        h, w = obj.mask.shape
        if h == 0 or w == 0:
            return None

        # Skaluje patch i maskę niezależnie (nearest dla maski)
        zoom_h = self.height / h
        zoom_w = self.width / w

        new_patch = zoom(obj.pixels, (zoom_h, zoom_w), order=0)  # nearest
        new_mask = zoom(obj.mask.astype(float), (zoom_h, zoom_w), order=0) > 0.5

        return GridObject(new_patch, mask=new_mask)

def run_program_on_objects(program: Operation, objects: List[GridObject], grid_shape: Tuple[int, int]) -> Grid:
    new_grid = np.zeros(grid_shape, dtype=int)

    for obj in objects:
        new_obj = program.apply(obj, grid_shape)
        if new_obj is None:
            continue

        y1, y2, x1, x2 = new_obj.bbox
        patch = new_obj.extract_patch().pixels
        mask = new_obj.mask

        # 🛠 Ręczne przypisanie – gwarancja poprawności
        for dy in range(y2 - y1):
            for dx in range(x2 - x1):
                if mask[dy, dx]:
                    new_grid[y1 + dy, x1 + dx] = patch[dy, dx]

    return Grid(new_grid)

# === GRID PROGRAM DSL (Heurystyki na całym gridzie) ===

class GridProgram:
    def __init__(self, operations: List):
        self.operations = operations

    def apply(self, grid: Grid) -> Optional[Grid]:
        current = grid.copy()  # 🧼 nigdy nie modyfikujemy inputa
        for op in self.operations:
            current = op.apply(current)
            if current is None:
                return None
        return current

    def __repr__(self):
        return "GridProgram[" + " >> ".join(op.__class__.__name__ for op in self.operations) + "]"


class CropGrid:
    def __init__(self, y1, y2, x1, x2):
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def apply(self, grid: Grid) -> Grid:
        return grid.crop(self.y1, self.y2, self.x1, self.x2)

    def __repr__(self):
        return f"Crop({self.y1}:{self.y2}, {self.x1}:{self.x2})"


class ResizeGridTo:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply(self, grid: Grid) -> Grid:
        from scipy.ndimage import zoom
        h, w = grid.shape()
        zoom_y = self.height / h
        zoom_x = self.width / w
        resized = zoom(grid.pixels, (zoom_y, zoom_x), order=0)
        return Grid(np.array(resized, dtype=int, copy=True))

    def __repr__(self):
        return f"ResizeTo({self.height}x{self.width})"

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

def try_masked_patterns(input_grid: Grid, output_grid: Grid, comparison, verbose=False) -> List:
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
                # Sprawdzenie czy wynik pasuje do oczekiwanego outputu
                if verbose:
                    print(f"[check] result.shape = {result.pixels.shape}, output.shape = {output_grid.pixels.shape}")
                    print(f"[check] result:\n{result.pixels}")
                    print(f"[check] output:\n{output_grid.pixels}")
                    print(f"[check] equal = {np.array_equal(result.pixels, output_grid.pixels)}")
                if np.array_equal(result.pixels.astype(int), output_grid.pixels.astype(int)):
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
        if verbose:
            print("[generate] używam raw dominant_color =", most_common_color)

        mask = (input_grid.pixels == most_common_color).astype(int)
        tile = input_grid.copy()
        result = expand_pattern_2d(Grid(mask), tile, mode='copy_if_1')

        if verbose:
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
        if verbose:
            print("[generate] używam raw rare_color =", least_common_color)

        mask = (input_grid.pixels == least_common_color).astype(int)
        tile = input_grid.copy()
        result = expand_pattern_2d(Grid(mask), tile, mode='copy_if_1')

        if verbose:
            print("[generate] mask:\n", mask)
            print("[generate] wynik heurystyki 6:\n", result.pixels)
            print("[generate] oczekiwany output:\n", output_grid.pixels)
            print("[generate] równość:", np.array_equal(result.pixels, output_grid.pixels))

        if np.array_equal(result.pixels, output_grid.pixels):
            candidates.append((result, "mask==least_common_color + tile=orig"))

        # Heurystyka 7: mask+tile engine
    for result, label in try_masked_patterns(input_grid, output_grid, comparison, verbose):
        candidates.append((result, f"mask_tile_engine: {label}"))

        # Heurystyka 8: cut-out detection
    # Heurystyka: cut-out detection
    cutout_task = {"train": [{"input": input_grid.pixels.tolist(), "output": output_grid.pixels.tolist()}]}
    cutout_programs = detect_cut_out_candidate(cutout_task)

    for program in cutout_programs:
        try:
            result = program.apply(input_grid)
            if np.array_equal(result.pixels, output_grid.pixels):
                candidates.append((program, "cut-out detection"))
        except Exception as e:
            if verbose:
                print(f"[cutout] Błąd wykonania {program}: {e}")

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

    # 🧠 LLM ANALYSIS - nowa heurystyka
    if verbose:
        print(f"\n🧠 LLM ANALYSIS for task {task_id}")
        print("=" * 50)
    
    llm_result = integrate_llm_analysis(task_id, input_grid, output_grid, verbose=verbose)
    if llm_result:
        success, explanation = llm_result
        if success:
            return True, f"LLM: {explanation}"

    # Dodatkowe debugowanie dla zadania 358ba94e
    if task_id == "358ba94e" and verbose:
        print("\n🔍 SZCZEGÓŁOWA ANALIZA ZADANIA 358ba94e")
        print("=" * 50)
        
        # Analiza obiektów w input
        input_views = extract_all_object_views(input_grid)
        print(f"\n📊 OBIEKTY W INPUT:")
        for view_name, objects in input_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}")
        
        # Analiza obiektów w output
        output_views = extract_all_object_views(output_grid)
        print(f"\n📊 OBIEKTY W OUTPUT:")
        for view_name, objects in output_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}")
        
        # Sprawdź czy output jest wycięciem z input
        print(f"\n🔍 SPRAWDZENIE CUT-OUT:")
        print(f"Input shape: {input_grid.shape()}")
        print(f"Output shape: {output_grid.shape()}")
        
        # Bruteforce sprawdzenie wszystkich możliwych wycięć
        ih, iw = input_grid.shape()
        oh, ow = output_grid.shape()
        
        if ih >= oh and iw >= ow:
            print(f"Output może być wycięciem z input")
            for y in range(ih - oh + 1):
                for x in range(iw - ow + 1):
                    subgrid = input_grid.crop(y, y + oh, x, x + ow)
                    if np.array_equal(subgrid.pixels, output_grid.pixels):
                        print(f"✅ ZNALEZIONO CUT-OUT: ({y}:{y+oh}, {x}:{x+ow})")
                        return True, f"cut-out at ({y}:{y+oh}, {x}:{x+ow})"
                    elif verbose:
                        diff = np.sum(subgrid.pixels != output_grid.pixels)
                        if diff <= 5:  # Pokaż bliskie dopasowania
                            print(f"  Blisko: diff={diff} @ ({y}:{y+oh}, {x}:{x+ow})")
        else:
            print(f"Output nie może być wycięciem z input (za duży)")

    # Dodatkowe debugowanie dla zadania 73ccf9c2
    if task_id == "73ccf9c2" and verbose:
        print("\n🔍 SZCZEGÓŁOWA ANALIZA ZADANIA 73ccf9c2")
        print("=" * 50)
        
        print(f"Input shape: {input_grid.shape()}")
        print(f"Output shape: {output_grid.shape()}")
        print(f"Input grid:\n{input_grid.pixels}")
        print(f"Output grid:\n{output_grid.pixels}")
        
        # Analiza obiektów w input
        input_views = extract_all_object_views(input_grid)
        print(f"\n📊 OBIEKTY W INPUT:")
        for view_name, objects in input_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}, "
                      f"shape_type={features['shape_type']}, touches_border={features['touches_border']}")
        
        # Analiza obiektów w output
        output_views = extract_all_object_views(output_grid)
        print(f"\n📊 OBIEKTY W OUTPUT:")
        for view_name, objects in output_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}, "
                      f"shape_type={features['shape_type']}, touches_border={features['touches_border']}")
        
        # Sprawdź czy output jest wycięciem z input
        print(f"\n🔍 SPRAWDZENIE CUT-OUT:")
        ih, iw = input_grid.shape()
        oh, ow = output_grid.shape()
        
        if ih >= oh and iw >= ow:
            print(f"Output może być wycięciem z input")
            best_cutout = None
            best_score = -1
            
            for y in range(ih - oh + 1):
                for x in range(iw - ow + 1):
                    subgrid = input_grid.crop(y, y + oh, x, x + ow)
                    if np.array_equal(subgrid.pixels, output_grid.pixels):
                        print(f"✅ ZNALEZIONO DOKŁADNE CUT-OUT: ({y}:{y+oh}, {x}:{x+ow})")
                        
                        # Oblicz score dla tego cut-out
                        score = score_cutout_by_features((y, y + oh, x, x + ow), input_grid, output_grid)
                        print(f"Score dla tego cut-out: {score}")
                        
                        if score > best_score:
                            best_score = score
                            best_cutout = (y, y + oh, x, x + ow)
                        
                        return True, f"cut-out at ({y}:{y+oh}, {x}:{x+ow}) with score {score}"
                    else:
                        # Pokaż różnice dla bliskich dopasowań
                        diff = np.sum(subgrid.pixels != output_grid.pixels)
                        if diff <= 10:  # Pokaż bliskie dopasowania
                            print(f"  Blisko: diff={diff} @ ({y}:{y+oh}, {x}:{x+ow})")
                            print(f"    Subgrid:\n{subgrid.pixels}")
                            print(f"    Expected:\n{output_grid.pixels}")
        else:
            print(f"Output nie może być wycięciem z input (za duży)")
        
        # Sprawdź cut-out detection heurystykę
        print(f"\n🔍 SPRAWDZENIE HEURYSTYKI CUT-OUT:")
        cutout_task = {"train": [{"input": input_grid.pixels.tolist(), "output": output_grid.pixels.tolist()}]}
        cutout_programs = detect_cut_out_candidate(cutout_task)
        print(f"Znalezione programy cut-out: {len(cutout_programs)}")
        for i, program in enumerate(cutout_programs):
            print(f"  Program {i+1}: {program}")
            try:
                result = program.apply(input_grid)
                if result is not None:
                    is_match = np.array_equal(result.pixels, output_grid.pixels)
                    print(f"    Wynik: {'✅ PASUJE' if is_match else '❌ NIE PASUJE'}")
                    if not is_match:
                        print(f"    Otrzymany:\n{result.pixels}")
                        print(f"    Oczekiwany:\n{output_grid.pixels}")
            except Exception as e:
                print(f"    Błąd wykonania: {e}")

    # Dodatkowe debugowanie dla zadania 7bb29440
    if task_id == "7bb29440" and verbose:
        print("\n🔍 SZCZEGÓŁOWA ANALIZA ZADANIA 7bb29440")
        print("=" * 50)
        
        print(f"Input shape: {input_grid.shape()}")
        print(f"Output shape: {output_grid.shape()}")
        print(f"Input grid:\n{input_grid.pixels}")
        print(f"Output grid:\n{output_grid.pixels}")
        
        # Analiza obiektów w input
        input_views = extract_all_object_views(input_grid)
        print(f"\n📊 OBIEKTY W INPUT:")
        for view_name, objects in input_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}, "
                      f"shape_type={features['shape_type']}, touches_border={features['touches_border']}")
        
        # Analiza obiektów w output
        output_views = extract_all_object_views(output_grid)
        print(f"\n📊 OBIEKTY W OUTPUT:")
        for view_name, objects in output_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}, "
                      f"shape_type={features['shape_type']}, touches_border={features['touches_border']}")
        
        # Analiza specyficzna dla tego zadania - szukanie prostokątów z dziurami
        print(f"\n🔍 ANALIZA PROSTOKĄTÓW Z DZIURAMI:")
        
        # Znajdź wszystkie prostokąty (obiekty o kształcie rectangle)
        rectangles = []
        for view_name, objects in input_views.items():
            for obj in objects:
                features = obj.features()
                if features['shape_type'] == 'rectangle':
                    rectangles.append((obj, features))
        
        print(f"Znalezione prostokąty: {len(rectangles)}")
        for i, (obj, features) in enumerate(rectangles):
            print(f"  Prostokąt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                  f"num_holes={features['num_holes']}, main_color={features['main_color']}")
            
            # Sprawdź czy ten prostokąt pasuje do output
            if obj.bbox[1] - obj.bbox[0] == output_grid.shape()[0] and obj.bbox[3] - obj.bbox[2] == output_grid.shape()[1]:
                print(f"    ⚠️  Rozmiar pasuje do output!")
                # Sprawdź czy zawartość pasuje
                y1, y2, x1, x2 = obj.bbox
                subgrid = input_grid.crop(y1, y2, x1, x2)
                diff = np.sum(subgrid.pixels != output_grid.pixels)
                print(f"    Różnica z output: {diff}")
                if diff == 0:
                    print(f"    ✅ DOKŁADNE DOPASOWANIE!")
                    return True, f"rectangle extraction at {obj.bbox}"
        
        # Sprawdź czy output jest wycięciem z input (standardowe sprawdzenie)
        print(f"\n🔍 SPRAWDZENIE CUT-OUT:")
        ih, iw = input_grid.shape()
        oh, ow = output_grid.shape()
        
        if ih >= oh and iw >= ow:
            print(f"Output może być wycięciem z input")
            for y in range(ih - oh + 1):
                for x in range(iw - ow + 1):
                    subgrid = input_grid.crop(y, y + oh, x, x + ow)
                    if np.array_equal(subgrid.pixels, output_grid.pixels):
                        print(f"✅ ZNALEZIONO DOKŁADNE CUT-OUT: ({y}:{y+oh}, {x}:{x+ow})")
                        return True, f"cut-out at ({y}:{y+oh}, {x}:{x+ow})"
                    else:
                        # Pokaż różnice dla bliskich dopasowań
                        diff = np.sum(subgrid.pixels != output_grid.pixels)
                        if diff <= 5:  # Pokaż bliskie dopasowania
                            print(f"  Blisko: diff={diff} @ ({y}:{y+oh}, {x}:{x+ow})")
        else:
            print(f"Output nie może być wycięciem z input (za duży)")

    # Dodatkowe debugowanie dla zadania 39a8645d
    if task_id == "39a8645d" and verbose:
        print("\n🔍 SZCZEGÓŁOWA ANALIZA ZADANIA 39a8645d")
        print("=" * 50)
        
        print(f"Input shape: {input_grid.shape()}")
        print(f"Output shape: {output_grid.shape()}")
        print(f"Input grid:\n{input_grid.pixels}")
        print(f"Output grid:\n{output_grid.pixels}")
        
        # Analiza obiektów w input
        input_views = extract_all_object_views(input_grid)
        print(f"\n📊 OBIEKTY W INPUT:")
        for view_name, objects in input_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}, "
                      f"shape_type={features['shape_type']}, touches_border={features['touches_border']}")
        
        # Analiza obiektów w output
        output_views = extract_all_object_views(output_grid)
        print(f"\n📊 OBIEKTY W OUTPUT:")
        for view_name, objects in output_views.items():
            print(f"\n{view_name}:")
            for i, obj in enumerate(objects):
                features = obj.features()
                print(f"  Obiekt {i+1}: bbox={obj.bbox}, area={features['area']}, "
                      f"num_holes={features['num_holes']}, main_color={features['main_color']}, "
                      f"shape_type={features['shape_type']}, touches_border={features['touches_border']}")
        
        # Analiza specyficzna dla tego zadania - szukanie najczęściej powtarzających się obiektów
        print(f"\n🔍 ANALIZA POWTARZAJĄCYCH SIĘ OBIEKTÓW:")
        
        # Zbierz wszystkie obiekty z różnych widoków
        all_objects = []
        for view_name, objects in input_views.items():
            all_objects.extend(objects)
        
        # Grupuj obiekty według podobieństwa (cechy)
        object_groups = group_similar_objects(all_objects)
        print(f"Znalezione grupy obiektów: {len(object_groups)}")
        
        for i, (group_features, objects) in enumerate(object_groups.items()):
            print(f"  Grupa {i+1}: {len(objects)} obiektów, cechy: {group_features}")
            
            # Sprawdź czy któryś obiekt z tej grupy pasuje do output
            for obj in objects:
                if obj.bbox[1] - obj.bbox[0] == output_grid.shape()[0] and obj.bbox[3] - obj.bbox[2] == output_grid.shape()[1]:
                    print(f"    ⚠️  Rozmiar pasuje do output!")
                    # Sprawdź czy zawartość pasuje
                    y1, y2, x1, x2 = obj.bbox
                    subgrid = input_grid.crop(y1, y2, x1, x2)
                    diff = np.sum(subgrid.pixels != output_grid.pixels)
                    print(f"    Różnica z output: {diff}")
                    if diff == 0:
                        print(f"    ✅ DOKŁADNE DOPASOWANIE!")
                        return True, f"most_frequent_object extraction at {obj.bbox}"
        
        # Sprawdź czy output jest wycięciem z input (standardowe sprawdzenie)
        print(f"\n🔍 SPRAWDZENIE CUT-OUT:")
        ih, iw = input_grid.shape()
        oh, ow = output_grid.shape()
        
        if ih >= oh and iw >= ow:
            print(f"Output może być wycięciem z input")
            for y in range(ih - oh + 1):
                for x in range(iw - ow + 1):
                    subgrid = input_grid.crop(y, y + oh, x, x + ow)
                    if np.array_equal(subgrid.pixels, output_grid.pixels):
                        print(f"✅ ZNALEZIONO DOKŁADNE CUT-OUT: ({y}:{y+oh}, {x}:{x+ow})")
                        return True, f"cut-out at ({y}:{y+oh}, {x}:{x+ow})"
                    else:
                        # Pokaż różnice dla bliskich dopasowań
                        diff = np.sum(subgrid.pixels != output_grid.pixels)
                        if diff <= 5:  # Pokaż bliskie dopasowania
                            print(f"  Blisko: diff={diff} @ ({y}:{y+oh}, {x}:{x+ow})")
        else:
            print(f"Output nie może być wycięciem z input (za duży)")

    # OGÓLNA HEURYSTYKA OBJECT EXTRACTION
    if verbose:
        print(f"\n🔍 OGÓLNA HEURYSTYKA OBJECT EXTRACTION:")
        
        # 1. Sprawdź czy output jest wycięciem z input (standardowe)
        ih, iw = input_grid.shape()
        oh, ow = output_grid.shape()
        
        if ih >= oh and iw >= ow:
            for y in range(ih - oh + 1):
                for x in range(iw - ow + 1):
                    subgrid = input_grid.crop(y, y + oh, x, x + ow)
                    if np.array_equal(subgrid.pixels, output_grid.pixels):
                        return True, f"cut-out at ({y}:{y+oh}, {x}:{x+ow})"
        
        # 2. Sprawdź object-based extraction
        result = try_object_extraction(input_grid, output_grid, verbose)
        if result:
            return result

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
            # Debugowanie dla zadań 358ba94e, 73ccf9c2 i 7bb29440
            verbose_debug = (task_id == "358ba94e" or task_id == "73ccf9c2" or task_id == "7bb29440" or task_id == "39a8645d")
            ok, why = debug_task(task_id, dataset_path, verbose=verbose_debug)
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

def group_similar_objects(objects: List[GridObject]) -> Dict[tuple, List[GridObject]]:
    """
    Grupuje obiekty według podobieństwa cech.
    Zwraca słownik: (cechy) -> lista podobnych obiektów
    """
    groups = {}
    
    for obj in objects:
        features = obj.features()
        # Klucz grupy: najważniejsze cechy dla porównania
        key = (
            features['shape_type'],
            features['main_color'],
            features['area'],
            features['num_holes']
        )
        
        if key not in groups:
            groups[key] = []
        groups[key].append(obj)
    
    return groups

def try_object_extraction(input_grid: Grid, output_grid: Grid, verbose=False) -> Optional[Tuple[bool, str]]:
    """
    Ogólna heurystyka object extraction - próbuje różne reguły wyciągania obiektów.
    """
    input_views = extract_all_object_views(input_grid)
    output_views = extract_all_object_views(output_grid)
    
    # Zbierz wszystkie obiekty z input
    all_input_objects = []
    for view_name, objects in input_views.items():
        all_input_objects.extend(objects)
    
    # Zbierz wszystkie obiekty z output
    all_output_objects = []
    for view_name, objects in output_views.items():
        all_output_objects.extend(objects)
    
    if verbose:
        print(f"  Analizuję {len(all_input_objects)} obiektów input vs {len(all_output_objects)} obiektów output")
    
    # Reguła 1: Najczęściej powtarzający się obiekt
    if len(all_output_objects) == 1:
        output_obj = all_output_objects[0]
        output_features = output_obj.features()
        
        # Grupuj obiekty input według podobieństwa
        object_groups = group_similar_objects(all_input_objects)
        
        # Znajdź grupę z największą liczbą obiektów
        largest_group = max(object_groups.items(), key=lambda x: len(x[1]))
        group_features, group_objects = largest_group
        
        if verbose:
            print(f"  Największa grupa: {len(group_objects)} obiektów z cechami {group_features}")
        
        # Sprawdź czy któryś obiekt z tej grupy pasuje do output
        for obj in group_objects:
            if obj.bbox[1] - obj.bbox[0] == output_grid.shape()[0] and obj.bbox[3] - obj.bbox[2] == output_grid.shape()[1]:
                y1, y2, x1, x2 = obj.bbox
                subgrid = input_grid.crop(y1, y2, x1, x2)
                if np.array_equal(subgrid.pixels, output_grid.pixels):
                    return True, f"most_frequent_object extraction at {obj.bbox}"
    
    # Reguła 2: Obiekt z najmniejszą liczbą dziur
    if len(all_output_objects) == 1:
        output_obj = all_output_objects[0]
        output_holes = output_obj.features()['num_holes']
        
        # Znajdź obiekt input z najmniejszą liczbą dziur
        min_holes_obj = min(all_input_objects, key=lambda obj: obj.features()['num_holes'])
        min_holes = min_holes_obj.features()['num_holes']
        
        if verbose:
            print(f"  Obiekt z najmniejszą liczbą dziur: {min_holes} dziur")
        
        if min_holes_obj.bbox[1] - min_holes_obj.bbox[0] == output_grid.shape()[0] and min_holes_obj.bbox[3] - min_holes_obj.bbox[2] == output_grid.shape()[1]:
            y1, y2, x1, x2 = min_holes_obj.bbox
            subgrid = input_grid.crop(y1, y2, x1, x2)
            if np.array_equal(subgrid.pixels, output_grid.pixels):
                return True, f"min_holes_object extraction at {min_holes_obj.bbox}"
    
    # Reguła 3: Obiekt o największym rozmiarze
    if len(all_output_objects) == 1:
        output_obj = all_output_objects[0]
        output_area = output_obj.features()['area']
        
        # Znajdź obiekt input o największym rozmiarze
        max_area_obj = max(all_input_objects, key=lambda obj: obj.features()['area'])
        max_area = max_area_obj.features()['area']
        
        if verbose:
            print(f"  Obiekt o największym rozmiarze: {max_area} pikseli")
        
        if max_area_obj.bbox[1] - max_area_obj.bbox[0] == output_grid.shape()[0] and max_area_obj.bbox[3] - max_area_obj.bbox[2] == output_grid.shape()[1]:
            y1, y2, x1, x2 = max_area_obj.bbox
            subgrid = input_grid.crop(y1, y2, x1, x2)
            if np.array_equal(subgrid.pixels, output_grid.pixels):
                return True, f"max_area_object extraction at {max_area_obj.bbox}"
    
    # Reguła 4: Obiekt o najmniejszym rozmiarze
    if len(all_output_objects) == 1:
        output_obj = all_output_objects[0]
        output_area = output_obj.features()['area']
        
        # Znajdź obiekt input o najmniejszym rozmiarze
        min_area_obj = min(all_input_objects, key=lambda obj: obj.features()['area'])
        min_area = min_area_obj.features()['area']
        
        if verbose:
            print(f"  Obiekt o najmniejszym rozmiarze: {min_area} pikseli")
        
        if min_area_obj.bbox[1] - min_area_obj.bbox[0] == output_grid.shape()[0] and min_area_obj.bbox[3] - min_area_obj.bbox[2] == output_grid.shape()[1]:
            y1, y2, x1, x2 = min_area_obj.bbox
            subgrid = input_grid.crop(y1, y2, x1, x2)
            if np.array_equal(subgrid.pixels, output_grid.pixels):
                return True, f"min_area_object extraction at {min_area_obj.bbox}"
    
    return None

def integrate_llm_analysis(task_id: str, input_grid: Grid, output_grid: Grid, verbose=False) -> Optional[Tuple[bool, str]]:
    """
    Integracja LLM z głównym solverem ARC.
    
    Args:
        task_id: ID zadania
        input_grid: Grid wejściowy
        output_grid: Grid wyjściowy
        verbose: Czy wyświetlać szczegółowe informacje
        
    Returns:
        Tuple[bool, str] lub None - (sukces, wyjaśnienie) lub None jeśli LLM nie jest dostępny
    """
    if not LLM_AVAILABLE:
        if verbose:
            print("⚠️  LLM not available, skipping LLM analysis")
        return None
    
    try:
        if verbose:
            print(f"\n🧠 LLM ANALYSIS for task {task_id}")
            print("=" * 50)
        
        # Initialize LLM interface
        llm = LLMAPIClient()
        if not llm.is_available:
            if verbose:
                print("⚠️  LLM API not available, skipping LLM analysis")
            return None
        
        # Extract objects from both grids
        input_views = extract_all_object_views(input_grid)
        output_views = extract_all_object_views(output_grid)
        
        # Use the most comprehensive view (conn8_multicolor)
        input_objects = input_views.get('conn8_multicolor', [])
        output_objects = output_views.get('conn8_multicolor', [])
        
        if verbose:
            print(f"📊 Extracted {len(input_objects)} input objects and {len(output_objects)} output objects")
        
        # Create task analysis for LLM
        task_analysis = create_task_analysis(
            task_id=task_id,
            input_grid=input_grid,
            output_grid=output_grid,
            input_objects=input_objects,
            output_objects=output_objects
        )
        
        if verbose:
            print(f"📋 Task analysis created: {task_analysis.input_shape} → {task_analysis.output_shape}")
        
        # Get LLM analysis
        start_time = time.time()
        llm_response = llm.analyze_task(task_analysis)
        analysis_time = time.time() - start_time
        
        if verbose:
            print(f"⏱️  LLM analysis time: {analysis_time:.2f}s")
            print(f"🎯 Strategy: {llm_response.strategy}")
            print(f"📊 Confidence: {llm_response.confidence}")
            print(f"💭 Reasoning: {llm_response.reasoning[:200]}...")
            print(f"🔧 Suggested operations: {llm_response.suggested_operations}")
        
        # If LLM suggests specific operations, try them
        if llm_response.suggested_operations and llm_response.confidence > 0.5:
            if verbose:
                print(f"\n🔧 Trying LLM-suggested operations...")
            
            for operation_desc in llm_response.suggested_operations:
                try:
                    # Parse and execute the suggested operation
                    result = execute_llm_operation(operation_desc, input_grid, output_grid)
                    if result:
                        success, explanation = result
                        if success:
                            if verbose:
                                print(f"✅ LLM operation successful: {explanation}")
                            return True, f"LLM: {explanation}"
                except Exception as e:
                    if verbose:
                        print(f"❌ LLM operation failed: {e}")
                    continue
        
        # If no direct operations worked, try LLM-generated programs
        if llm_response.new_operations and llm_response.confidence > 0.7:
            if verbose:
                print(f"\n🆕 Trying LLM-generated new operations...")
            
            for new_op in llm_response.new_operations:
                try:
                    result = try_llm_generated_operation(new_op, input_grid, output_grid)
                    if result:
                        success, explanation = result
                        if success:
                            if verbose:
                                print(f"✅ LLM-generated operation successful: {explanation}")
                            return True, f"LLM-generated: {explanation}"
                except Exception as e:
                    if verbose:
                        print(f"❌ LLM-generated operation failed: {e}")
                    continue
        
        if verbose:
            print(f"⚠️  LLM analysis completed but no solution found")
        
        return None
        
    except Exception as e:
        if verbose:
            print(f"❌ LLM analysis failed: {e}")
        return None

def execute_llm_operation(operation_desc: str, input_grid: Grid, output_grid: Grid) -> Optional[Tuple[bool, str]]:
    """
    Wykonuje operację sugerowaną przez LLM.
    
    Args:
        operation_desc: Opis operacji od LLM
        input_grid: Grid wejściowy
        output_grid: Grid wyjściowy
        
    Returns:
        Tuple[bool, str] lub None - (sukces, wyjaśnienie)
    """
    operation_desc = operation_desc.lower().strip()
    
    # Mapowanie prostych operacji
    if "crop" in operation_desc or "cut" in operation_desc:
        # Try different crop strategies
        ih, iw = input_grid.shape()
        oh, ow = output_grid.shape()
        
        if ih >= oh and iw >= ow:
            for y in range(ih - oh + 1):
                for x in range(iw - ow + 1):
                    subgrid = input_grid.crop(y, y + oh, x, x + ow)
                    if np.array_equal(subgrid.pixels, output_grid.pixels):
                        return True, f"crop({y}:{y+oh}, {x}:{x+ow})"
    
    elif "flip" in operation_desc:
        # Try different flips
        for flip_op in ['flip_x', 'flip_y']:
            flipped = input_grid.flip('x' if flip_op == 'flip_x' else 'y')
            if np.array_equal(flipped.pixels, output_grid.pixels):
                return True, flip_op
    
    elif "rotate" in operation_desc:
        # Try different rotations
        for k in [1, 2, 3]:  # 90, 180, 270 degrees
            rotated = input_grid.rotate(k)
            if np.array_equal(rotated.pixels, output_grid.pixels):
                return True, f"rotate_{k*90}"
    
    elif "recolor" in operation_desc or "color" in operation_desc:
        # Try color transformations
        comparison = compare_grids(input_grid, output_grid)
        for from_color in comparison.input_hist:
            for to_color in comparison.output_hist:
                if from_color != to_color:
                    recolored = input_grid.recolor(from_color, to_color)
                    if np.array_equal(recolored.pixels, output_grid.pixels):
                        return True, f"recolor({from_color}→{to_color})"
    
    return None

def try_llm_generated_operation(new_op: str, input_grid: Grid, output_grid: Grid) -> Optional[Tuple[bool, str]]:
    """
    Próbuje wykonać nową operację wygenerowaną przez LLM.
    
    Args:
        new_op: Nowa operacja od LLM
        input_grid: Grid wejściowy
        output_grid: Grid wyjściowy
        
    Returns:
        Tuple[bool, str] lub None - (sukces, wyjaśnienie)
    """
    # This is a placeholder for more sophisticated LLM-generated operations
    # In the future, this could parse and execute custom DSL operations
    return None

def _main():
    # # debug_task("0692e18c", dataset_path="dane/", verbose=True)
    # TASK_IDS = get_all_task_ids_from_json("dane/")
    # debug_many_tasks(TASK_IDS, dataset_path="dane/")
    TEST_TASK_IDS = [
        "00576224", "007bbfb7", "0692e18c", "0c786b71", "15696249",
        "27f8ce4f", "3af2c5a8", "3c9b0459", "46442a0e", "48131b3c",
        "48f8583b", "4c4377d9", "4e7e0eb9", "59341089", "5b6cbef5",
        "6150a2bd", "62c24649", "67a3c6ac", "67e8384a", "68b16354",
        "6d0aefbc", "6fa7a44f", "74dd1130", "7953d61e", "7fe24cdd",
        "833dafe3", "8be77c9e", "8d5021e8", "8e2edd66", "90347967",
        "9dfd6313", "a416b8f3", "a59b95c0", "ad7e01d0", "bc4146bd",
        "c3e719e8", "c48954c1", "c9e6f938", "ccd554ac", "cce03e0d",
        "cf5fd0ad", "ed36ccf7", "ed98d772",
        "358ba94e", "73ccf9c2", "7bb29440", "39a8645d"
    ]

    debug_many_tasks(TEST_TASK_IDS, dataset_path="dane/")


if __name__ == "__main__":
    _main()

