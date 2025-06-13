#!/usr/bin/env python3
"""
ARC Task Viewer Module

This module provides functionality to save ARC tasks as text files and PNG images.
It allows viewing any task from the full ARC dataset by its ID.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import os
from arc_agi_system import Grid

class TaskViewer:
    def __init__(self, challenges_path: str = None, solutions_path: str = None, output_dir: str = "task_views"):
        """Initialize TaskViewer with paths to ARC data and output directory"""
        if challenges_path is None:
            challenges_path = str(Path(__file__).parent / "dane" / "arc-agi_training_challenges.json")
        if solutions_path is None:
            solutions_path = str(Path(__file__).parent / "dane" / "arc-agi_training_solutions.json")
            
        self.challenges_path = challenges_path
        self.solutions_path = solutions_path
        self.output_dir = output_dir
        self.png_dir = os.path.join(output_dir, "png")
        self.text_dir = os.path.join(output_dir, "text")
        self.challenges = self._load_data(challenges_path)
        self.solutions = self._load_data(solutions_path)
        
        # Create output directories if they don't exist
        os.makedirs(self.png_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)
    
    def _load_data(self, path: str) -> Dict:
        """Load ARC dataset from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Nie znaleziono pliku z danymi: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Nieprawidłowy format pliku JSON: {path}")
    
    def get_task_ids(self) -> List[str]:
        """Get list of all available task IDs"""
        return list(self.challenges.keys())
    
    def _save_grid_as_text(self, grid: np.ndarray, filename: str) -> None:
        """Save a 2D grid to a text file"""
        if not grid.size:
            return
        
        # Get dimensions
        height, width = grid.shape
        
        # Create file path
        filepath = os.path.join(self.text_dir, filename)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write top border
            f.write("+" + "-" * width + "+\n")
            
            # Write grid content
            for row in grid:
                f.write("|")
                for cell in row:
                    f.write(str(cell))
                f.write("|\n")
            
            # Write bottom border
            f.write("+" + "-" * width + "+\n")
    
    def save_task(self, task_id: str) -> None:
        """Save a specific task with all its training and test examples to text files and PNG images"""
        if task_id not in self.challenges:
            print(f"❌ Zadanie o ID {task_id} nie istnieje w zbiorze danych")
            return
        
        task = self.challenges[task_id]
        print(f"\n🔍 Zapisuję zadanie {task_id}")
        
        # Save training examples
        print(f"Liczba przykładów treningowych: {len(task['train'])}")
        for i, example in enumerate(task['train'], 1):
            # Save input grid
            input_grid = Grid(np.array(example['input']))
            input_grid.plot(title=f"Input {i}", task_id=task_id, save=True)
            self._save_grid_as_text(input_grid.pixels, f"{task_id}_input_{i}.txt")
            
            # Save output grid
            output_grid = Grid(np.array(example['output']))
            output_grid.plot(title=f"Expected {i}", task_id=task_id, save=True)
            self._save_grid_as_text(output_grid.pixels, f"{task_id}_output_{i}.txt")
            
            print(f"✓ Zapisano przykład treningowy {i}")
        
        # Save test examples if they exist
        if 'test' in task:
            print(f"\nLiczba przykładów testowych: {len(task['test'])}")
            for i, example in enumerate(task['test'], 1):
                # Save input grid
                input_grid = Grid(np.array(example['input']))
                input_grid.plot(title=f"Input {i}", task_id=task_id, save=True)
                self._save_grid_as_text(input_grid.pixels, f"{task_id}_input_{i}.txt")
                
                # Save output grid if solution exists
                if task_id in self.solutions and i <= len(self.solutions[task_id]):
                    output_grid = Grid(np.array(self.solutions[task_id][i-1]))
                    output_grid.plot(title=f"Expected {i}", task_id=task_id, save=True)
                    self._save_grid_as_text(output_grid.pixels, f"{task_id}_output_{i}.txt")
                    print(f"✓ Zapisano przykład testowy {i} (z rozwiązaniem)")
                else:
                    print(f"✓ Zapisano przykład testowy {i} (bez rozwiązania)")

def main():
    """Example usage of TaskViewer"""
    import sys
    
    viewer = TaskViewer()
    
    # Get list of available tasks
    task_ids = viewer.get_task_ids()
    print(f"Dostępne zadania: {len(task_ids)}")
    
    # Get task ID from command line argument
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        viewer.save_task(task_id)
    else:
        print("\nUżycie: python task_viewer.py <task_id>")
        print("Przykład: python task_viewer.py 09629e4f")

if __name__ == "__main__":
    main() 