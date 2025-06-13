#!/usr/bin/env python3
"""
ARC-AGI System - Main Entry Point

This is the main entry point for the ARC-AGI system. It provides different
testing modes and automatic report generation.

Usage:
    python main.py [mode]
    
Modes:
    training    - Test on 10 training tasks (default)
    solved      - Test on all 26 solved tasks  
    full        - Test on full ARC dataset (1000 tasks)
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
import os
from arc_agi_system import Grid, IntelligentSolver

# Task sets
TRAINING_TASKS = [
    "00d62c1b", "31adaf00", "358ba94e", "1f85a75f", "007bbfb7",
    "0692e18c", "009d5c81", "0520fde7", "05f2a901", "0b148d64"
]

SOLVED_TASKS = [
    "00576224", "007bbfb7", "009d5c81", "00d62c1b", "0520fde7",
    "05f2a901", "0692e18c", "0b148d64", "1cf80156", "1f85a75f",
    "23b5c85d", "31adaf00", "358ba94e", "3c9b0459", "72ca375d",
    "a416b8f3", "a59b95c0", "a87f7484", "be94b721", "bf699163",
    "c8f0f002", "d037b0a7", "d06dbe63", "d9f24cd1", "dc0a314f",
    "e9afcf9a"
]

def load_arc_data(challenges_path: str, solutions_path: str):
    """Load ARC dataset"""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)
    
    return challenges, solutions

def test_single_task(challenges: dict, task_id: str, verbose: bool = False) -> tuple:
    """Test a single task and return results"""
    try:
        task_data = challenges[task_id]
        train_examples = task_data['train']
        
        solver = IntelligentSolver()
        solver.set_training_examples(train_examples)
        
        success_count = 0
        total_count = len(train_examples)
        details = []
        
        for i, example in enumerate(train_examples):
            input_grid = Grid(example['input'])
            target_output = Grid(example['output'])
            
            solution, attempts = solver.solve(input_grid, target_output, max_attempts=10)
            
            if solution is not None:
                success_count += 1
                status = "✅"
            else:
                status = "❌"
            
            if verbose:
                details.append(f"  Example {i+1}: {status}")
        
        return success_count, total_count, success_count == total_count, details
        
    except Exception as e:
        if verbose:
            print(f"Error in task {task_id}: {e}")
        return 0, 0, False, [f"Error: {e}"]

def print_task_lists():
    """Print available task sets"""
    print("\n📋 AVAILABLE TASK SETS:")
    print("=" * 50)
    
    print(f"\n🎯 TRAINING TASKS ({len(TRAINING_TASKS)}):")
    for i, task_id in enumerate(TRAINING_TASKS, 1):
        print(f"  {i:2d}. {task_id}")
    
    print(f"\n✅ SOLVED TASKS ({len(SOLVED_TASKS)}):")
    for i, task_id in enumerate(SOLVED_TASKS, 1):
        print(f"  {i:2d}. {task_id}")
    
    print(f"\n🌐 FULL DATASET: 1000 tasks total")
    print("=" * 50)

def run_test_mode(mode: str, verbose: bool = True) -> dict:
    """Run test in specified mode"""
    
    # Determine task set
    if mode == "training":
        task_set = TRAINING_TASKS
        description = "10 Training Tasks"
    elif mode == "solved":
        task_set = SOLVED_TASKS
        description = "26 Solved Tasks"
    elif mode == "full":
        task_set = None  # Will load all tasks
        description = "Full ARC Dataset"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Load data
    base_dir = Path(__file__).parent
    challenges_path = base_dir / "dane" / "arc-agi_training_challenges.json"
    solutions_path = base_dir / "dane" / "arc-agi_training_solutions.json"
    
    if not challenges_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {challenges_path}")
    if not solutions_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {solutions_path}")
    
    challenges, solutions = load_arc_data(str(challenges_path), str(solutions_path))
    
    if task_set is None:
        task_set = list(challenges.keys())
    
    # Run tests
    print(f"🚀 ARC-AGI System Test - {description}")
    print("=" * 60)
    print(f"Testing {len(task_set)} tasks...")
    
    start_time = time.time()
    
    solved_tasks = []
    partially_solved = []
    failed_tasks = []
    total_examples_solved = 0
    total_examples = 0
    
    for i, task_id in enumerate(task_set):
        if len(task_set) > 50 and i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(task_set)} ({i/len(task_set)*100:.1f}%) - {elapsed:.1f}s")
        
        success, total, fully_solved, details = test_single_task(challenges, task_id, verbose and len(task_set) <= 50)
        
        total_examples_solved += success
        total_examples += total
        
        if fully_solved:
            solved_tasks.append(task_id)
            if verbose and len(task_set) <= 50:
                print(f"✅ {task_id}: {success}/{total} (100%)")
        elif success > 0:
            partially_solved.append((task_id, success, total))
            if verbose and len(task_set) <= 50:
                print(f"⚠️  {task_id}: {success}/{total} ({success/total*100:.1f}%)")
        else:
            failed_tasks.append(task_id)
            if verbose and len(task_set) <= 50:
                print(f"❌ {task_id}: 0/{total} (0%)")
    
    elapsed_total = time.time() - start_time
    
    # Results
    results = {
        'mode': mode,
        'description': description,
        'total_tasks': len(task_set),
        'solved_tasks': solved_tasks,
        'partially_solved': partially_solved,
        'failed_tasks': failed_tasks,
        'total_examples_solved': total_examples_solved,
        'total_examples': total_examples,
        'execution_time': elapsed_total,
        'timestamp': datetime.now().isoformat()
    }
    
    return results

def print_summary(results: dict):
    """Print short summary to console"""
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print(f"Mode: {results['description']}")
    print(f"Tasks fully solved: {len(results['solved_tasks'])}/{results['total_tasks']} ({len(results['solved_tasks'])/results['total_tasks']*100:.2f}%)")
    print(f"Examples solved: {results['total_examples_solved']}/{results['total_examples']} ({results['total_examples_solved']/results['total_examples']*100:.2f}%)")
    print(f"Execution time: {results['execution_time']:.1f}s")
    print("=" * 60)

def save_detailed_report(results: dict):
    """Save detailed report to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arc_test_report_{results['mode']}_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("ARC-AGI System Test Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Mode: {results['description']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Execution Time: {results['execution_time']:.2f} seconds\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Tasks: {results['total_tasks']}\n")
        f.write(f"Fully Solved: {len(results['solved_tasks'])} ({len(results['solved_tasks'])/results['total_tasks']*100:.2f}%)\n")
        f.write(f"Partially Solved: {len(results['partially_solved'])}\n")
        f.write(f"Failed: {len(results['failed_tasks'])}\n")
        f.write(f"Total Examples: {results['total_examples']}\n")
        f.write(f"Examples Solved: {results['total_examples_solved']} ({results['total_examples_solved']/results['total_examples']*100:.2f}%)\n\n")
        
        if results['solved_tasks']:
            f.write(f"FULLY SOLVED TASKS ({len(results['solved_tasks'])})\n")
            f.write("-" * 30 + "\n")
            for i, task_id in enumerate(results['solved_tasks'], 1):
                f.write(f"{i:3d}. {task_id}\n")
            f.write("\n")
        
        if results['partially_solved']:
            f.write(f"PARTIALLY SOLVED TASKS ({len(results['partially_solved'])})\n")
            f.write("-" * 35 + "\n")
            for i, (task_id, success, total) in enumerate(results['partially_solved'], 1):
                f.write(f"{i:3d}. {task_id}: {success}/{total} ({success/total*100:.1f}%)\n")
            f.write("\n")
        
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 20 + "\n")
        f.write("ARC-AGI System v1.0\n")
        f.write("10 DSL Operations:\n")
        f.write("- FillEnclosedAreas\n")
        f.write("- FillLargestSquares\n")
        f.write("- SelectObjectWithDifferentHoleCount\n")
        f.write("- ExtractSmallestObject\n")
        f.write("- GridSelfTiling\n")
        f.write("- GridInvertedSelfTiling\n")
        f.write("- ShapeToColorMapping\n")
        f.write("- SubgridIntersection\n")
        f.write("- ObjectGravitationalMove\n")
        f.write("- ExtractObjectByUniqueColor\n")
    
    print(f"📄 Detailed report saved: {filename}")
    return filename

def main():
    """Main entry point"""
    # Parse arguments
    mode = "training"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    # Special case for listing tasks
    if mode == "list":
        print_task_lists()
        return
    
    if mode not in ["training", "solved", "full"]:
        print("Usage: python main.py [training|solved|full|list]")
        print("  training - Test on 10 training tasks (default)")
        print("  solved   - Test on 26 solved tasks")
        print("  full     - Test on full ARC dataset")
        print("  list     - Show all task lists")
        sys.exit(1)
    
    try:
        # Run test
        results = run_test_mode(mode, verbose=(mode != "full"))
        
        # Print summary
        print_summary(results)
        
        # Save detailed report
        report_file = save_detailed_report(results)
        
        print(f"\n🎉 Test completed successfully!")
        print(f"📊 Quick stats: {len(results['solved_tasks'])}/{results['total_tasks']} tasks solved")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

