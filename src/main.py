import os
import logging
import json
from pathlib import Path
from typing import Dict, Optional

from schema import Problem, Solution
from utils import read_problem_folder, read_solution_folder, write_problem_folder
from generator import easy_generator

# from generator import EasySedPuzzleGenerator, MediumSedPuzzleGenerator, HardSedPuzzleGenerator
from baseline import bfs
from llm_solver import ZeroShotSolver, FewShotSolver, CoTSolver, CreativeSolver
from evaluation import PuzzleEvaluator

def generate_dataset():
    """Generate a dataset of puzzles based on user input."""
    num_puzzles = int(input("Enter the number of puzzles to generate: "))
    output_dir = input("Enter the output directory (default: data/dataset): ") or 'data/dataset'
    
    logging.info(f"Generating {num_puzzles} puzzles...")
    
    generator = easy_generator.EasySedPuzzleGenerator()
    puzzles = generator.generate_dataset(num_puzzles)
    
    # Write puzzles to the specified directory
    data_path = Path(output_dir)
    data_path.mkdir(exist_ok=True, parents=True)
    
    write_problem_folder(puzzles, data_path)
    logging.info(f"Generated {len(puzzles)} puzzles in {output_dir}")

def solve_puzzles():
    """Solve puzzles using the specified method based on user input."""
    puzzle_dir = input("Enter the puzzle directory (default: data/dataset): ") or 'data/dataset'
    output_dir = input("Enter the output directory (default: data/solutions): ") or 'data/solutions'
    method = input("Enter the solution method (baseline, zero_shot, few_shot, cot, creative; default: baseline): ") or 'baseline'
    model = input("Enter the model name (default: gemini-1.5-flash): ") or 'gemini-1.5-flash'
    api_key = input("Enter your API key: ")
    examples_file = input("Enter the examples file for few-shot prompting (optional): ")
    
    logging.info(f"Solving puzzles using {method} method with {model} model...")
    
    # Load problems
    problems = read_problem_folder(Path(puzzle_dir))
    logging.info(f"Loaded {len(problems)} puzzles")
    
    # Create output directory
    output_path = Path(output_dir) / method
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize appropriate solver
    if method == 'baseline':
        # Use the provided baseline solver
        solutions = {}
        for problem_id, problem in problems.items():
            logging.info(f"Solving problem {problem_id}...")
            solution = bfs(problem)
            if solution is not None:
                solutions[problem_id] = Solution(
                    problem_id=problem_id,
                    solution=solution
                )
                logging.info(f"Solution found for puzzle {problem_id}")
            else:
                logging.info(f"No solution found for puzzle {problem_id}")
    else:
        # Initialize LLM-based solver
        if method == 'zero_shot':
            solver = ZeroShotSolver(model_name=model, api_key=api_key)
        elif method == 'few_shot':
            # Load examples for few-shot prompting
            examples = []
            if examples_file:
                with open(examples_file, 'r') as f:
                    examples = json.load(f)
            solver = FewShotSolver(model_name=model, api_key=api_key, examples=examples)
        elif method == 'cot':
            solver = CoTSolver(model_name=model, api_key=api_key)
        elif method == 'creative':
            solver = CreativeSolver(model_name=model, api_key=api_key)
        else:
            logging.error(f"Unsupported method: {method}")
            return
        
        # Solve problems
        solutions = {}
        for problem_id, problem in problems.items():
            logging.info(f"Solving problem {problem_id}...")
            solution = solver.solve(problem)
            if solution:
                solutions[problem_id] = solution
                logging.info(f"Solution found for puzzle {problem_id}")
            else:
                logging.info(f"No solution found for puzzle {problem_id}")
    
    # Write solutions
    for problem_id, solution in solutions.items():
        solution_path = output_path / f"{problem_id}.json"
        with open(solution_path, 'w') as f:
            f.write(solution.model_dump_json())
    
    logging.info(f"Wrote {len(solutions)} solutions to {output_path}")

def evaluate_solutions():
    """Evaluate puzzle solutions based on user input."""
    puzzle_dir = input("Enter the puzzle directory (default: data/dataset): ") or 'data/dataset'
    solution_dir = input("Enter the solution directory (default: data/solutions): ") or 'data/solutions'
    output_dir = input("Enter the output directory (default: data/evaluation): ") or 'data/evaluation'
    methods_str = input("Enter the methods to evaluate (space-separated, default: baseline zero_shot few_shot cot creative): ") or 'baseline zero_shot few_shot cot creative'
    methods = methods_str.split()
    
    logging.info("Evaluating solutions...")
    
    # Load problems
    problems = read_problem_folder(Path(puzzle_dir))
    logging.info(f"Loaded {len(problems)} puzzles")
    
    # Create evaluator
    evaluator = PuzzleEvaluator()
    
    # Load solutions for each method
    solutions_by_method = {}
    for method in methods:
        method_path = Path(solution_dir) / method
        if method_path.exists():
            solutions = read_solution_folder(method_path)
            solutions_by_method[method] = solutions
            logging.info(f"Loaded {len(solutions)} solutions for method '{method}'")
        else:
            logging.warning(f"No solutions found for method '{method}'")
    
    # Compare solvers
    results = evaluator.compare_solvers(problems, solutions_by_method)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save results
    with open(output_path / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logging.info("Evaluation summary:")
    for method, rate in results["comparative"]["success_rates"].items():
        logging.info(f"  {method}: {rate*100:.2f}% success rate")
    
    logging.info(f"Detailed results saved to {output_path / 'evaluation_results.json'}")

def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    while True:
        print("\nChoose an action:")
        print("1. Generate dataset")
        print("2. Solve puzzles")
        print("3. Evaluate solutions")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            generate_dataset()
        elif choice == '2':
            solve_puzzles()
        elif choice == '3':
            evaluate_solutions()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
