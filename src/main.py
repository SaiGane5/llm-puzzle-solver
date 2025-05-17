# src/main.py
import os
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from schema import Problem, Solution
from utils import read_problem_folder, read_solution_folder, write_problem_folder
from generator import PuzzleGenerator
from baseline import bfs
from llm_solver import ZeroShotSolver, FewShotSolver, CoTSolver, CreativeSolver
from evaluation import PuzzleEvaluator

def generate_dataset(args):
    """Generate a dataset of puzzles."""
    logging.info(f"Generating {args.num_puzzles} puzzles...")
    
    generator = PuzzleGenerator()
    puzzles = generator.generate_dataset(args.num_puzzles)
    
    # Write puzzles to the specified directory
    data_path = Path(args.output_dir)
    data_path.mkdir(exist_ok=True, parents=True)
    
    write_problem_folder(puzzles, data_path)
    logging.info(f"Generated {len(puzzles)} puzzles in {args.output_dir}")

def solve_puzzles(args):
    """Solve puzzles using the specified method."""
    logging.info(f"Solving puzzles using {args.method} method with {args.model} model...")
    
    # Load problems
    problems = read_problem_folder(Path(args.puzzle_dir))
    logging.info(f"Loaded {len(problems)} puzzles")
    
    # Create output directory
    output_path = Path(args.output_dir) / args.method
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize appropriate solver
    if args.method == 'baseline':
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
        if args.method == 'zero_shot':
            solver = ZeroShotSolver(model_name=args.model, api_key=args.api_key)
        elif args.method == 'few_shot':
            # Load examples for few-shot prompting
            examples = []
            if args.examples:
                with open(args.examples, 'r') as f:
                    examples = json.load(f)
            solver = FewShotSolver(model_name=args.model, api_key=args.api_key, examples=examples)
        elif args.method == 'cot':
            solver = CoTSolver(model_name=args.model, api_key=args.api_key)
        elif args.method == 'creative':
            solver = CreativeSolver(model_name=args.model, api_key=args.api_key)
        else:
            logging.error(f"Unsupported method: {args.method}")
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

def evaluate_solutions(args):
    """Evaluate puzzle solutions."""
    logging.info("Evaluating solutions...")
    
    # Load problems
    problems = read_problem_folder(Path(args.puzzle_dir))
    logging.info(f"Loaded {len(problems)} puzzles")
    
    # Create evaluator
    evaluator = PuzzleEvaluator()
    
    # Load solutions for each method
    solutions_by_method = {}
    for method in args.methods:
        method_path = Path(args.solution_dir) / method
        if method_path.exists():
            solutions = read_solution_folder(method_path)
            solutions_by_method[method] = solutions
            logging.info(f"Loaded {len(solutions)} solutions for method '{method}'")
        else:
            logging.warning(f"No solutions found for method '{method}'")
    
    # Compare solvers
    results = evaluator.compare_solvers(problems, solutions_by_method)
    
    # Create output directory
    output_path = Path(args.output_dir)
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
    parser = argparse.ArgumentParser(description='Sed puzzle solver')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate dataset
    gen_parser = subparsers.add_parser('generate', help='Generate dataset')
    gen_parser.add_argument('--num_puzzles', type=int, default=100, help='Number of puzzles to generate')
    gen_parser.add_argument('--output_dir', type=str, default='data/dataset', help='Output directory')
    
    # Solve puzzles
    solve_parser = subparsers.add_parser('solve', help='Solve puzzles')
    solve_parser.add_argument('--puzzle_dir', type=str, default='data/dataset', help='Puzzle directory')
    solve_parser.add_argument('--output_dir', type=str, default='data/solutions', help='Output directory')
    solve_parser.add_argument('--method', type=str, default='baseline', 
                             choices=['baseline', 'zero_shot', 'few_shot', 'cot', 'creative'],
                             help='Solution method')
    solve_parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='Model name')
    solve_parser.add_argument('--api_key', type=str, help='API key')
    solve_parser.add_argument('--examples', type=str, help='Examples file for few-shot prompting')
    
    # Evaluate solutions
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate solutions')
    eval_parser.add_argument('--puzzle_dir', type=str, default='data/dataset', help='Puzzle directory')
    eval_parser.add_argument('--solution_dir', type=str, default='data/solutions', help='Solution directory')
    eval_parser.add_argument('--output_dir', type=str, default='data/evaluation', help='Output directory')
    eval_parser.add_argument('--methods', nargs='+', 
                            default=['baseline', 'zero_shot', 'few_shot', 'cot', 'creative'],
                            help='Methods to evaluate')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Run the appropriate command
    if args.command == 'generate':
        generate_dataset(args)
    elif args.command == 'solve':
        solve_puzzles(args)
    elif args.command == 'evaluate':
        evaluate_solutions(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
