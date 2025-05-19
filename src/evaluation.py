# src/evaluation.py
import logging
import json
from typing import Dict, List, Optional, Any, DefaultDict
from pathlib import Path
from collections import defaultdict
from statistics import median
import numpy as np
from scipy.stats import pearsonr

from schema import Problem, Solution, Transition

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from utils import read_problem_folder, read_solution_folder
from schema import Problem, Solution, Transition


class PuzzleEvaluator:
    """Evaluator for LLM puzzle solving performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def apply_transition(self, string: str, transition: Transition) -> str:
        """Apply a transition to the current string."""
        pos = string.find(transition.src) if transition.src else 0
        if pos != -1:
            return string[:pos] + transition.tgt + string[pos + len(transition.src):]
        return string
    
    def validate_solution(self, problem: Problem, solution: Solution) -> Dict[str, Any]:
        """
        Validate a solution by applying each transition in sequence.
        Returns a dictionary with validation results.
        """
        if not solution or not solution.solution:
            return {"valid": False, "reason": "Empty solution", "steps": 0, "final_state": problem.initial_string}
        
        current = problem.initial_string
        steps = []
        
        for step_idx, transition_idx in enumerate(solution.solution):
            if transition_idx >= len(problem.transitions):
                return {
                    "valid": False, 
                    "reason": f"Invalid transition index {transition_idx}", 
                    "steps": step_idx, 
                    "final_state": current
                }
            
            transition = problem.transitions[transition_idx]
            prev_state = current
            current = self.apply_transition(current, transition)
            
            steps.append({
                "step": step_idx + 1,
                "transition_idx": transition_idx,
                "transition": str(transition),
                "before": prev_state,
                "after": current
            })
            
            # Early termination if we reach empty string
            if current == "":
                break
        
        valid = current == ""
        return {
            "valid": valid,
            "reason": "Success" if valid else "Failed to reach empty string",
            "steps": len(solution.solution),
            "final_state": current,
            "trace": steps
        }
    
    def evaluate_solver(self, problems: Dict[str, Problem], 
                       solutions: Dict[str, Solution]) -> Dict[str, Any]:
        """
        Evaluate the performance of a solver across multiple problems.
        Returns metrics about the solver's performance.
        """
        total = len(problems)
        valid_count = 0
        complexity_stats = []
        solution_lengths = []
        unsolved = []
        failure_reasons: DefaultDict[str, int] = defaultdict(int)
        transition_usage: DefaultDict[int, int] = defaultdict(int)
        
        results = {}
        
        for problem_id, problem in problems.items():
            if problem_id not in solutions:
                unsolved.append(problem_id)
                failure_reasons["Unsolved"] += 1
                continue
                
            solution = solutions[problem_id]
            validation = self.validate_solution(problem, solution)
            results[problem_id] = validation
            
            failure_reasons[validation["reason"]] += 1
            
            if validation["valid"]:
                valid_count += 1
                solution_lengths.append(validation["steps"])
                # Track transition usage
                for idx in solution.solution:
                    transition_usage[idx] += 1
                # Complexity stats
                complexity_stats.append({
                    "problem_id": problem_id,
                    "initial_length": len(problem.initial_string),
                    "transitions_count": len(problem.transitions),
                    "solution_length": validation["steps"],
                })
        
        # Calculate metrics
        success_rate = valid_count / total if total > 0 else 0
        avg_solution_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
        solution_length_stats = {
            "min": min(solution_lengths) if solution_lengths else 0,
            "max": max(solution_lengths) if solution_lengths else 0,
            "avg": avg_solution_length,
            "median": median(solution_lengths) if solution_lengths else 0,
        }
        
        return {
            "total_problems": total,
            "solved_problems": valid_count,
            "unsolved_problems": len(unsolved),
            "unsolved_ids": unsolved,
            "success_rate": success_rate,
            "solution_length_stats": solution_length_stats,
            "failure_reasons": dict(failure_reasons),
            "transition_usage": dict(transition_usage),
            "detailed_results": results,
            "complexity_stats": complexity_stats
        }
    
    def compare_solvers(self, problems: Dict[str, Problem], 
                      solutions_by_method: Dict[str, Dict[str, Solution]]) -> Dict[str, Any]:
        """
        Compare the performance of multiple solvers.
        Returns comparative metrics between solvers.
        """
        methods = list(solutions_by_method.keys())
        evaluations = {}
        
        # Evaluate each method
        for method, solutions in solutions_by_method.items():
            evaluations[method] = self.evaluate_solver(problems, solutions)
        
        # Generate comparative metrics
        comparative = {
            "methods": methods,
            "success_rates": {m: evaluations[m]["success_rate"] for m in methods},
            "solution_lengths": {m: evaluations[m]["solution_length_stats"] for m in methods},
            "failure_reasons": {m: evaluations[m]["failure_reasons"] for m in methods},
            "transition_usage": {m: evaluations[m]["transition_usage"] for m in methods},
        }
        
        # Step efficiency compared to baseline
        if "baseline" in methods:
            baseline_eval = evaluations["baseline"]
            baseline_results = {pid: res for pid, res in baseline_eval["detailed_results"].items() if res["valid"]}
            step_efficiency = {}
            for method in methods:
                if method == "baseline":
                    continue
                method_eval = evaluations[method]
                method_results = {pid: res for pid, res in method_eval["detailed_results"].items() if res["valid"]}
                ratios = []
                for pid in method_results:
                    if pid in baseline_results:
                        method_steps = method_results[pid]["steps"]
                        baseline_steps = baseline_results[pid]["steps"]
                        if baseline_steps > 0:
                            ratios.append(method_steps / baseline_steps)
                step_efficiency[method] = sum(ratios) / len(ratios) if ratios else 0
            comparative["step_efficiency"] = step_efficiency
        
        # Complexity correlations
        complexity_correlations = {}
        for method in methods:
            solved = []
            initial_lengths = []
            transitions_counts = []
            for pid, problem in problems.items():
                initial_lengths.append(len(problem.initial_string))
                transitions_counts.append(len(problem.transitions))
                solved.append(1 if evaluations[method]["detailed_results"].get(pid, {}).get("valid", False) else 0)
            # Calculate Pearson correlations
            corr_initial, _ = pearsonr(initial_lengths, solved) if len(solved) > 1 else (0, 0)
            corr_transitions, _ = pearsonr(transitions_counts, solved) if len(solved) > 1 else (0, 0)
            complexity_correlations[method] = {
                "initial_length": corr_initial,
                "transitions_count": corr_transitions
            }
        comparative["complexity_correlations"] = complexity_correlations
        
        # Unique solutions
        method_specific_solutions = {}
        for method in methods:
            solved_by_this = set(
                pid for pid, result in evaluations[method]["detailed_results"].items() 
                if result["valid"]
            )
            unique_to_method = solved_by_this.copy()
            for other_method in methods:
                if other_method != method:
                    solved_by_other = set(
                        pid for pid, result in evaluations[other_method]["detailed_results"].items() 
                        if result["valid"]
                    )
                    unique_to_method -= solved_by_other
            method_specific_solutions[method] = list(unique_to_method)
        comparative["unique_solutions"] = method_specific_solutions
        
        # Efficiency metrics
        complexity_by_method = {}
        for method in methods:
            complexity_stats = evaluations[method]["complexity_stats"]
            if complexity_stats:
                avg_efficiency = sum(
                    s["initial_length"] / s["solution_length"] if s["solution_length"] > 0 else 0
                    for s in complexity_stats
                ) / len(complexity_stats)
                complexity_by_method[method] = {"avg_efficiency": avg_efficiency}
        comparative["complexity_metrics"] = complexity_by_method
        
        return {
            "comparative": comparative,
            "detailed": evaluations
        }
    def plot_success_rates(self, comparative_results: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """Plot success rates for different solvers."""
        methods = comparative_results["methods"]
        success_rates = comparative_results["success_rates"]
        
        plt.figure(figsize=(10, 6))
        plt.bar(methods, [success_rates[m] for m in methods])
        plt.xlabel('Solvers')
        plt.ylabel('Success Rate')
        plt.title('Success Rates Across Solvers')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path / 'success_rates.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_solution_lengths(self, comparative_results: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """Plot solution length statistics for different solvers."""
        methods = comparative_results["methods"]
        solution_lengths = comparative_results["solution_lengths"]
        
        x = np.arange(len(methods))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot min, max, avg, median lengths
        ax.bar(x - width*1.5, [solution_lengths[m]["min"] for m in methods], width, label='Min')
        ax.bar(x - width/2, [solution_lengths[m]["avg"] for m in methods], width, label='Avg')
        ax.bar(x + width/2, [solution_lengths[m]["median"] for m in methods], width, label='Median')
        ax.bar(x + width*1.5, [solution_lengths[m]["max"] for m in methods], width, label='Max')
        
        ax.set_xlabel('Solvers')
        ax.set_ylabel('Solution Length')
        ax.set_title('Solution Length Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path / 'solution_lengths.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_complexity_correlations(self, comparative_results: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """Plot correlations between problem complexity and solver success."""
        methods = comparative_results["methods"]
        correlations = comparative_results["complexity_correlations"]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot correlations
        ax.bar(x - width/2, [correlations[m]["initial_length"] for m in methods], width, label='Initial Length')
        ax.bar(x + width/2, [correlations[m]["transitions_count"] for m in methods], width, label='Transitions Count')
        
        ax.set_xlabel('Solvers')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Correlation Between Problem Complexity and Success')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path / 'complexity_correlations.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def main():
    """Main function to execute the PuzzleEvaluator and generate plots."""
    print("=== Puzzle Solver Evaluation ===")
    
    # Get user inputs
    problems_path = input("Path to problems folder (default: data/dataset/hard): ").strip() or "data/dataset/hard"
    
    # Get solution paths for different methods
    solution_paths = {}
    method_count = int(input("How many solver methods to evaluate? ").strip() or "2")
    
    for i in range(method_count):
        method_name = input(f"Enter name for solver {i+1}: ").strip()
        method_path = input(f"Path to {method_name} solutions (default: data/solutions/{method_name}/hard): ").strip() or f"data/solutions/{method_name}/hard"
        solution_paths[method_name] = method_path
    
    output_path = input("Path to save evaluation results (default: data/evaluation): ").strip() or "data/evaluation"
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load problems
    logging.info("Loading problems...")
    problems = read_problem_folder(Path(problems_path))
    logging.info(f"Loaded {len(problems)} problems")
    
    # Load solutions for each method
    solutions_by_method = {}
    for method_name, method_path in solution_paths.items():
        logging.info(f"Loading solutions for {method_name}...")
        solutions = read_solution_folder(Path(method_path))
        logging.info(f"Loaded {len(solutions)} solutions for {method_name}")
        solutions_by_method[method_name] = solutions
    
    # Initialize evaluator
    evaluator = PuzzleEvaluator()
    
    # Perform evaluation
    logging.info("Evaluating solvers...")
    results = evaluator.compare_solvers(problems, solutions_by_method)
    
    # Generate and save plots
    logging.info("Generating plots...")
    evaluator.plot_success_rates(results["comparative"], output_dir)
    evaluator.plot_solution_lengths(results["comparative"], output_dir)
    evaluator.plot_complexity_correlations(results["comparative"], output_dir)
    
    # Save evaluation results
    logging.info("Saving evaluation results...")
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
