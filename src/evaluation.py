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