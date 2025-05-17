# src/evaluation.py
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

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
        
        results = {}
        
        for problem_id, problem in problems.items():
            if problem_id not in solutions:
                unsolved.append(problem_id)
                continue
                
            solution = solutions[problem_id]
            validation = self.validate_solution(problem, solution)
            results[problem_id] = validation
            
            if validation["valid"]:
                valid_count += 1
                solution_lengths.append(len(solution.solution))
                complexity_stats.append({
                    "problem_id": problem_id,
                    "initial_length": len(problem.initial_string),
                    "transitions_count": len(problem.transitions),
                    "solution_length": len(solution.solution),
                })
        
        # Calculate metrics
        success_rate = valid_count / total if total > 0 else 0
        avg_solution_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
        
        return {
            "total_problems": total,
            "solved_problems": valid_count,
            "unsolved_problems": len(unsolved),
            "unsolved_ids": unsolved,
            "success_rate": success_rate,
            "avg_solution_length": avg_solution_length,
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
            "avg_solution_lengths": {m: evaluations[m]["avg_solution_length"] for m in methods}
        }
        
        # Find problems solved by some methods but not others
        method_specific_solutions = {}
        for method in methods:
            solved_by_this = set(
                pid for pid, result in evaluations[method]["detailed_results"].items() 
                if result["valid"]
            )
            
            # Find problems solved by this method but not by others
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
        
        # Calculate complexity metrics
        complexity_by_method = {}
        for method in methods:
            complexity_stats = evaluations[method]["complexity_stats"]
            if complexity_stats:
                avg_efficiency = sum(
                    s["initial_length"] / s["solution_length"] if s["solution_length"] > 0 else 0
                    for s in complexity_stats
                ) / len(complexity_stats)
                
                complexity_by_method[method] = {
                    "avg_efficiency": avg_efficiency
                }
        
        comparative["complexity_metrics"] = complexity_by_method
        
        return {
            "comparative": comparative,
            "detailed": evaluations
        }
