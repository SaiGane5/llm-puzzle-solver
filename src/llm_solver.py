# src/llm_solver.py
import os
import json
import time
import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from schema import Problem, Solution, Transition

class LLMSolver:
    """Base class for LLM-based solvers."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the LLM solver."""
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize client based on model type
        if model_name.startswith("gemini"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model_name)
            except ImportError:
                logging.error("Failed to import google.generativeai. Please install it with: pip install google-generativeai")
                self.client = None
        else:
            logging.warning(f"Unsupported model: {model_name}. Using a mock implementation.")
            self.client = None
    
    def format_puzzle(self, problem: Problem) -> str:
        """Format a puzzle for inclusion in prompts."""
        puzzle_str = f"Initial string: {problem.initial_string}\n"
        puzzle_str += "Available transitions:\n"
        
        for i, transition in enumerate(problem.transitions):
            src_str = f'"{transition.src}"' if transition.src else '""'
            tgt_str = f'"{transition.tgt}"' if transition.tgt else '""'
            puzzle_str += f"{i}. {src_str} -> {tgt_str}\n"
            
        return puzzle_str
    
    def parse_solution(self, response: str) -> List[int]:
        """Parse the solution from the LLM's response."""
        # Look for sequences of indices like [0, 1, 2]
        pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        matches = re.search(pattern, response)
        
        if matches:
            solution_str = matches.group(1)
            return [int(idx.strip()) for idx in solution_str.split(',')]
        
        # Look for numbered steps like "Step 1: Apply transition 2"
        step_pattern = r'(?:step|transition)\s+(\d+)'
        steps = re.findall(step_pattern, response.lower())
        if steps:
            return [int(step) for step in steps]
        
        # As a last resort, extract all numbers
        nums = re.findall(r'\b(\d+)\b', response)
        if nums:
            return [int(num) for num in nums if int(num) < 100]  # Filter out unlikely large indices
            
        return []
    
    def solve(self, problem: Problem) -> Optional[Solution]:
        """
        Solve the puzzle using the model-specific implementation.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

class ZeroShotSolver(LLMSolver):
    """LLM solver using zero-shot prompting."""

    def solve(self, problem: Problem) -> Optional[Solution]:
        """Solve the puzzle using zero-shot prompting."""
        if not self.client:
            return None

        puzzle_str = self.format_puzzle(problem)

        prompt = f"""
You are given a "sed puzzle". The goal is to transform the initial string into an empty string by applying a sequence of transitions. Each transition replaces all occurrences of a source pattern with a target pattern.

{puzzle_str}

Your task:
- Find the **minimal and correct sequence** of transition indices (from the list above) that, when applied in order, will turn the initial string into an empty string.
- **Do NOT simply list all indices or guess.**
- Only include the indices of transitions that are actually used in the solution, in the correct order.
- Simulate the process step by step and ensure the final string is empty.
- Output your answer as a Python list of indices, e.g., [0, 2, 1].
- **Do not output [0, 1, 2] unless that is truly the only correct solution.**

Answer:
"""
        try:
            if self.model_name.startswith("gemini"):
                response = self.client.generate_content(prompt)
                solution_indices = self.parse_solution(response.text)
            else:
                # Mock implementation for testing
                solution_indices = []

            if solution_indices:
                return Solution(
                    problem_id=problem.problem_id,
                    solution=solution_indices
                )
            else:
                logging.warning(f"Failed to parse solution for problem {problem.problem_id}")
                return None

        except Exception as e:
            logging.error(f"Error while solving problem {problem.problem_id}: {e}")
            return None

class FewShotSolver(LLMSolver):
    """LLM solver using few-shot prompting."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, examples: List[Dict] = None):
        """Initialize the few-shot solver with examples."""
        super().__init__(model_name, api_key)
        self.examples = examples or []
    
    def solve(self, problem: Problem) -> Optional[Solution]:
        """Solve the puzzle using few-shot prompting."""
        if not self.client:
            return None
            
        puzzle_str = self.format_puzzle(problem)
        
        # Build examples part of the prompt
        examples_str = ""
        for i, example in enumerate(self.examples):
            examples_str += f"\nExample {i+1}:\n"
            examples_str += f"Initial string: {example['initial_string']}\n"
            examples_str += "Available transitions:\n"
            
            for j, transition in enumerate(example["transitions"]):
                src = transition["src"]
                tgt = transition["tgt"]
                src_str = f'"{src}"' if src else '""'
                tgt_str = f'"{tgt}"' if tgt else '""'
                examples_str += f"{j}. {src_str} -> {tgt_str}\n"
                
            examples_str += f"Solution: {example['solution']}\n"
        
        prompt = f"""
You are solving a "sed puzzle". The goal is to transform the initial string into an empty string by applying transitions in sequence. Each transition replaces all occurrences of a source pattern with a target pattern.

Here are some solved examples:{examples_str}

Now, solve this puzzle:

{puzzle_str}

Instructions:
- Find a sequence of transition indices (from the list above) that, when applied in order, will turn the initial string into an empty string.
- Only use the provided transitions.
- Output your answer as a Python list of indices, e.g., [0, 2, 1].

Answer:
"""
        try:
            if self.model_name.startswith("gemini"):
                response = self.client.generate_content(prompt)
                solution_indices = self.parse_solution(response.text)
            else:
                solution_indices = []
                
            if solution_indices:
                return Solution(
                    problem_id=problem.problem_id,
                    solution=solution_indices
                )
            else:
                logging.warning(f"Failed to parse solution for problem {problem.problem_id}")
                return None
                
        except Exception as e:
            logging.error(f"Error while solving problem {problem.problem_id}: {e}")
            return None

class CoTSolver(LLMSolver):
    """LLM solver using Chain of Thought prompting."""
    
    def solve(self, problem: Problem) -> Optional[Solution]:
        """Solve the puzzle using Chain of Thought prompting."""
        if not self.client:
            return None
            
        puzzle_str = self.format_puzzle(problem)
        
        prompt = f"""
You are solving a "sed puzzle". The goal is to transform the initial string into an empty string by applying transitions in sequence. Each transition replaces all occurrences of a source pattern with a target pattern.

{puzzle_str}

Let's solve this step by step:
1. Start with the initial string.
2. At each step, examine the available transitions and choose one that can be applied.
3. Apply the transition and update the string.
4. Repeat until the string is empty.

After reasoning through the steps, output your answer as a Python list of transition indices, e.g., [0, 2, 1].

Show your reasoning, then provide the answer in the required format.

Answer:
"""
        try:
            if self.model_name.startswith("gemini"):
                response = self.client.generate_content(prompt)
                solution_indices = self.parse_solution(response.text)
            else:
                # Mock implementation for testing
                solution_indices = []
                
            if solution_indices:
                return Solution(
                    problem_id=problem.problem_id,
                    solution=solution_indices
                )
            else:
                logging.warning(f"Failed to parse solution for problem {problem.problem_id}")
                return None
                
        except Exception as e:
            logging.error(f"Error while solving problem {problem.problem_id}: {e}")
            return None

class CreativeSolver(LLMSolver):
    """Creative approach combining multiple techniques."""
    
    def solve(self, problem: Problem) -> Optional[Solution]:
        """
        Solve the puzzle using a creative combination of techniques:
        1. First attempts step-by-step simulation with intermediate states
        2. Falls back to pattern-matching heuristics if the first approach fails
        """
        if not self.client:
            return None
            
        puzzle_str = self.format_puzzle(problem)
        
        prompt = f"""
You are solving a "sed puzzle". The goal is to transform the initial string into an empty string by applying transitions in sequence. Each transition replaces all occurrences of a source pattern with a target pattern.

{puzzle_str}

Approach:
- Analyze which transitions can be applied to the current string.
- At each step, apply a transition and update the string.
- If a path doesn't work, backtrack and try another.
- Show the intermediate strings after each transition.

After completing your reasoning, output your answer as a Python list of transition indices, e.g., [0, 2, 1].

Answer:
"""
        try:
            if self.model_name.startswith("gemini"):
                response = self.client.generate_content(
                    prompt, 
                    generation_config={"temperature": 0.7, "max_output_tokens": 2048}
                )
                solution_indices = self.parse_solution(response.text)
            else:
                solution_indices = []
                
            if solution_indices:
                return Solution(
                    problem_id=problem.problem_id,
                    solution=solution_indices
                )
            else:
                logging.warning(f"Failed to parse solution for problem {problem.problem_id}")
                return None
                
        except Exception as e:
            logging.error(f"Error while solving problem {problem.problem_id}: {e}")
            return None


if __name__ == "__main__":
    from utils import read_problem_folder, write_solution_folder, validate_solutions

    print("=== LLM Puzzle Solver ===")
    model = input("Enter model name (default: gemini-pro): ").strip() or "gemini-pro"
    api_key = input("Enter API key (leave blank if not needed): ").strip() or None
    solver_type = input("Choose solver [zero-shot, few-shot, cot, creative] (default: zero-shot): ").strip() or "zero-shot"
    problems_path = input("Path to problems folder (default: data/dataset/hard): ").strip() or "data/dataset/hard"
    solutions_path = input("Path to save solutions (default: data/solutions/baseline/hard): ").strip() or "data/solutions/baseline/hard"

    # Read problems
    problems = read_problem_folder(Path(problems_path))

    # Choose solver
    if solver_type == "zero-shot":
        solver = ZeroShotSolver(model, api_key)
    elif solver_type == "few-shot":
        # You can load few-shot examples here if available
        solver = FewShotSolver(model, api_key, examples=[])
    elif solver_type == "cot":
        solver = CoTSolver(model, api_key)
    elif solver_type == "creative":
        solver = CreativeSolver(model, api_key)
    else:
        print("Unknown solver type. Exiting.")
        exit(1)

    # Solve problems
    solutions = {}
    for problem_id, problem in problems.items():
        logging.info(f"Solving problem {problem_id}...")
        solution = solver.solve(problem)
        if solution:
            solutions[problem_id] = solution
        else:
            logging.warning(f"No solution found for problem {problem_id}")

    # Write solutions
    write_solution_folder(solutions, Path(solutions_path))

    # Validate solutions
    validate_solutions(problems, solutions)