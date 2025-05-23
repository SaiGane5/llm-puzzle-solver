# src/llm_solver.py
import os
import json
import time
import logging
import time
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from schema import Problem, Solution, Transition
from utils import read_problem_folder, write_solution_folder, validate_solutions
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
You are given a 'sed puzzle'. The goal is to transform the initial string into an empty string by applying a sequence of valid transitions. Each transition replaces all **non-overlapping** occurrences of a source pattern with a target pattern.

{puzzle_str}

Guidelines:
- Select the **minimum number of transitions** required to fully reduce the string to empty.
- Do **not hallucinate** steps or use transitions not listed.
- Simulate each step logically—**no guessing**.
- Output should be a Python list of transition indices in execution order, e.g., [0, 2, 1].

Answer:
"""

        try:
            if self.model_name.startswith("gemini"):
                response = self.client.generate_content(prompt)
                time.sleep(6)
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
You are solving a 'sed puzzle'. Your task is to transform a string into an empty string using transition rules.

Below are some **fully-solved examples**, including transition reasoning and final answer:

{examples_str}

Now solve the following puzzle:

{puzzle_str}

Instructions:
- Think step by step through the transitions, applying them logically.
- **Do not reuse transitions unnecessarily**. Favor minimal, valid solutions.
- Return only the indices of transitions used in order, e.g., [1, 0, 2].
- The final string **must be empty**.

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
You are solving a 'sed puzzle' using Chain-of-Thought reasoning.

{puzzle_str}

Process:
1. Begin with the initial string.
2. At each step, scan available transitions to see which one can be applied.
3. Apply the best transition and update the string.
4. If multiple paths are possible, **prefer the shortest complete path**.
5. Continue until the string becomes empty.

Reflect on your reasoning at each step before providing the final answer.

Final Output:
- Python list of transition indices in order of application, e.g., [1, 2, 0].
- Ensure only **actually applied** transitions are included.

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
You are solving a 'sed puzzle'. The objective is to reduce the given string to an empty string using the fewest transitions.

{puzzle_str}

Approach:
- Simulate the application of transitions **step by step**.
- After each step, print the intermediate string.
- If you reach a dead end, **backtrack and try an alternate path**.
- Avoid redundant transitions. Aim for minimal, correct solutions.

Output:
- A Python list of applied transition indices (e.g., [2, 0, 1]).
- Only list transitions that are actually used.
- Ensure the final string is empty.

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


def main():
    """Main function to execute the LLM solvers on the dataset."""
    
    print("=== LLM Puzzle Solver ===")
    
    # Get user inputs
    model = input("Enter model name (default: gemini-1.5-flash): ").strip() or "gemini-2.0-flash"
    api_key = input("Enter API key (leave blank if not needed): ").strip() or "AIzaSyBVCjp6cbCEuzWaJB81XHT2afHvX_6bAVI"
    solver_type = input("Choose solver [zero-shot, few-shot, cot, creative] (default: zero-shot): ").strip() or "zero-shot"
    problems_path = input("Path to problems folder (default: data/dataset/easy): ").strip() or "data/dataset/easy"
    solutions_path = input("Path to save solutions (default: data/solutions/zero_shot/easy): ").strip() or "data/solutions/zero_shot/easy"
    
    # Read problems
    logging.info("Loading problems...")
    problems = read_problem_folder(Path(problems_path))
    logging.info(f"Loaded {len(problems)} problems")
    
    # Choose solver
    if solver_type == "zero-shot":
        solver = ZeroShotSolver(model, api_key)
    elif solver_type == "few-shot":
        solver = FewShotSolver(model, api_key, examples=[])
    elif solver_type == "cot":
        solver = CoTSolver(model, api_key)
    elif solver_type == "creative":
        solver = CreativeSolver(model, api_key)
    else:
        logging.error("Unknown solver type. Exiting.")
        return
    
    # Solve problems
    solutions = {}
    for problem_id, problem in problems.items():
        logging.info("=====================================================")
        logging.info(f"Solving problem {problem_id}...")
        solution = solver.solve(problem)
        if solution:
            solutions[problem_id] = solution
            logging.info(f"Solution found for puzzle {problem_id}")
        else:
            logging.info(f"No solution found for problem {problem_id}")
    
    # Write solutions
    logging.info("Writing solutions...")
    write_solution_folder(solutions, Path(solutions_path))
    
    # Validate solutions
    logging.info("Validating solutions...")
    validate_solutions(problems, solutions)

if __name__ == "__main__":
    main()
