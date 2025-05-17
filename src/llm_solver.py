# src/llm_solver.py
import os
import json
import time
import logging
import re
from typing import Dict, List, Tuple, Any, Optional

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
        I need to solve a "sed puzzle" where the goal is to transform the initial string to an empty string by applying the available transitions in sequence.
        
        {puzzle_str}
        
        Each transition can be applied by replacing the source pattern with the target pattern. For example, if there's a transition "ABC" -> "", applying it would remove "ABC" from the string.
        
        Please provide a sequence of transition indices that will transform the initial string to an empty string.
        Format your answer as a list of indices like [0, 1, 3, 2].
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
        I need to solve a "sed puzzle" where the goal is to transform the initial string to an empty string by applying the available transitions in sequence.
        
        Here are some examples:{examples_str}
        
        Now, solve this puzzle:
        
        {puzzle_str}
        
        Please provide a sequence of transition indices that will transform the initial string to an empty string.
        Format your answer as a list of indices like [0, 1, 3, 2].
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
        I need to solve a "sed puzzle" where the goal is to transform the initial string to an empty string by applying the available transitions in sequence.
        
        {puzzle_str}
        
        Let's solve this step by step:
        
        1. Start with the initial string: {problem.initial_string}
        2. For each step, I'll examine the available transitions and choose one that can be applied.
        3. I'll apply the transition and update the current string.
        4. I'll continue until I reach the empty string.
        
        Let me think through the solution carefully. I'll trace through how the string changes with each transition.
        
        After analyzing the puzzle, I'll provide the solution as a list of transition indices.
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
        I need to solve a "sed puzzle" where the goal is to transform the initial string to an empty string.
        
        {puzzle_str}
        
        I'll solve this systematically by following these steps:
        
        1. First, I'll analyze which transitions can be applied to the initial string.
        2. For each possible transition, I'll apply it and then recursively analyze the resulting string.
        3. I'll track my progress by showing how the string changes with each transition.
        4. If a particular path doesn't work, I'll backtrack and try another approach.
        
        Let me work through this step by step, showing the complete transformation from initial string to empty string.
        After completing my analysis, I'll summarize my solution as a sequence of transition indices.
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
