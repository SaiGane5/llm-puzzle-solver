# src/generator.py
import random
import string
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from schema import Problem, Transition

class PuzzleGenerator:
    """Generator for sed puzzles of varying difficulty."""
    
    def __init__(self):
        """Initialize the puzzle generator."""
        self.difficulty_levels = {
            "easy": {
                "string_length": (5, 10),
                "num_transitions": (2, 4),
                "transition_complexity": "simple"
            },
            "medium": {
                "string_length": (10, 20),
                "num_transitions": (4, 7),
                "transition_complexity": "medium"
            },
            "hard": {
                "string_length": (15, 30),
                "num_transitions": (7, 10),
                "transition_complexity": "complex"
            }
        }
    
    def generate_random_string(self, length: int) -> str:
        """Generate a random string of specified length."""
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
    
    def generate_transitions(self, initial_string: str, difficulty: str) -> List[Transition]:
        """Generate transitions based on difficulty."""
        params = self.difficulty_levels[difficulty]
        num_transitions = random.randint(*params["num_transitions"])
        complexity = params["transition_complexity"]
        
        # Working backwards approach
        transitions = []
        current = ""  # Start with empty string (goal)
        
        # Generate transitions that build up to the initial string
        remaining_chars = list(initial_string)
        
        for _ in range(num_transitions - 1):  # Save one transition for final step
            if not remaining_chars:
                break
                
            # Decide on transition complexity
            if complexity == "simple":
                chunk_size = random.randint(1, min(3, len(remaining_chars)))
            elif complexity == "medium":
                chunk_size = random.randint(1, min(5, len(remaining_chars)))
            else:  # complex
                chunk_size = random.randint(1, min(7, len(remaining_chars)))
            
            # Take a chunk of characters from remaining
            chunk = ''.join(remaining_chars[:chunk_size])
            remaining_chars = remaining_chars[chunk_size:]
            
            # Create transition (empty string -> chunk)
            transitions.append(Transition(src=chunk, tgt=""))
        
        # Create a final transition for any remaining characters
        if remaining_chars:
            final_chunk = ''.join(remaining_chars)
            transitions.append(Transition(src=final_chunk, tgt=""))
        
        # Shuffle transitions
        random.shuffle(transitions)
        
        return transitions
    
    def generate_puzzle(self, difficulty: str, problem_id: str) -> Problem:
        """Generate a single puzzle of specified difficulty."""
        params = self.difficulty_levels[difficulty]
        
        # Generate string
        string_length = random.randint(*params["string_length"])
        initial_string = self.generate_random_string(string_length)
        
        # Generate transitions
        transitions = self.generate_transitions(initial_string, difficulty)
        
        # Create puzzle
        return Problem(
            problem_id=problem_id,
            initial_string=initial_string,
            transitions=transitions
        )
    
    def generate_dataset(self, num_puzzles: int = 100) -> Dict[str, Problem]:
        """Generate a dataset of puzzles with balanced difficulty distribution."""
        puzzles = {}
        
        # Distribution across difficulty levels
        easy_count = num_puzzles // 3
        medium_count = num_puzzles // 3
        hard_count = num_puzzles - easy_count - medium_count
        
        # Generate puzzles for each difficulty
        puzzle_id = 0
        
        # Easy puzzles
        for i in range(easy_count):
            pid = str(puzzle_id).zfill(3)
            puzzles[pid] = self.generate_puzzle("easy", pid)
            puzzle_id += 1
            
        # Medium puzzles
        for i in range(medium_count):
            pid = str(puzzle_id).zfill(3)
            puzzles[pid] = self.generate_puzzle("medium", pid)
            puzzle_id += 1
            
        # Hard puzzles
        for i in range(hard_count):
            pid = str(puzzle_id).zfill(3)
            puzzles[pid] = self.generate_puzzle("hard", pid)
            puzzle_id += 1
        
        return puzzles
