import json
import random
import os
from typing import List, Dict, Tuple

class EasySedPuzzleGenerator:
    def __init__(self):
        self.problem_id = 0
        self.words = [
            "HELLO", "WORLD", "PYTHON", "CODE", "PUZZLE", "SIMPLE", "EASY", "TEST",
            "BASIC", "START", "BEGIN", "CLEAN", "CLEAR", "DONE", "FINISH", "END",
            "APPLE", "ORANGE", "BANANA", "FRUIT", "FOOD", "DRINK", "WATER", "MILK"
        ]
        self.chars = ["A", "B", "C", "X", "Y", "Z", "1", "2", "3"]
        
    def generate_simple_substitution(self) -> Dict:
        """Generate puzzles with simple word/character replacement"""
        puzzle_type = random.choice(["word", "char", "repeated_char"])
        
        if puzzle_type == "word":
            # Simple word replacement
            words = random.sample(self.words, random.randint(2, 4))
            initial_string = "".join(words)
            transitions = [{"src": word, "tgt": ""} for word in words]
            
        elif puzzle_type == "char":
            # Character replacement
            chars = random.sample(self.chars, random.randint(3, 5))
            initial_string = "".join(random.choices(chars, k=random.randint(8, 15)))
            transitions = [{"src": char, "tgt": ""} for char in set(initial_string)]
            
        else:  # repeated_char
            # Replace repeated characters
            char = random.choice(self.chars)
            pattern = char * random.randint(2, 4)
            other_chars = random.choices([c for c in self.chars if c != char], k=random.randint(3, 6))
            initial_string = pattern + "".join(other_chars) + pattern
            transitions = [
                {"src": pattern, "tgt": ""},
                {"src": "".join(other_chars), "tgt": ""}
            ]
            
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_pattern_removal(self) -> Dict:
        """Generate puzzles with pattern-based removal"""
        pattern_types = ["prefix_suffix", "alternating", "sandwich"]
        pattern_type = random.choice(pattern_types)
        
        if pattern_type == "prefix_suffix":
            # Remove prefix and suffix
            prefix = random.choice(self.words[:8])
            suffix = random.choice(self.words[:8])
            middle = random.choice(self.words[8:])
            initial_string = prefix + middle + suffix
            transitions = [
                {"src": prefix, "tgt": ""},
                {"src": suffix, "tgt": ""},
                {"src": middle, "tgt": ""}
            ]
            
        elif pattern_type == "alternating":
            # Remove alternating patterns
            pattern1 = random.choice(self.chars)
            pattern2 = random.choice([c for c in self.chars if c != pattern1])
            length = random.randint(6, 12)
            initial_string = "".join([pattern1 if i % 2 == 0 else pattern2 for i in range(length)])
            transitions = [
                {"src": pattern1, "tgt": ""},
                {"src": pattern2, "tgt": ""}
            ]
            
        else:  # sandwich
            # Remove outer layers
            outer = random.choice(self.chars)
            inner = random.choice([c for c in self.chars if c != outer])
            initial_string = outer * 2 + inner * random.randint(3, 6) + outer * 2
            transitions = [
                {"src": outer, "tgt": ""},
                {"src": inner, "tgt": ""}
            ]
            
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_sequential_removal(self) -> Dict:
        """Generate puzzles requiring sequential operations"""
        seq_types = ["nested", "dependent", "cascade"]
        seq_type = random.choice(seq_types)
        
        if seq_type == "nested":
            # Nested patterns
            outer_word = random.choice(self.words[:10])
            inner_word = random.choice(self.words[10:])
            initial_string = outer_word + inner_word + outer_word
            transitions = [
                {"src": outer_word, "tgt": ""},
                {"src": inner_word, "tgt": ""}
            ]
            
        elif seq_type == "dependent":
            # One removal enables another
            base = random.choice(self.words)
            separator = random.choice(self.chars)
            initial_string = base + separator + base
            transitions = [
                {"src": separator, "tgt": ""},
                {"src": base, "tgt": ""}
            ]
            
        else:  # cascade
            # Multiple small removals
            parts = random.sample(self.words[:12], 3)
            initial_string = "".join(parts)
            transitions = [{"src": part, "tgt": ""} for part in parts]
            
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def is_solvable(self, puzzle: Dict) -> bool:
        """Check if puzzle is solvable"""
        current = puzzle["initial_string"]
        transitions = puzzle["transitions"][:]
        
        # Try all possible orders of applying transitions
        from itertools import permutations
        
        for perm in permutations(range(len(transitions))):
            temp_current = current
            valid = True
            
            for i in perm:
                trans = transitions[i]
                if trans["src"] in temp_current:
                    temp_current = temp_current.replace(trans["src"], trans["tgt"], 1)
                else:
                    valid = False
                    break
                    
            if valid and temp_current == "":
                return True
                
        return False
    
    def generate_puzzle(self) -> Dict:
        """Generate a random easy puzzle"""
        generators = [
            self.generate_simple_substitution,
            self.generate_pattern_removal,
            self.generate_sequential_removal
        ]
        
        max_attempts = 50
        for _ in range(max_attempts):
            generator = random.choice(generators)
            puzzle = generator()
            puzzle["problem_id"] = f"{self.problem_id:03d}"
            
            if self.is_solvable(puzzle):
                self.problem_id += 1
                return puzzle
                
        # Fallback to simple substitution if all attempts fail
        word = random.choice(self.words)
        puzzle = {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": word,
            "transitions": [{"src": word, "tgt": ""}]
        }
        self.problem_id += 1
        return puzzle
    
    def generate_dataset(self, num_problems: int = 35) -> List[Dict]:
        """Generate a dataset of easy puzzles"""
        dataset = []
        self.problem_id = 0
        
        for _ in range(num_problems):
            puzzle = self.generate_puzzle()
            dataset.append(puzzle)
            
        return dataset

def main():
    generator = EasySedPuzzleGenerator()
    dataset = generator.generate_dataset(35)
    
    # Create directory if it doesn't exist
    os.makedirs("data/dataset/easy", exist_ok=True)
    
    # Save each puzzle to a separate file
    for puzzle in dataset:
        filename = f"data/dataset/easy/{puzzle['problem_id']}.json"
        with open(filename, 'w') as f:
            json.dump(puzzle, f, indent=2)
    
    print(f"Generated {len(dataset)} easy puzzles in data/dataset/easy/")

if __name__ == "__main__":
    main()