import json
import random
import os
from typing import List, Dict, Tuple

class MediumSedPuzzleGenerator:
    def __init__(self):
        self.problem_id = 0
        self.words = [
            "ALGORITHM", "FUNCTION", "VARIABLE", "CONSTANT", "PARAMETER", "STRUCTURE",
            "INTERFACE", "PROTOCOL", "NETWORK", "DATABASE", "SECURITY", "ENCRYPTION",
            "FRAMEWORK", "LIBRARY", "MODULE", "COMPONENT", "SERVICE", "CLIENT",
            "SERVER", "HANDLER", "PARSER", "COMPILER", "RUNTIME", "MEMORY"
        ]
        self.chars = ["A", "B", "C", "D", "E", "X", "Y", "Z", "1", "2", "3", "0", "+", "-", "*"]
        self.special_chars = ["?", "#", "@", "&", "%", "!", "~", "|"]
        
    def generate_conditional_replacement(self) -> Dict:
        """Generate puzzles with conditional replacements using placeholders"""
        placeholder = random.choice(self.special_chars)
        char1 = random.choice(self.chars)
        char2 = random.choice([c for c in self.chars if c != char1])
        
        # Create string with placeholders that can become either char1 or char2
        placeholders = [placeholder] * random.randint(4, 8)
        fixed_chars = random.choices(self.chars, k=random.randint(2, 4))
        
        # Interleave placeholders and fixed characters
        initial_parts = []
        for i in range(max(len(placeholders), len(fixed_chars))):
            if i < len(placeholders):
                initial_parts.append(placeholders[i])
            if i < len(fixed_chars):
                initial_parts.append(fixed_chars[i])
        
        initial_string = "".join(initial_parts)
        
        # Create elimination patterns
        elimination_pattern1 = char1 * random.randint(2, 3)
        elimination_pattern2 = char2 * random.randint(2, 3)
        
        transitions = [
            {"src": placeholder, "tgt": char1},
            {"src": placeholder, "tgt": char2},
            {"src": elimination_pattern1, "tgt": ""},
            {"src": elimination_pattern2, "tgt": ""}
        ]
        
        # Add transitions to remove remaining fixed characters
        for char in set(fixed_chars):
            transitions.append({"src": char, "tgt": ""})
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_multi_step_pattern(self) -> Dict:
        """Generate puzzles requiring multiple pattern matching steps"""
        # Create a puzzle with nested replacements
        base_word = random.choice(self.words[:12])
        wrapper_char = random.choice(self.chars)
        separator = random.choice(self.special_chars)
        
        # Create pattern like: XBASEX#XBASEX where X is wrapper, # is separator
        segment = wrapper_char + base_word + wrapper_char
        initial_string = segment + separator + segment
        
        # Create multi-step solution
        intermediate_char = random.choice([c for c in self.chars if c != wrapper_char])
        
        transitions = [
            {"src": wrapper_char + base_word + wrapper_char, "tgt": intermediate_char},
            {"src": separator, "tgt": ""},
            {"src": intermediate_char, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_regex_like_pattern(self) -> Dict:
        """Generate puzzles that simulate regex-like behavior"""
        # Create patterns that require understanding of grouping
        group_chars = random.sample(self.chars, 3)
        a, b, c = group_chars
        
        # Pattern like: AaBbCcAaBbCc -> remove paired characters
        pattern = a + a.lower() + b + b.lower() + c + c.lower()
        if hasattr(str, 'lower'):  # Simple fallback
            pattern = a + "a" + b + "b" + c + "c"
        
        initial_string = pattern * 2
        
        # Remove each paired group
        transitions = [
            {"src": a + "a", "tgt": ""},
            {"src": b + "b", "tgt": ""},
            {"src": c + "c", "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_transformation_chain(self) -> Dict:
        """Generate puzzles with transformation chains"""
        # Start with one pattern and transform it through multiple steps
        start_char = random.choice(self.chars)
        mid_char = random.choice([c for c in self.chars if c != start_char])
        end_marker = random.choice(self.special_chars)
        
        # Create a chain: start -> mid -> end_marker -> empty
        initial_length = random.randint(6, 10)
        initial_string = start_char * initial_length
        
        # Add some complexity with mixed patterns
        noise_chars = random.sample([c for c in self.chars if c not in [start_char, mid_char]], 2)
        for noise in noise_chars:
            pos = random.randint(1, len(initial_string) - 1)
            initial_string = initial_string[:pos] + noise + initial_string[pos:]
        
        transitions = [
            {"src": start_char, "tgt": mid_char},
            {"src": mid_char + mid_char, "tgt": end_marker},
            {"src": end_marker, "tgt": ""}
        ]
        
        # Add transitions to clean up noise
        for noise in noise_chars:
            transitions.append({"src": noise, "tgt": ""})
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_context_dependent(self) -> Dict:
        """Generate puzzles where replacements depend on context"""
        # Create a pattern where the same text has different replacements based on position
        base = random.choice(self.words[:8])
        prefix = random.choice(self.chars)
        suffix = random.choice([c for c in self.chars if c != prefix])
        middle_marker = random.choice(self.special_chars)
        
        # Pattern: prefixBASEprefix#suffixBASEsuffix
        initial_string = prefix + base + prefix + middle_marker + suffix + base + suffix
        
        # Different replacements based on context
        replacement1 = random.choice(self.chars)
        replacement2 = random.choice([c for c in self.chars if c != replacement1])
        
        transitions = [
            {"src": prefix + base + prefix, "tgt": replacement1},
            {"src": suffix + base + suffix, "tgt": replacement2},
            {"src": middle_marker, "tgt": ""},
            {"src": replacement1, "tgt": ""},
            {"src": replacement2, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_nested_structures(self) -> Dict:
        """Generate puzzles with nested structures"""
        # Create nested brackets-like structures
        outer = random.choice(self.chars)
        inner = random.choice([c for c in self.chars if c != outer])
        content = random.choice(self.words[:10])
        
        # Pattern: outer[inner[content]inner]outer
        initial_string = outer + inner + content + inner + outer
        
        # Add some repeated patterns
        initial_string = initial_string + random.choice(self.special_chars) + initial_string
        
        # Work from inside out
        separator = random.choice(self.special_chars)
        transitions = [
            {"src": inner + content + inner, "tgt": content},
            {"src": outer + content + outer, "tgt": separator},
            {"src": separator, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def is_solvable(self, puzzle: Dict) -> bool:
        """Check if puzzle is solvable using backtracking"""
        def solve_recursive(current_string: str, remaining_transitions: List[Dict], used_indices: List[int]) -> bool:
            if current_string == "":
                return True
            
            if not remaining_transitions:
                return False
            
            # Try each remaining transition
            for i, trans in enumerate(remaining_transitions):
                if i in used_indices:
                    continue
                    
                if trans["src"] in current_string:
                    # Apply transition
                    new_string = current_string.replace(trans["src"], trans["tgt"], 1)
                    new_used = used_indices + [i]
                    
                    # Check if this transition can be applied again
                    new_remaining = remaining_transitions.copy()
                    if trans["src"] not in new_string:
                        new_remaining = [t for j, t in enumerate(remaining_transitions) if j != i]
                        new_used = [idx if idx < i else idx - 1 for idx in new_used if idx != i]
                    
                    if solve_recursive(new_string, new_remaining, [] if len(new_remaining) != len(remaining_transitions) else new_used):
                        return True
            
            return False
        
        return solve_recursive(puzzle["initial_string"], puzzle["transitions"], [])
    
    def generate_puzzle(self) -> Dict:
        """Generate a random medium puzzle"""
        generators = [
            self.generate_conditional_replacement,
            self.generate_multi_step_pattern,
            self.generate_regex_like_pattern,
            self.generate_transformation_chain,
            self.generate_context_dependent,
            self.generate_nested_structures
        ]
        
        max_attempts = 50
        for _ in range(max_attempts):
            generator = random.choice(generators)
            puzzle = generator()
            puzzle["problem_id"] = f"{self.problem_id:03d}"
            
            if self.is_solvable(puzzle):
                self.problem_id += 1
                return puzzle
        
        # Fallback puzzle
        word = random.choice(self.words[:5])
        wrapper = random.choice(self.chars)
        puzzle = {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": wrapper + word + wrapper,
            "transitions": [
                {"src": wrapper, "tgt": ""},
                {"src": word, "tgt": ""}
            ]
        }
        self.problem_id += 1
        return puzzle
    
    def generate_dataset(self, num_problems: int = 35) -> List[Dict]:
        """Generate a dataset of medium puzzles"""
        dataset = []
        self.problem_id = 0
        
        for _ in range(num_problems):
            puzzle = self.generate_puzzle()
            dataset.append(puzzle)
        
        return dataset

def main():
    generator = MediumSedPuzzleGenerator()
    dataset = generator.generate_dataset(35)
    
    # Create directory if it doesn't exist
    os.makedirs("data/dataset/medium", exist_ok=True)
    
    # Save each puzzle to a separate file
    for puzzle in dataset:
        filename = f"data/dataset/medium/{puzzle['problem_id']}.json"
        with open(filename, 'w') as f:
            json.dump(puzzle, f, indent=2)
    
    print(f"Generated {len(dataset)} medium puzzles in data/dataset/medium/")

if __name__ == "__main__":
    main()