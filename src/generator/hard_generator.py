import json
import random
import os
from typing import List, Dict, Tuple

class HardSedPuzzleGenerator:
    def __init__(self):
        self.problem_id = 0
        self.complex_words = [
            "ALGORITHM", "ARCHITECTURE", "IMPLEMENTATION", "SPECIFICATION", "OPTIMIZATION",
            "SYNCHRONIZATION", "PARALLELIZATION", "VIRTUALIZATION", "ABSTRACTION", "ENCAPSULATION",
            "POLYMORPHISM", "INHERITANCE", "COMPOSITION", "AGGREGATION", "ASSOCIATION",
            "AUTHENTICATION", "AUTHORIZATION", "CRYPTOGRAPHY", "STEGANOGRAPHY", "TOKENIZATION"
        ]
        self.chars = ["A", "B", "C", "D", "E", "F", "G", "H", "X", "Y", "Z", "0", "1", "2", "3"]
        self.operators = ["+", "-", "*", "/", "=", "<", ">", "&", "|", "^"]
        self.delimiters = ["(", ")", "[", "]", "{", "}", "?", "#", "@", "%", "!", "~"]
        
    def generate_multi_phase_elimination(self) -> Dict:
        """Generate puzzles requiring multiple phases of transformation"""
        # Phase 1: Transform operators
        # Phase 2: Eliminate patterns
        # Phase 3: Final cleanup
        
        base_ops = random.sample(self.operators, 3)
        base_chars = random.sample(self.chars, 3)
        delim = random.choice(self.delimiters)
        
        # Create complex expression: A+B*C-A+B*C
        expr_parts = []
        for i in range(6):
            if i % 2 == 0:
                expr_parts.append(random.choice(base_chars))
            else:
                expr_parts.append(random.choice(base_ops))
        
        initial_string = "".join(expr_parts) + delim + "".join(expr_parts)
        
        # Multi-phase elimination
        intermediate1 = random.choice(self.chars)
        intermediate2 = random.choice([c for c in self.chars if c != intermediate1])
        
        transitions = []
        
        # Phase 1: Replace operators with intermediate markers
        for op in base_ops:
            transitions.append({"src": op, "tgt": intermediate1})
        
        # Phase 2: Eliminate paired patterns
        transitions.append({"src": intermediate1 + intermediate1, "tgt": intermediate2})
        transitions.append({"src": delim, "tgt": ""})
        
        # Phase 3: Final cleanup
        for char in base_chars:
            transitions.append({"src": char, "tgt": ""})
        transitions.append({"src": intermediate1, "tgt": ""})
        transitions.append({"src": intermediate2, "tgt": ""})
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_recursive_pattern(self) -> Dict:
        """Generate puzzles with recursive/self-similar patterns"""
        # Create fractal-like patterns
        base = random.choice(self.chars)
        expander = random.choice(self.delimiters)
        
        # Initial pattern that expands recursively
        pattern = base + expander + base
        
        # Create expansion rules
        expansion1 = random.choice(self.chars) + expander + random.choice(self.chars)
        expansion2 = pattern  # Self-referential
        
        # Build initial string with some expansions already applied
        initial_string = pattern + expansion1 + pattern
        
        # Add complexity with nested expansions
        complex_part = expander + expansion1 + expander
        initial_string = initial_string + complex_part
        
        # Collapse transitions (reverse of expansion)
        collapse_marker = random.choice([c for c in self.chars if c != base])
        final_marker = random.choice(self.operators)
        
        transitions = [
            {"src": expander, "tgt": collapse_marker},
            {"src": collapse_marker + base + collapse_marker, "tgt": final_marker},
            {"src": final_marker + final_marker, "tgt": ""},
            {"src": base, "tgt": ""},
            {"src": collapse_marker, "tgt": ""}
        ]
        
        # Add transitions for expansion characters
        for char in expansion1:
            if char != expander:
                transitions.append({"src": char, "tgt": ""})
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_state_machine_pattern(self) -> Dict:
        """Generate puzzles that simulate state machine behavior"""
        # Create a pattern that represents state transitions
        states = random.sample(self.chars, 4)
        s1, s2, s3, s4 = states
        
        transition_marker = random.choice(self.delimiters)
        accept_marker = random.choice(self.operators)
        
        # State transition sequence: s1->s2->s3->s4->accept
        state_sequence = [s1, transition_marker, s2, transition_marker, s3, transition_marker, s4]
        initial_string = "".join(state_sequence) + accept_marker + "".join(state_sequence)
        
        # Add some non-deterministic paths
        branch_state = random.choice(self.chars)
        error_marker = random.choice([c for c in self.delimiters if c != transition_marker])
        
        # Insert branches
        initial_string = initial_string[:len(initial_string)//2] + error_marker + branch_state + error_marker + initial_string[len(initial_string)//2:]
        
        # State machine transitions
        intermediate = random.choice(self.chars)
        
        transitions = [
            # State transitions
            {"src": s1 + transition_marker + s2, "tgt": intermediate},
            {"src": s2 + transition_marker + s3, "tgt": intermediate},
            {"src": s3 + transition_marker + s4, "tgt": intermediate},
            
            # Error handling
            {"src": error_marker + branch_state + error_marker, "tgt": ""},
            
            # Final acceptance
            {"src": s4 + accept_marker + intermediate, "tgt": accept_marker},
            {"src": intermediate + accept_marker, "tgt": ""},
            {"src": accept_marker, "tgt": ""},
            {"src": intermediate, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_parser_like_pattern(self) -> Dict:
        """Generate puzzles that simulate parsing complex structures"""
        # Create nested structure like programming language syntax
        open_delim = random.choice(["(", "[", "{"])
        close_delim = {"(": ")", "[": "]", "{": "}"}[open_delim]
        
        identifier = random.choice(self.complex_words[:8])
        operator = random.choice(self.operators)
        separator = random.choice([",", ";", ":"])
        
        # Create nested structure: FUNC(ARG1,ARG2,(NESTED))
        arg1 = random.choice(self.chars)
        arg2 = random.choice([c for c in self.chars if c != arg1])
        nested = random.choice(self.chars[:5])
        
        inner = open_delim + nested + close_delim
        outer = identifier + open_delim + arg1 + separator + arg2 + separator + inner + close_delim
        
        # Add operator context
        initial_string = outer + operator + outer
        
        # Parse from inside out
        reduce_marker1 = random.choice(self.delimiters[:3])
        reduce_marker2 = random.choice(self.delimiters[3:6])
        reduce_marker3 = random.choice(self.delimiters[6:])
        
        transitions = [
            # Reduce innermost expressions
            {"src": open_delim + nested + close_delim, "tgt": reduce_marker1},
            
            # Reduce argument lists
            {"src": arg1 + separator + arg2 + separator + reduce_marker1, "tgt": reduce_marker2},
            
            # Reduce function calls
            {"src": identifier + open_delim + reduce_marker2 + close_delim, "tgt": reduce_marker3},
            
            # Handle binary operation
            {"src": reduce_marker3 + operator + reduce_marker3, "tgt": reduce_marker3},
            
            # Final reduction
            {"src": reduce_marker3, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_cryptographic_pattern(self) -> Dict:
        """Generate puzzles resembling cryptographic operations"""
        # Create patterns that simulate encryption/decryption
        key_chars = random.sample(self.chars, 3)
        plaintext_chars = random.sample([c for c in self.chars if c not in key_chars], 4)
        cipher_marker = random.choice(self.delimiters)
        
        # Simulate XOR-like operation with key expansion
        key_pattern = "".join(key_chars)
        
        # Create "encrypted" segments
        encrypted_segments = []
        for char in plaintext_chars:
            encrypted_segments.append(cipher_marker + char + cipher_marker)
        
        initial_string = key_pattern + "".join(encrypted_segments) + key_pattern
        
        # Decryption process
        decrypt_marker = random.choice(self.operators)
        final_marker = random.choice([c for c in self.delimiters if c != cipher_marker])
        
        transitions = [
            # Remove cipher markers to reveal plaintext
            {"src": cipher_marker, "tgt": ""},
            
            # Apply key to decrypt (simulate XOR with key)
            {"src": key_pattern, "tgt": decrypt_marker},
            
            # Combine decrypted parts
            {"src": decrypt_marker, "tgt": final_marker},
            
            # Remove plaintext characters
            *[{"src": char, "tgt": ""} for char in plaintext_chars],
            
            # Final cleanup
            {"src": final_marker, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_compiler_optimization_pattern(self) -> Dict:
        """Generate puzzles that simulate compiler optimizations"""
        # Create patterns that represent code optimization steps
        vars = random.sample(self.chars, 4)
        ops = random.sample(self.operators, 3)
        temp_marker = random.choice(self.delimiters)
        
        # Create expression like: A+B*C+A+B*C (redundant computation)
        expr1 = vars[0] + ops[0] + vars[1] + ops[1] + vars[2]
        expr2 = vars[0] + ops[0] + vars[1] + ops[1] + vars[2]  # Duplicate
        
        initial_string = expr1 + temp_marker + expr2
        
        # Optimization phases
        cse_marker = random.choice(self.chars)  # Common Subexpression Elimination
        opt_marker = random.choice(self.operators)
        dead_code_marker = random.choice(self.delimiters)
        
        transitions = [
            # Phase 1: Common Subexpression Elimination
            {"src": expr1, "tgt": cse_marker},
            {"src": cse_marker + temp_marker + cse_marker, "tgt": opt_marker},
            
            # Phase 2: Dead Code Elimination
            {"src": opt_marker, "tgt": dead_code_marker},
            
            # Phase 3: Final optimization - remove everything
            {"src": dead_code_marker, "tgt": ""}
        ]
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def generate_complex_nested_structure(self) -> Dict:
        """Generate highly nested structures requiring careful unwrapping"""
        # Create deeply nested structure
        layers = random.sample(self.delimiters, 4)
        core = random.choice(self.complex_words[:5])
        separators = random.sample(self.operators, 2)
        
        # Build nested structure: layer1[layer2[layer3[core]layer3]layer2]layer1
        nested = core
        for i, layer in enumerate(layers):
            if i > 0:
                sep = separators[i % len(separators)]
                nested = layer + sep + nested + sep + layer
            else:
                nested = layer + nested + layer
        
        # Add some parallel structure
        parallel_core = random.choice(self.chars)
        parallel_nested = parallel_core
        for layer in layers[:2]:
            parallel_nested = layer + parallel_nested + layer
        
        initial_string = nested + parallel_nested
        
        # Unwrap from outside in
        unwrap_markers = random.sample(self.chars, len(layers))
        
        transitions = []
        
        # Unwrap nested structure
        for i, layer in enumerate(layers):
            if i == 0:
                transitions.append({"src": layer + core + layer, "tgt": unwrap_markers[i]})
            else:
                prev_marker = unwrap_markers[i-1]
                sep = separators[i % len(separators)]
                transitions.append({"src": layer + sep + prev_marker + sep + layer, "tgt": unwrap_markers[i]})
        
        # Handle parallel structure
        for layer in layers[:2]:
            transitions.append({"src": layer, "tgt": ""})
        
        # Final cleanup
        transitions.append({"src": parallel_core, "tgt": ""})
        for marker in unwrap_markers:
            transitions.append({"src": marker, "tgt": ""})
        
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": initial_string,
            "transitions": transitions
        }
    
    def is_solvable(self, puzzle: Dict) -> bool:
        """Check if puzzle is solvable using iterative deepening"""
        def can_solve_with_depth(current: str, transitions: List[Dict], depth: int, used_count: Dict[int, int]) -> bool:
            if current == "":
                return True
            if depth <= 0:
                return False
            
            # Try each transition
            for i, trans in enumerate(transitions):
                if trans["src"] in current:
                    # Apply transition
                    new_current = current.replace(trans["src"], trans["tgt"], 1)
                    new_used = used_count.copy()
                    new_used[i] = new_used.get(i, 0) + 1
                    
                    # Limit how many times a transition can be used
                    if new_used[i] <= 10:  # Reasonable limit
                        if can_solve_with_depth(new_current, transitions, depth - 1, new_used):
                            return True
            
            return False
        
        # Try with increasing depth limits
        for max_depth in [10, 20, 30]:
            if can_solve_with_depth(puzzle["initial_string"], puzzle["transitions"], max_depth, {}):
                return True
        
        return False
    
    def generate_puzzle(self) -> Dict:
        """Generate a random hard puzzle"""
        generators = [
            self.generate_multi_phase_elimination,
            self.generate_recursive_pattern,
            self.generate_state_machine_pattern,
            self.generate_parser_like_pattern,
            self.generate_cryptographic_pattern,
            self.generate_compiler_optimization_pattern,
            self.generate_complex_nested_structure
        ]
        
        max_attempts = 30
        for _ in range(max_attempts):
            generator = random.choice(generators)
            puzzle = generator()
            puzzle["problem_id"] = f"{self.problem_id:03d}"
            
            if self.is_solvable(puzzle):
                self.problem_id += 1
                return puzzle
        
        # Fallback puzzle for hard level
        word = random.choice(self.complex_words[:3])
        wrapper1 = random.choice(self.chars)
        wrapper2 = random.choice([c for c in self.chars if c != wrapper1])
        separator = random.choice(self.delimiters)
        
        puzzle = {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": wrapper1 + wrapper2 + word + wrapper2 + wrapper1 + separator + wrapper1 + word + wrapper1,
            "transitions": [
                {"src": wrapper2 + word + wrapper2, "tgt": word},
                {"src": wrapper1 + word + wrapper1, "tgt": separator},
                {"src": separator, "tgt": ""}
            ]
        }
        self.problem_id += 1
        return puzzle
    
    def generate_dataset(self, num_problems: int = 30) -> List[Dict]:
        """Generate a dataset of hard puzzles"""
        dataset = []
        self.problem_id = 0
        
        for _ in range(num_problems):
            puzzle = self.generate_puzzle()
            dataset.append(puzzle)
        
        return dataset

def main():
    generator = HardSedPuzzleGenerator()
    dataset = generator.generate_dataset(30)
    
    # Create directory if it doesn't exist
    os.makedirs("data/dataset/hard", exist_ok=True)
    
    # Save each puzzle to a separate file
    for puzzle in dataset:
        filename = f"data/dataset/hard/{puzzle['problem_id']}.json"
        with open(filename, 'w') as f:
            json.dump(puzzle, f, indent=2)
    
    print(f"Generated {len(dataset)} hard puzzles in data/dataset/hard/")

if __name__ == "__main__":
    main()