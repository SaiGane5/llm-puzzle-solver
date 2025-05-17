import random
import string
import logging
from pathlib import Path
from typing import List, Dict

from pydantic import ValidationError
from schema import Problem, Transition


class PuzzleGenerator:
    """Generator for SED puzzles of varying difficulty, with optional special characters."""

    def __init__(
        self,
        include_special: bool = True
    ):
        """Initialize the puzzle generator.

        include_special: whether to include special symbols in puzzles.
        """
        self.include_special = include_special
        # base pools
        self.base_chars = list(string.ascii_uppercase + string.digits)
        self.special_chars = ['?', '!', '.', '#', 'o', 'x']

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

    def _char_pool(self) -> List[str]:
        pool = list(self.base_chars)
        if self.include_special:
            pool += self.special_chars
        return pool

    def generate_random_string(self, length: int) -> str:
        """Generate a random string of specified length from allowed pool."""
        pool = self._char_pool()
        return ''.join(random.choice(pool) for _ in range(length))

    def generate_transitions(
        self,
        initial_string: str,
        difficulty: str
    ) -> List[Transition]:
        """Generate a mix of delete, replace, and swap transitions, ensuring at least one deletion."""
        params = self.difficulty_levels[difficulty]
        num_transitions = random.randint(*params["num_transitions"])
        complexity = params["transition_complexity"]
        transitions: List[Transition] = []
        chars = list(initial_string)

        for _ in range(num_transitions):
            kind = random.choices(
                ['delete', 'replace', 'swap'],
                weights=[1,
                         1 if complexity != 'simple' else 0,
                         1 if complexity == 'complex' else 0]
            )[0]
            max_chunk = {'simple': 3, 'medium': 5, 'complex': 7}[complexity]
            chunk_size = random.randint(1, min(max_chunk, len(chars)))
            src = ''.join(chars[:chunk_size])
            # rotate chars for variety
            chars = chars[chunk_size:] + chars[:chunk_size]

            if kind == 'delete':
                tgt = ''
            elif kind == 'replace':
                tgt_len = random.randint(1, chunk_size + 2)
                pool = self._char_pool()
                tgt = ''.join(random.choice(pool) for _ in range(tgt_len))
            else:
                tgt = src[::-1] if complexity != 'complex' else ''.join(random.sample(src, len(src)))

            transitions.append(Transition(src=src, tgt=tgt))

        # ensure at least one delete rule exists
        if not any(t.tgt == '' for t in transitions):
            # pick random chunk from initial_string
            chunk_size = random.randint(1, len(initial_string))
            src = initial_string[:chunk_size]
            transitions.append(Transition(src=src, tgt=''))

        random.shuffle(transitions)
        return transitions

    def generate_puzzle(
        self,
        difficulty: str,
        problem_id: str
    ) -> Problem:
        """Generate a single puzzle with valid delete transitions."""
        params = self.difficulty_levels[difficulty]
        length = random.randint(*params["string_length"])
        initial_string = self.generate_random_string(length)
        transitions = self.generate_transitions(initial_string, difficulty)

        # Construct Problem; let ValidationError propagate
        return Problem(
            problem_id=problem_id,
            initial_string=initial_string,
            transitions=transitions
        )

    def generate_dataset(
        self,
        num_puzzles: int = 100
    ) -> Dict[str, Problem]:
        """Generate a dataset with balanced difficulties, skipping invalid puzzles."""
        puzzles: Dict[str, Problem] = {}
        easy = num_puzzles // 3
        medium = num_puzzles // 3
        hard = num_puzzles - easy - medium
        counts = {'easy': easy, 'medium': medium, 'hard': hard}
        pid = 0

        for difficulty, total in counts.items():
            generated = 0
            while generated < total:
                key = str(pid).zfill(3)
                try:
                    puzzle = self.generate_puzzle(difficulty, key)
                    puzzles[key] = puzzle
                    generated += 1
                except ValidationError as e:
                    logging.warning(f"Skipping invalid puzzle {key}: {e}")
                finally:
                    pid += 1

        return puzzles
