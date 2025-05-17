import random
import string
import logging
from pathlib import Path
from typing import List, Dict

from schema import Problem, Transition
from pydantic import ValidationError


class PuzzleGenerator:
    """Generator for SED puzzles of varying difficulty, ensuring solvable puzzles with explicit ? handling."""

    def __init__(self, include_special: bool = True, extra_rules: int = 2):
        """
        include_special: whether to include special symbols
        extra_rules: number of spurious transitions to add
        """
        self.include_special = include_special
        self.extra_rules = extra_rules
        self.base_chars = list(string.ascii_uppercase + string.digits)
        self.special_chars = ['?', '!', '.', '#', 'o', 'x']
        self.difficulty_levels = {
            "easy": {"string_length": (5, 10), "step_complexity": 3, "max_steps": 4},
            "medium": {"string_length": (10, 20), "step_complexity": 5, "max_steps": 7},
            "hard": {"string_length": (15, 30), "step_complexity": 7, "max_steps": 10}
        }

    def _char_pool(self) -> List[str]:
        pool = list(self.base_chars)
        if self.include_special:
            # include all specials except '?' for replacement
            pool += [c for c in self.special_chars if c != '?']
        return pool

    def generate_random_string(self, length: int) -> str:
        pool = list(self.base_chars) + (self.special_chars if self.include_special else [])
        return ''.join(random.choice(pool) for _ in range(length))

    def generate_transitions(self, initial: str, difficulty: str) -> List[Transition]:
        """Construct transitions by first replacing all '?' individually, then simulate a solvable path, then add noise."""
        params = self.difficulty_levels[difficulty]
        complexity = params['step_complexity']
        max_steps = params['max_steps']

        s = initial
        solution_rules: List[Transition] = []

        # Handle '?' as standalone replacements
        qm_replacements: List[Transition] = []
        while '?' in s:
            # pick a replacement for '?'
            pool = self._char_pool()
            if not pool:
                break
            rep = random.choice(pool)
            qm_replacements.append(Transition(src='?', tgt=rep))
            # replace first occurrence
            idx = s.index('?')
            s = s[:idx] + rep + s[idx+1:]
        solution_rules.extend(qm_replacements)

        # Now greedily reduce to empty
        steps = 0
        while s and steps < max_steps:
            l = random.randint(1, min(complexity, len(s)))
            i = random.randrange(0, len(s) - l + 1)
            src = s[i:i+l]
            if random.random() < 0.7:
                tgt = ''
            else:
                tgt_len = random.randint(1, max(1, l-1))
                tgt = ''.join(random.choice(self._char_pool()) for _ in range(tgt_len))
            solution_rules.append(Transition(src=src, tgt=tgt))
            s = s[:i] + tgt + s[i+l:]
            steps += 1

        if s:
            solution_rules.append(Transition(src=s, tgt=''))

        # Add spurious noise rules
        noise_rules: List[Transition] = []
        pool = self._char_pool() + (['?'] if self.include_special else [])
        for _ in range(self.extra_rules):
            l = random.randint(1, complexity)
            src = ''.join(random.choice(pool) for _ in range(l))
            tgt = ''.join(random.choice(pool) for _ in range(random.randint(1, l+1)))
            noise_rules.append(Transition(src=src, tgt=tgt))

        all_rules = solution_rules + noise_rules
        random.shuffle(all_rules)
        return all_rules

    def generate_puzzle(self, difficulty: str, problem_id: str) -> Problem:
        params = self.difficulty_levels[difficulty]
        length = random.randint(*params["string_length"])
        initial = self.generate_random_string(length)
        transitions = self.generate_transitions(initial, difficulty)
        return Problem(problem_id=problem_id, initial_string=initial, transitions=transitions)

    def generate_dataset(self, num_puzzles: int = 100) -> Dict[str, Problem]:
        puzzles: Dict[str, Problem] = {}
        counts = {d: num_puzzles//3 for d in self.difficulty_levels}
        counts['hard'] += num_puzzles - sum(counts.values())
        pid = 0
        for difficulty, total in counts.items():
            for _ in range(total):
                key = str(pid).zfill(3)
                try:
                    puzzles[key] = self.generate_puzzle(difficulty, key)
                except ValidationError as e:
                    logging.warning(f"Invalid puzzle {key}: {e}")
                pid += 1
        return puzzles
