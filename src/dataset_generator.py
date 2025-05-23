import random
import logging
from typing import Dict, List
from pathlib import Path
from types import SimpleNamespace

from utils import write_problem_folder
from baseline import bfs  # BFS expects a problem with attributes

class OptimalSedPuzzleGenerator:
    """
    Curated sed-puzzle generator ensuring each puzzle is solvable via BFS.
    Generates exactly `total` puzzles: 20% easy, 40% medium, 40% hard.
    """

    def __init__(self):
        self.problem_id = 0
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.symbols = ["|", "_", "#", "@", "$", "%", "&", "*", "+", "-", "=", "~"]
        self.brackets = ["()", "[]", "{}", "<>"]

    def _is_solvable(self, initial: str, transitions: List[Dict[str, str]]) -> bool:
        # Wrap into an object with attributes expected by baseline.bfs
        prob = SimpleNamespace()
        prob.initial_string = initial
        prob.transitions = [SimpleNamespace(src=t['src'], tgt=t['tgt']) for t in transitions]
        # Use a shorter time limit to avoid long blocking
        return bfs(prob, time_limit=1) is not None

    def _format(self, init: str, trans: List[Dict[str, str]], ptype: str) -> Dict:
        return {
            "problem_id": f"{self.problem_id:03d}",
            "initial_string": init,
            "transitions": trans,
            "puzzle_type": ptype
        }

    def generate_bracket_matching(self) -> Dict:
        depth = random.randint(2, 4)
        types = random.sample(self.brackets, k=depth)
        opening = ''.join(b[0] for b in types)
        closing = ''.join(b[1] for b in reversed(types))
        initial = opening + closing
        for _ in range(random.randint(1, 2)):
            b = random.choice(self.brackets)
            initial = b[0] + initial + b[1]
        transitions = [
            # decoy
            {"src": types[0][0]*2, "tgt": types[0][1]*2},
            {"src": types[1][0]*2, "tgt": types[1][1]*2},
            # exit
            {"src": types[0][0]+types[0][1], "tgt": ""},
            # padding
            {"src": opening[:1], "tgt": closing[-1:]},
            {"src": closing[:1], "tgt": opening[-1:]}
        ]
        return self._format(initial, transitions, "bracket_matching")

    def generate_bubble_sort_simulation(self) -> Dict:
        c1, c2 = random.sample(self.letters, 2)
        n1, n2 = random.randint(2, 4), random.randint(2, 4)
        seq = [c1]*n1 + [c2]*n2
        random.shuffle(seq)
        initial = ''.join(seq)
        target = ''.join(sorted(seq))
        transitions = [
            {"src": c2+c1, "tgt": c1+c2},
            {"src": initial[:2], "tgt": initial[-2:]},
            {"src": target, "tgt": ""},
            {"src": seq[0]+seq[-1], "tgt": seq[-1]+seq[0]},
            {"src": seq[-1]+seq[0], "tgt": initial[:2]}  # extra
        ]
        return self._format(initial, transitions, "bubble_sort")

    def generate_sequential_dependency(self) -> Dict:
        ch = random.choice(self.letters)
        sep = random.choice(self.symbols)
        parts = [ch * random.randint(1, 3) for _ in range(random.randint(3, 6))]
        initial = sep.join(parts)
        transitions = [
            {"src": parts[0], "tgt": parts[-1]},
            {"src": parts[1], "tgt": parts[0]},
            {"src": sep, "tgt": ""},
            {"src": parts[-1]+parts[0], "tgt": parts[0]+parts[-1]}
        ]
        return self._format(initial, transitions, "sequential_dependency")

    def generate_tower_of_hanoi_simulation(self) -> Dict:
        d = random.randint(2, 4)
        pegs = {0: list(range(d, 0, -1)), 1: [], 2: []}
        initial = '|'.join(''.join(map(str, pegs[i])) for i in range(3))
        transitions = []
        # legal moves
        for s in range(3):
            for t in range(3):
                if s != t and pegs[s]:
                    disk = pegs[s][-1]
                    transitions.append({"src": f"{disk}|", "tgt": f"|{disk}"})
        # exit
        goal = ''.join(map(str, range(d, 0, -1)))
        transitions.append({"src": f"||{goal}", "tgt": ""})
        # padding
        transitions.append({"src": str(d), "tgt": ""})
        transitions.append({"src": "|", "tgt": ""})
        return self._format(initial, transitions, "tower_of_hanoi")

    def generate_symbolic_logic_puzzle(self) -> Dict:
        l = random.randint(2, 3)
        props = random.sample(self.letters, l + 1)
        impl = '->'.join(props)
        prem = props[0]
        concl = props[-1]
        initial = f"{impl}&{prem}"
        transitions = [
            {"src": '->'.join(props[:2]), "tgt": props[1]},
            {"src": f"{impl}&{prem}", "tgt": concl},
            {"src": concl, "tgt": ""},
            {"src": prem, "tgt": "~"+prem},
            {"src": props[1], "tgt": props[2] if len(props)>2 else prem}
        ]
        return self._format(initial, transitions, "symbolic_logic")

    def generate_single_rule_elimination(self) -> Dict:
        a, b = random.sample(self.letters, 2)
        initial = a + b
        transitions = [
            {"src": a, "tgt": a*2},
            {"src": b, "tgt": b*2},
            {"src": initial, "tgt": ""},
            {"src": b+a, "tgt": a+b},
            {"src": a+b+a, "tgt": b}  # extra
        ]
        return self._format(initial, transitions, "single_rule_elimination")

    def generate_puzzle_by_difficulty(self, diff: str) -> Dict:
        gens = {
            "easy": [self.generate_bracket_matching],
            "medium": [self.generate_bubble_sort_simulation, self.generate_sequential_dependency],
            "hard": [self.generate_tower_of_hanoi_simulation, self.generate_symbolic_logic_puzzle, self.generate_single_rule_elimination]
        }
        while True:
            puzzle = random.choice(gens[diff])()
            if self._is_solvable(puzzle["initial_string"], puzzle["transitions"]):
                puzzle["difficulty"] = diff
                puzzle["problem_id"] = f"{self.problem_id:03d}"
                self.problem_id += 1
                return puzzle

    def generate_dataset(self, total: int = 100) -> Dict[str, Dict]:
        dataset = {}
        self.problem_id = 0
        counts = {"easy": total * 20 // 100, "medium": total * 40 // 100, "hard": total * 40 // 100}
        for diff, count in counts.items():
            for _ in range(count):
                puzzle = self.generate_puzzle_by_difficulty(diff)
                dataset[puzzle["problem_id"]] = puzzle
        return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = OptimalSedPuzzleGenerator()
    data = gen.generate_dataset(100)
    write_problem_folder(data, Path("data/dataset"))
    dist = {}
    for p in data.values():
        dist[p["puzzle_type"]] = dist.get(p["puzzle_type"], 0) + 1
    logging.info(f"Generated puzzles by type: {dist}")
