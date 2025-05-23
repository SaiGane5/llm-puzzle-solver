# Optimal Sed-Puzzle Generator

This repository provides a curated generator for **sed-puzzle** problems (string rewriting puzzles) designed to test a variety of reasoning skills. Each puzzle is guaranteed to be **solvable** by a baseline BFS solver within a short time limit, with no fallbacks. The dataset is balanced across three difficulty levels:

* **Easy (20%)**: Bracket matching
* **Medium (40%)**: Bubble sort simulation, Sequential dependencies
* **Hard (40%)**: Tower of Hanoi simulation, Symbolic logic, Single-rule elimination

## Puzzle Types & Examples

### 1. Bracket Matching

**Description**: Remove matching bracket pairs until the string is empty. Tests stack-like (LIFO) pattern recognition.

* **Generator**: `generate_bracket_matching` (Line  ... )
* **Example**:

  * Initial: `([{}])`
  * Transitions:

    1. `[[` → `]]` (decoy)
    2. `()` → \`\`  (exit)
    3. `([` → `] )` (padding)
    4. `])` → `([` (padding)
* **Edge Cases**: Mixed nested types; BFS may fail if depth > 4 or unusual nesting.
* **Reference**: [Bracket Matching on GeeksforGeeks](https://www.geeksforgeeks.org/check-for-balanced-parentheses-in-an-expression/)

### 2. Bubble Sort Simulation

**Description**: Simulate bubble sort through local swaps and eliminate when sorted.

* **Generator**: `generate_bubble_sort_simulation` (Line ...)
* **Example**:

  * Initial: `BBAA`
  * Transitions:

    1. `BA` → `AB`
    2. `BB` → `AA` (decoy)
    3. `AABB` → \`\` (exit)
    4. `AB` → `BA` (extra)
* **Edge Cases**: Random shuffles; potential long BFS if sequence length > 8.
* **Reference**: [Bubble Sort on Wikipedia](https://en.wikipedia.org/wiki/Bubble_sort)

### 3. Sequential Dependencies

**Description**: Remove segments in a specific order (dependencies) to empty the string.

* **Generator**: `generate_sequential_dependency` (Line ...)
* **Example**:

  * Initial: `X#XX#X`
  * Transitions:

    1. `X` → `XX` (decoy)
    2. `#` → \`\`  (exit)
    3. `XX` → `XX` (decoy)
    4. `#X` → `X#` (decoy)
* **Edge Cases**: Varying part lengths; BFS timeouts if too many segments.
* **Reference**: [Topological Sort (Dependency Resolution)](https://en.wikipedia.org/wiki/Topological_sorting)

### 4. Tower of Hanoi Simulation

**Description**: Simulate moving disks between pegs represented as strings.

* **Generator**: `generate_tower_of_hanoi_simulation` (Line ...)
* **Example**:

  * Initial: `321| | ` (3 disks on peg A)
  * Transitions:

    1. `1|` → `|1`
    2. `2|` → `|2`
    3. `3|` → `|3`
    4. `||321` → \`\` (exit)
    5. `3` → \`\` (padding)
    6. `|` → \`\` (padding)
* **Edge Cases**: Up to 4 disks; BFS may struggle with deeper recursion.
* **Reference**: [Tower of Hanoi on Wikipedia](https://en.wikipedia.org/wiki/Tower_of_Hanoi)

### 5. Symbolic Logic

**Description**: Apply propositional logic steps (e.g., modus ponens) via string rules.

* **Generator**: `generate_symbolic_logic_puzzle` (Line ...)
* **Example**:

  * Initial: `A->B->C&A`
  * Transitions:

    1. `A->B` → `B` (partial)
    2. `A->B->C&A` → `C` (exit)
    3. `C` → \`\`
    4. `A` → `~A`
    5. `B` → `C` (decoy)
* **Edge Cases**: Chains of length 3; unusual operator placement.
* **Reference**: [Modus Ponens on Stanford Encyclopedia](https://plato.stanford.edu/entries/modus-ponens/)

### 6. Single-Rule Elimination

**Description**: Only one rule removes the entire string; others expand it, requiring strategic choice.

* **Generator**: `generate_single_rule_elimination` (Line ...)
* **Example**:

  * Initial: `AB`
  * Transitions:

    1. `A` → `AA` (trap)
    2. `B` → `BB` (trap)
    3. `AB` → \`\`  (exit)
    4. `BA` → `AB` (decoy)
    5. `ABA` → `B` (extra)
* **Edge Cases**: Immediate choice needed; BFS identifies the correct exit.
* **Reference**: [Markov Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Markov_algorithm)

## Usage

1. **Install** dependencies: `pip install -r requirements.txt`
2. **Generate** puzzles:

   ```bash
   python dataset_generator.py
   ```
3. **Results** in `data/dataset/`, one JSON per puzzle.

## Caveats & Known Limitations

* BFS time limit set to **1 second** may still exceed on complex edge cases.
* Generators use randomness; extreme cases (max depth/shuffling) may rarely hang.
* No fallback, so some rare generator configurations may loop briefly before success.

## License

MIT License
