# SED Puzzle Solver Codebase Documentation

This codebase implements a comprehensive framework for generating, solving, and evaluating "sed puzzles" - string transformation puzzles where the goal is to transform an initial string to an empty string by applying a sequence of replacement rules.

## Core Components

### Data Models (schema.py)

The fundamental data structures are defined using Pydantic models:

- **Transition**: Represents a string replacement rule with source and target patterns
- **Problem**: Contains a problem ID, initial string, and list of available transitions
- **Solution**: Consists of a problem ID and ordered list of transition indices to apply

These models include validation logic to ensure data integrity, such as checking that no transition is empty and every problem has at least one transition with an empty target.

### Puzzle Generator (generator.py)

The `PuzzleGenerator` class creates solvable puzzles with varying difficulty levels:

```python
generator = PuzzleGenerator(include_special=True, extra_rules=2)
puzzles = generator.generate_dataset(num_puzzles=100)
```

Features:
- Configurable difficulty levels (easy, medium, hard)
- Special character handling, including '?' wildcards
- Generation of both solution-critical and noise transitions
- Ensures all generated puzzles are solvable

### Solvers

#### Baseline Solver (baseline.py)

Implements a breadth-first search algorithm to find optimal solutions:

```python
solution = bfs(problem, time_limit=5)
```

The solver:
- Explores all possible transition applications
- Tracks parent states to reconstruct solution paths
- Terminates when an empty string is reached or time limit is exceeded

#### LLM-Based Solvers (llm_solver.py)

Multiple LLM-based approaches for solving puzzles:

1. **ZeroShotSolver**: Basic prompting without examples
2. **FewShotSolver**: Uses example puzzles and solutions to guide the LLM
3. **CoTSolver**: Chain of Thought prompting for step-by-step reasoning
4. **CreativeSolver**: Advanced approach with temperature adjustment and longer outputs

Currently supports Gemini models with extensibility for other LLM providers.

### Evaluation Framework (evaluation.py)

The `PuzzleEvaluator` class provides tools to assess solver performance:

```python
evaluator = PuzzleEvaluator()
results = evaluator.compare_solvers(problems, solutions_by_method)
```

Capabilities:
- Validates solutions by applying transitions sequentially
- Calculates performance metrics (success rate, solution length)
- Compares multiple solvers on the same problem set
- Generates detailed performance reports

### Utilities (utils.py)

Helper functions for file operations and solution validation:

- Reading/writing problems and solutions from disk
- Validating solutions by checking if they result in an empty string
- Logging utilities for debugging and monitoring

## Command-Line Interface (main.py)

The framework provides a comprehensive CLI with three main commands:

### Generate Dataset
```
python main.py generate --num_puzzles 100 --output_dir data/dataset
```

### Solve Puzzles
```
python main.py solve --puzzle_dir data/dataset --output_dir data/solutions --method creative --model gemini-1.5-flash
```

Available methods: baseline, zero_shot, few_shot, cot, creative

### Evaluate Solutions
```
python main.py evaluate --puzzle_dir data/dataset --solution_dir data/solutions --output_dir data/evaluation --methods baseline zero_shot cot
```

## Data Flow

1. Puzzles are generated with `generator.py` and stored as JSON files
2. Solvers attempt to find solutions using different approaches
3. Solutions are validated and compared using the evaluation framework
4. Results are saved for analysis and comparison