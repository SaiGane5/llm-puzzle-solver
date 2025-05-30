import os
from pathlib import Path
import schema
import json
import logging

import pydantic

def read_problem_folder(path=Path("data/dataset/")):
    """Opens all problems at the folder and reads them using Pydantic"""
    problems = {}
    for file_path in path.iterdir():
        problem_data = json.loads(file_path.read_text())
        try:
            problem = schema.Problem(**problem_data)
            problems[problem.problem_id] = problem
        except pydantic.ValidationError as e:
            logging.warning(f"Validation error while processing {file_path}! skipping...", exc_info=True)
    return problems

def read_solution_folder(path=Path("data/solutions/baseline/")):
    """Opens all solutions at the folder and reads them using Pydantic"""
    solutions = {}
    for file_path in path.iterdir():
        solution_data = json.loads(file_path.read_text())
        try:
            solution = schema.Solution(**solution_data)
            solutions[solution.problem_id] = solution
        except pydantic.ValidationError as e:
            logging.warning(f"Validation error while processing {file_path}! skipping... ", exc_info=True)
    return solutions

def write_problem_folder(problems, path=Path("data/dataset/")):
    path.mkdir(parents=True, exist_ok=True)

    for problem_id, problem in problems.items():
        logging.info(f"Saving problem {problem_id}...")
        with open(path / f"{problem_id}.json", 'w') as f:
            # Handle both Pydantic objects and regular dictionaries
            if hasattr(problem, 'json'):
                # Pydantic v1 style
                f.write(problem.json())
            elif hasattr(problem, 'model_dump_json'):
                # Pydantic v2 style
                f.write(problem.model_dump_json())
            else:
                # Regular dictionary - use json.dumps
                json.dump(problem, f, indent=2)

def write_solution_folder(solutions, path=Path("data/solutions/baseline/")):
    path.mkdir(parents=True, exist_ok=True)

    for problem_id, solution in solutions.items():
        logging.info("=====================================================")
        logging.info(f"Saving solution to problem {problem_id}...")
        with open(path / f"{problem_id}.json", 'w') as f:
            # Handle both Pydantic objects and regular dictionaries
            if hasattr(solution, 'json'):
                # Pydantic v1 style
                f.write(solution.json())
            elif hasattr(solution, 'model_dump_json'):
                # Pydantic v2 style
                f.write(solution.model_dump_json())
            else:
                # Regular dictionary - use json.dumps
                json.dump(solution, f, indent=2)

def validate_solutions(problems, solutions):
    """
    Validates solutions by checking if they result in an empty string at the end of their transitions.
    """
    for problem_id in problems:
        logging.info("=====================================================")
        if problem_id not in solutions:
            logging.warning(f"Problem {problem_id} does not have a solution, skipping...")
            continue

        problem = problems[problem_id]
        solution = solutions[problem_id]

        transitions = problem.transitions
        current = problem.initial_string

        for step in solution.solution:
            if step >= len(transitions):
                logging.warning(f"Invalid step number {step} found! skipping problem...")
                break
            from_pattern = transitions[step].src
            to_pattern = transitions[step].tgt
            current = current.replace(from_pattern, to_pattern, 1)
            logging.info(f"Pattern: {from_pattern} -> {to_pattern}, String: {current}")

        if current != '':
            logging.warning(f"Problem {problem_id} has an invalid solution!")
        else:
            logging.info(f"Problem {problem_id} has a valid solution!")
