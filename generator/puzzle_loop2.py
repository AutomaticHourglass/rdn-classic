#!/usr/bin/env python3
"""
puzzle_loop.py

A perpetual system that:
 1️⃣ Generates puzzles on demand (relative paths only).
 2️⃣ Sends each puzzle to the LLM (tool‑enabled) to get a solution.
 3️⃣ Creates the initial_state files on disk (via pathlib/os).
 4️⃣ Executes the tool calls in a dedicated workspace directory.
 5️⃣ Verifies that the final state matches the puzzle’s goal.

The script actually creates, renames, updates files on disk (within a
dedicated workspace), so you can inspect the changes yourself.
"""

import json, pathlib, shlex, subprocess, shutil, hashlib
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import random

from fuzzywuzzy import fuzz
from json_repair import json_repair

# ----------------------------------------------------------------------
# 1️⃣ Load tool command mapping
# ----------------------------------------------------------------------
TOOL_CMDS_PATH = pathlib.Path(__file__).parent / "tool_commands.json"

try:
    with TOOL_CMDS_PATH.open("r", encoding="utf-8") as f:
        tool_commands: Dict[str, Any] = json.load(f)
except FileNotFoundError:
    print(f"[ERROR] {TOOL_CMDS_PATH} not found. Exiting.")
    exit(1)


def sanitize_puzzle(d):
    if isinstance(d, list):
        return [sanitize_puzzle(i) for i in d]
    r = {}
    if isinstance(d, str):
        return {}
    for k, v in d.items():
        if k in ['src', 'src_path', 'source_path', 'source']:
            k = 'src'
        if k in ['dst', 'dst_path', 'dest', 'dest_path', 'target', 'destination_path', 'destination']:
            k = 'dst'
        if k in ['path', 'filename', 'file', 'folder_name', 'folder']:
            k = 'path'
        r[k] = v

    return r


# ----------------------------------------------------------------------
# 2️⃣ Helper: call LLM endpoint
# ----------------------------------------------------------------------
import requests


def _call_llm(messages: dict, seed: Optional[int] = None) -> str:
    payload = {
        "model": "qwen3-coder:30b",  # Ensure this model name matches your Ollama list exactly
        # "model": "gpt-oss:20b",  # Ensure this model name matches your Ollama list exactly
        "prompt": json.dumps(messages),
        "stream": False,
    }

    if seed is not None:
        payload["seed"] = seed

    data = None

    try:
        r = requests.post(
            # FIX 1: Add http:// protocol
            # FIX 2: Use the OpenAI-compatible endpoint to support 'messages' and 'choices'
            # "http://163.5.212.83:62667/api/generate/",
            "http://212.85.84.41:53569/api/generate/",
            json=payload,
            timeout=10,
        )
        try:
            r.raise_for_status()
        except:
            return ""
        data = r.json()
    except:
        time.sleep(1)


    # This line works because /v1/chat/completions returns the OpenAI format
    # return data["choices"][0]["message"]["content"]
    if data is not None:
        return data['response']
    return ""


# ----------------------------------------------------------------------
# 3️⃣ Generator: yield puzzles on demand (relative paths only)
# ----------------------------------------------------------------------
def generate_puzzles(max_puzzles: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    counter = 0
    while max_puzzles is None or counter < max_puzzles:
        cur_tools = random.sample(sorted(tool_commands), 5)
        tool_desc = json.dumps({k:tool_commands[k] for k in cur_tools})
        system_msg = {
            "role": "system",
            "content":
                ("You are a puzzle generator that creates short, clear file‑system puzzles. "
                 "All file paths must be relative to the workspace root and should not contain '..'. "
                 f"Use only the following tools: {tool_desc}. "
                 "Return a JSON object with keys: description, initial_state (optional dict of filename→content), "
                 "goal_state (dict of filename→expected content), and solution (optional minimal tool call sequence).")
        }
        raw = _call_llm(system_msg)

        if len(raw) <= 1:
            continue

        try:
            puzzle = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[WARN] Puzzle #{counter + 1} output not valid JSON:\n{raw}")

            puzzle = json_repair.loads(raw)

        yield sanitize_puzzle(puzzle)
        counter += 1


# ----------------------------------------------------------------------
# 4️⃣ Format a tool command from the mapping
# ----------------------------------------------------------------------
def format_tool_command(tool_name: str, args: Dict[str, Any]) -> str:
    if tool_name not in tool_commands:
        raise ValueError(f"Unknown tool: {tool_name}")

    template = tool_commands[tool_name]["command_template"]
    safe_args = {}
    for k, v in args.items():
        if isinstance(v, str):
            safe_args[k] = shlex.quote(v)
        else:
            safe_args[k] = str(v)

    try:
        cmd = template.format(**safe_args)
    except KeyError as e:
        raise ValueError(f"Missing argument for tool '{tool_name}': {e}")

    return cmd


# ----------------------------------------------------------------------
# 5️⃣ Apply the initial_state dict to a workspace directory
# ----------------------------------------------------------------------
def apply_initial_state(workspace_dir: pathlib.Path, initial_state: Dict[str, str]) -> None:
    """
    Create all files listed in `initial_state` inside `workspace_dir`.
    Each key is a relative path; the value is the file content.
    """
    for rel_path, content in initial_state.items():
        abs_path = workspace_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")


# ----------------------------------------------------------------------
# 6️⃣ Evaluate a solver's solution (executed in workspace)
# ----------------------------------------------------------------------
def evaluate_solution(solution: List[Dict[str, Any]], puzzle: Dict[str, Any],
                      workspace_dir: pathlib.Path) -> Tuple[int, str]:
    score = 1
    details: List[str] = []

    # Create workspace (clean any existing)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 6.1 Write initial state files
    init_state = puzzle.get("initial_state", {})
    try:
        apply_initial_state(workspace_dir, init_state)
    except:
        return 0, ""

    # 6.2 Execute each tool call
    for i, call in enumerate(solution):
        tool_name = call.get("type")
        if not tool_name:
            details.append(f"Tool call #{i + 1} missing 'type'")
            score = 0
            continue

        args = {k: v for k, v in call.items() if k != "type"}
        try:
            cmd = format_tool_command(tool_name, args)
        except Exception as e:
            details.append(f"Error formatting command for tool '{tool_name}': {e}")
            score = 0
            continue

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
        except subprocess.TimeoutExpired:
            details.append(f"Tool call #{i + 1} ({tool_name}) timed out")
            score = 0
            continue

        if result.returncode != 0:
            details.append(
                f"Tool call #{i + 1} ({tool_name}) failed (rc={result.returncode})\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            score = 0

    # 6.3 Verify goal state
    goal_state: Dict[str, Any] = puzzle.get("goal_state", {})
    for rel_path, expected in goal_state.items():
        abs_path = workspace_dir / rel_path
        if not abs_path.exists():
            details.append(f"Expected file {rel_path} does NOT exist")
            score = 0
            continue

        try:
            actual_content = abs_path.read_text(encoding="utf-8")
            if fuzz.partial_ratio(expected, actual_content) < 98:
                details.append(
                    f"File {rel_path} mismatch:\n"
                    f"Expected: {expected}\nActual: {actual_content}"
                )
                score = 0

        except:
            score = 0

    return score, "\n".join(details)


# ----------------------------------------------------------------------
# 7️⃣ Solver: ask LLM to solve a puzzle
# ----------------------------------------------------------------------
def solve_puzzle(puzzle: Dict[str, Any], seed: Optional[int] = None, attempt=0, fails=None) -> List[Dict[str, Any]]:
    system_msg = {
        "role": "system",
        "content":
            ("You are an agent that can use the following tools:\n"
             f"{json.dumps(tool_commands)}"
             "When solving a puzzle, output ONLY a JSON array of tool calls.\n"
             "Each call must be an object with 'type' (tool name) and the necessary arguments.\n"
             "Do not output any explanation or text – only the JSON.")
    }

    if fails:
        for fail in fails:
            system_msg['content'] += f"\n\nFail:\n{fail}\n\n"

    puzzle_desc = puzzle.get("description", "No description provided.")
    goal_desc = puzzle.get("goal_state", {})

    raw = _call_llm({'system_msg': system_msg, 'description': puzzle_desc, 'goal': goal_desc}, seed=seed)

    # Extract first JSON array from raw text
    try:
        solution = sanitize_puzzle(json.loads(raw))
    except json.JSONDecodeError:
        # Heuristic: find first '[' and last ']'
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                solution = json.loads(raw[start:end + 1])
                solution = sanitize_puzzle(solution)
            except Exception as e:
                print(f"[ERROR] Could not parse solution JSON: {e}")
                return []
        else:
            print("[WARN] No JSON array found in solver output.")
            return []

    if not isinstance(solution, list):
        print("[WARN] Solver did not return a JSON array.")
        return []

    return solution


# ----------------------------------------------------------------------
# 8️⃣ Helper: print workspace contents (for debugging)
# ----------------------------------------------------------------------
def print_workspace(workspace_dir: pathlib.Path):
    print(f"\nContents of {workspace_dir}:")
    for path in workspace_dir.rglob("*"):
        if path.is_file():
            print(f"  {path.relative_to(workspace_dir)}")
            try:
                content = path.read_text(encoding="utf-8")
                print(f"    (content) {repr(content[:80])}…")
            except Exception:
                pass


# ----------------------------------------------------------------------
# 9️⃣ Save puzzle and solution to disk
# ----------------------------------------------------------------------
def save_puzzle_and_solution(puzzle: Dict[str, Any], solution: List[Dict[str, Any]],
                             puzzle_dir: pathlib.Path) -> None:
    """Save puzzle and solution to the given directory."""
    # Create puzzle directory if it doesn't exist
    puzzle_dir.mkdir(parents=True, exist_ok=True)

    # Save puzzle as JSON
    puzzle_file = puzzle_dir / "puzzle.json"
    with open(puzzle_file, "w", encoding="utf-8") as f:
        json.dump(puzzle, f, indent=2, ensure_ascii=False)

    # Save solution as JSON
    solution_file = puzzle_dir / "solution.json"
    with open(solution_file, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# 10️⃣ Main loop
# ----------------------------------------------------------------------
def main():
    NUM_PUZZLES = None  # change as desired
    workspace_root = pathlib.Path("./puzzle_workspace")
    workspace_root.mkdir(exist_ok=True)

    # Directory to save valid puzzles
    puzzles_dir = pathlib.Path("./saved_puzzles")
    puzzles_dir.mkdir(exist_ok=True)

    for idx, puzzle in enumerate(generate_puzzles(NUM_PUZZLES), start=1):
        if isinstance(puzzle, list):
            continue
        print(f"\n=== Puzzle #{idx} ===")
        print("Description:", puzzle.get("description", ""))
        print("\nGoal state:")
        print(json.dumps(puzzle.get("goal_state", {}), indent=2))

        # Create hash from puzzle content to avoid collisions
        puzzle_content = json.dumps(puzzle, sort_keys=True)
        puzzle_hash = hashlib.md5(puzzle_content.encode()).hexdigest()[:8]

        # Create unique directory for this puzzle
        puzzle_dir = puzzles_dir / f"puzzle_{puzzle_hash}"

        # Try solving with different seeds until we get a valid solution
        max_attempts = 5
        success = False
        attempt = 0
        fails = []

        while not success and attempt < max_attempts:
            # Use a random seed for retrying
            seed = random.randint(0, 1000000)

            # Solver
            solution = solve_puzzle(puzzle, seed=seed, attempt=attempt, fails=fails)
            if not solution:
                print(f"[FAIL] Solver produced no valid solution (attempt {attempt + 1}).")
                attempt += 1
                continue

            # Workspace for this puzzle
            puzzle_workspace = workspace_root / f"puzzle_{idx}_attempt_{attempt + 1}"
            score, details = evaluate_solution(solution, puzzle, puzzle_workspace)

            print("\nEvaluation:")
            if score:
                print("✅  Success! All goal conditions met.")

                # Save puzzle and solution to disk
                save_puzzle_and_solution(puzzle, solution, puzzle_dir)

                success = True
            else:
                print("❌  Failure:")
                print(details)
                fails += ["Failed trial:\n\n" + json.dumps(solution) + '\n\n' + details + '-'*80 + '\n\n']

                if puzzle_workspace.exists():
                    try:
                        shutil.rmtree(puzzle_workspace)
                    except:
                        continue

                attempt += 1


        # Clean up workspace directory if it exists (even for failed attempts)
        puzzle_workspace = workspace_root / f"puzzle_{idx}_attempt_{attempt}"
        if puzzle_workspace.exists():
            try:
                shutil.rmtree(puzzle_workspace)
            except:
                continue


if __name__ == "__main__":
    main()
