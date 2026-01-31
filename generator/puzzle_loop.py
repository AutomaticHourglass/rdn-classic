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

import json, pathlib, shlex, subprocess, shutil
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
    for k, v in d.items():
        if k in ['src', 'src_path', 'source_path', 'source']:
            k = 'src'
        if k in ['dst', 'dst_path', 'dest', 'dest_path', 'target', 'destination_path', 'destination']:
            k = 'dst'
        if k in ['path', 'filename', 'file','folder_name','folder']:
            k = 'path'
        r[k] = v

    return r

# ----------------------------------------------------------------------
# 2️⃣ Helper: call LLM endpoint
# ----------------------------------------------------------------------
import requests

def _call_llm(messages: List[Dict[str, str]], temperature: float = 0.5,
             max_tokens: int = 800) -> str:
    payload = {
        # "model": "openai/gpt-oss-20b",
        "mode": "qwen/qwen3-coder-30b",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    r = requests.post(
        "http://localhost:8901/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------------------------------------------------
# 3️⃣ Generator: yield puzzles on demand (relative paths only)
# ----------------------------------------------------------------------
def generate_puzzles(max_puzzles: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    counter = 0
    while max_puzzles is None or counter < max_puzzles:
        system_msg = {
            "role": "system",
            "content":
                ("You are a puzzle generator that creates short, clear file‑system puzzles. "
                 "All file paths must be relative to the workspace root and should not contain '..'. "
                 f"Use only the following tools: {', '.join(tool_commands.keys())}. "
                 "Return a JSON object with keys: description, initial_state (optional dict of filename→content), "
                 "goal_state (dict of filename→expected content), and solution (optional minimal tool call sequence).")
        }
        user_msg = {"role": "user", "content": f"Generate puzzle #{counter + 1}"}

        raw = _call_llm([system_msg, user_msg], temperature=0.6)

        try:
            puzzle = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[WARN] Puzzle #{counter+1} output not valid JSON:\n{raw}")

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
    apply_initial_state(workspace_dir, init_state)

    # 6.2 Execute each tool call
    for i, call in enumerate(solution):
        tool_name = call.get("type")
        if not tool_name:
            details.append(f"Tool call #{i+1} missing 'type'")
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
                timeout=10
            )
        except subprocess.TimeoutExpired:
            details.append(f"Tool call #{i+1} ({tool_name}) timed out")
            score = 0
            continue

        if result.returncode != 0:
            details.append(
                f"Tool call #{i+1} ({tool_name}) failed (rc={result.returncode})\n"
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

        actual_content = abs_path.read_text(encoding="utf-8")
        if actual_content != expected:
            details.append(
                f"File {rel_path} mismatch:\n"
                f"Expected: {expected}\nActual:   {actual_content}"
            )
            score = 0

    return score, "\n".join(details)

# ----------------------------------------------------------------------
# 7️⃣ Solver: ask LLM to solve a puzzle
# ----------------------------------------------------------------------
def solve_puzzle(puzzle: Dict[str, Any]) -> List[Dict[str, Any]]:
    system_msg = {
        "role": "system",
        "content":
            ("You are an agent that can use the following tools:\n"
             f"{', '.join(tool_commands.keys())}\n\n"
             "When solving a puzzle, output ONLY a JSON array of tool calls.\n"
             "Each call must be an object with 'type' (tool name) and the necessary arguments.\n"
             "Do not output any explanation or text – only the JSON.")
    }

    puzzle_desc = puzzle.get("description", "No description provided.")
    goal_desc = json.dumps(puzzle.get("goal_state", {}), indent=2)
    user_msg = {
        "role": "user",
        "content":
            f"Puzzle:\n{puzzle_desc}\nGoal state:\n{goal_desc}\n\nProvide the solution as a JSON array of tool calls."
    }

    raw = _call_llm([system_msg, user_msg], temperature=0.4)

    # Extract first JSON array from raw text
    try:
        solution = sanitize_puzzle(json.loads(raw))
    except json.JSONDecodeError:
        # Heuristic: find first '[' and last ']'
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                solution = json.loads(raw[start:end+1])
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
# 9️⃣ Main loop
# ----------------------------------------------------------------------
def main():
    NUM_PUZZLES = 5   # change as desired
    workspace_root = pathlib.Path("./puzzle_workspace")
    workspace_root.mkdir(exist_ok=True)

    for idx, puzzle in enumerate(generate_puzzles(NUM_PUZZLES), start=1):
        print(f"\n=== Puzzle #{idx} ===")
        print("Description:", puzzle.get("description", ""))
        print("\nGoal state:")
        print(json.dumps(puzzle.get("goal_state", {}), indent=2))

        # Solver
        solution = solve_puzzle(puzzle)
        if not solution:
            print("[FAIL] Solver produced no valid solution.")
            continue

        # Workspace for this puzzle
        puzzle_workspace = workspace_root / f"puzzle_{idx}"
        score, details = evaluate_solution(solution, puzzle, puzzle_workspace)

        print("\nEvaluation:")
        if score:
            print("✅  Success! All goal conditions met.")
            # TODO: save both puzzle and solution to a file
        else:
            print("❌  Failure:")
            print(details)
            # TODO: save puzzle to a file

        # Optional: show final file contents
        print_workspace(puzzle_workspace)

if __name__ == "__main__":
    main()
