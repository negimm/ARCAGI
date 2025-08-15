import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np

# =================== Prompt ===================
BASE_PROMPT_OLD = r"""
You are an expert ARC (Abstraction and Reasoning Corpus) solver.
You infer discrete programmatic transformations between colored integer grids (0–9)
represented as lists of lists of integers. You MUST return safe, pure-Python code
that solves the given task by learning a transformation from the training pairs and
applying it to the test inputs.
check if cells are making a group which exists in ouput.
Think these problem as block puzzle.
find tranformation logic for each group.

Constraints:
- Representation: Grids are rectangular List[List[int]] with values 0–9.
- No external libraries. Use only Python standard library and built-ins.
- Deterministic and pure functions: no randomness, no mutation of inputs, no global state.
- Performance: Keep solutions simple; prefer O(n*m) scans.
- Safety: No `exec`, `eval`, `import`, `open`, or reflection.
- Return format: JSON with fields: "name", "explanation", "python".
- Implement a single function solve_task(train_pairs, test_inputs) -> List[Grid].

Task JSON will contain:
{
  "train_pairs": [ {"input": grid, "output": grid}, ... ],
  "test_inputs": [grid, ...]
}

Your job: infer the rule and return ONLY the JSON per schema above.
"""
BASE_PROMPT = r"""
You are an expert Abstraction and Reasoning Corpus (ARC) solver. Your sole purpose is to infer a discrete programmatic transformation from a set of grid pairs and apply that transformation to a test input. The grids are List[List[int]] containing integers from 0 to 9.

Mental Model & Deliberation
Approach each task with a methodical, step-by-step reasoning process. Do not rush to a conclusion. Think of the problem as a multi-stage puzzle where you must first identify the core components (objects, background, etc.) and then deduce the rules governing their manipulation. Take as much time as needed to be certain of the logic before writing the code.

Core Task & Constraints
-Representation: Grids are List[List[int]] with values 0–9.
-Code Purity: The solution must be a pure-Python function, deterministic, stateless, and should not mutate its inputs.
-Libraries: Use only Python's standard library and built-in functions. No external libraries are allowed.
-Efficiency: Keep solutions simple and efficient, preferring O(n*m) scans.
-Safety: Do not use exec, eval, import, or reflection.
-Output Format: Your final output must be a single JSON object with the following fields:
--"name": A descriptive title for the solution.
--"explanation": A detailed, step-by-step explanation of the inferred logic.
--"python": The complete, runnable Python code for the solve_task function.

Systematic Problem-Solving Strategy
When presented with a task, analyze the training pairs and follow this hierarchical strategy to find the rule.
1.Analyze Global Transformations (Grid Level):
-Tiling & Repetition: Is the output grid composed of repeated or tiled copies of the input grid? Are these copies identical or do they have minor modifications?
-Scaling: Is the output grid a scaled version of the input grid (e.g., 2x, 3x)? If so, what is the scaling factor and what is the rule for the new cells?
-Cropping/Slicing: Is the output grid a specific slice or sub-grid of the input grid?
-Rotation & Reflection: Is the entire grid rotated (90°, 180°, 270°) or reflected (horizontally, vertically)?
-Color-by-Position: Does the output grid's value depend on the row/column index of the input grid?
2.Analyze Object-Oriented Transformations (Group Level):
-Group Identification: Identify all contiguous groups of non-zero, same-colored cells in the input. Treat each group as a distinct "object" with properties like color, shape, and bounding box.
-Object Mapping: Determine how each input object corresponds to an output object. Was it moved, resized, or recolored?
-Movement:
--Translation: Did the object move to a new location? Is there a consistent offset (e.g., (x+1, y+1))?
--Boundary Rule: Did the object move to a corner, a border, or the center of the grid?
-Color & Value Manipulation:
--Recoloring: Did the object's color change? Is there a rule (e.g., old_color -> new_color)?
--Background Fill: Is the output grid filled with a single color, with objects overlaid?
--Color-by-Proximity: Does a cell's color depend on the color of its neighbors?
-Shape & Size Alteration:
--Growth/Shrinkage: Did the object expand or contract? Is the change based on surrounding cells?
--Object-to-Shape: Is the output a canonical shape (e.g., a square) replacing the input object?
--Inversion: Did a shape's color and background swap?

Example Inferences:
Task 00576224 (2x2 to 6x6):

Observation: The output grid is exactly three times the size of the input grid (2x2 input becomes a 6x6 output). This suggests a tiling or repetition pattern.

Rule Inference: The output grid is constructed from a 3x3 arrangement of 2x2 sub-grids. The sub-grids in the odd-numbered rows (0 and 2) of this 3x3 arrangement are identical copies of the original input grid. The sub-grids in the even-numbered row (1) are a 180-degree rotation of the original input grid. This pattern of "original, rotated, original" is repeated for each horizontal group of 2x2 blocks.

Task 007bbfb7 (3x3 to 9x9):

Observation: Similar to the previous task, the output grid is three times the size of the input. However, the pattern is not a simple tiling of the input grid.

Rule Inference: The output is an empty 9x9 grid, initially filled with zeros. The transformation is a form of "object-to-grid" mapping. For every non-zero cell in the original 3x3 input grid, a full copy of the original input grid is placed onto the output grid, with its top-left corner aligned with the non-zero cell's corresponding position in a 3x3 tiled structure (e.g., (r*3, c*3)).

Task ff805c23 (24x24 to 5x5):

Observation: This is a compression or abstraction task where a large input grid maps to a much smaller output grid. The input grid contains multiple, distinct contiguous shapes of various colors.

Rule Inference: The goal is to extract a specific shape from the input grid and make it the output. Based on the test case, the output is a shape made of the color 8, which resembles the letter 'H'. This implies the transformation is to find the object of a specific, pre-determined color (likely 8 for the test case) and return a new grid containing only that object and a background of 0.

Final Output
Infer the most general rule that applies to all train_pairs. Then, construct the final JSON object with your solution code. The code must implement the solve_task function, which accepts the training and testing data and returns the results for the test cases.
"""

Grid = List[List[int]]
printgrid=False

def pretty(g: Grid) -> str:
  if printgrid:
    return "\n".join(" ".join(str(v) for v in row) for row in g)
  else:
    return ""
# =================== LLM client (OpenAI v1) ===================
class LLMClient:
    def __init__(self, model: str = "deepseek-ai/DeepSeek-R1-Zero"):
        self.model = model
        self.client = OpenAI()

    def complete(self, prompt: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert ARC problem solver."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content

# =================== Safe execution ===================
class SafeEnv(dict):
    def __init__(self):
        super().__init__()
        self["__builtins__"] = {
            "range": range, "len": len, "min": min, "max": max,
            "sum": sum, "abs": abs, "enumerate": enumerate, "zip": zip,
            "all": all, "any": any, "sorted": sorted, "map": map,
            "filter": filter, "list": list, "tuple": tuple, "set": set, "dict": dict
        }

def compile_candidate(module_src: str):
    env = SafeEnv()
    try:
        code = compile(module_src, filename="<candidate>", mode="exec")
        exec(code, env, env)
    except Exception:
        return None
    return env.get("solve_task", None)

@dataclass
class Score:
    train_accuracy: float
    per_pair: List[bool]
    errors: List[str]

def score_candidate(solve_task, train_pairs: List[Dict[str, Grid]]) -> Score:
    inputs = [p["input"] for p in train_pairs]
    gold = [p["output"] for p in train_pairs]
    try:
        preds = solve_task(train_pairs, inputs)
    except Exception as e:
        return Score(0.0, [False]*len(train_pairs), [repr(e)])
    ok = []
    errs = []
    for i, (pred, ref) in enumerate(zip(preds, gold)):
        same = pred == ref
        ok.append(same)
        if not same:
            errs.append(f"Pair {i} mismatch:\nPred:\n{pretty(pred)}\nGold:\n{pretty(ref)}")
    return Score(sum(ok)/len(ok), ok, errs)

def build_initial_prompt(train_pairs, test_inputs):
    task_blob = json.dumps({"train_pairs": train_pairs, "test_inputs": test_inputs})
    return BASE_PROMPT + "\n### Task JSON\n" + task_blob + "\nReturn ONLY the JSON."

def build_critique_prompt(last_json: Dict[str, Any], critique: str, train_pairs, test_inputs):
    task_blob = json.dumps({"train_pairs": train_pairs, "test_inputs": test_inputs})
    return (
        BASE_PROMPT
        + "\n### Your last attempt (JSON):\n"
        + json.dumps(last_json)
        + "\n### Critique:\n"
        + critique
        + "\n### Task JSON\n"
        + task_blob
        + "\nRevise and return ONLY the JSON."
    )

def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last < first:
        return None
    try:
        return json.loads(s[first:last+1])
    except Exception:
        return None

def run_task(task_id, challenge, solution, attempts, model) -> Dict[str, Any]:
    train_pairs = [{"input": x["input"], "output": x["output"]} for x in challenge["train"]]
    test_inputs = [x["input"] for x in challenge["test"]]
    ground_truth = solution.get(task_id)
    count=0
    client = LLMClient(model)
    best = {"score": -1.0, "json": None, "preds": None}

    prompt = build_initial_prompt(train_pairs, test_inputs)
    last_json = None

    for t in range(1, attempts+1):
        print(f"\n--- Task {task_id} | Attempt {t}/{attempts} ---")
        raw = client.complete(prompt)
        j = try_parse_json(raw)
        if j is None:
            print("Parse fail")
            critique = "Your response was not valid JSON."
            prompt = build_critique_prompt(last_json or {}, critique, train_pairs, test_inputs)
            continue
        last_json = j
        solve_fn = compile_candidate(j.get("python", ""))
        if solve_fn is None:
            print("Compilation fail")
            critique = "Python failed to compile. Provide a self-contained module with solve_task(...)."
            prompt = build_critique_prompt(last_json, critique, train_pairs, test_inputs)
            continue
        sc = score_candidate(solve_fn, train_pairs)
        print(f"Train acc: {sc.train_accuracy:.2f}")
        if sc.train_accuracy >= best["score"]:
            try:
                test_preds = solve_fn(train_pairs, test_inputs)
            except Exception:
                test_preds = None
            best = {"score": sc.train_accuracy, "json": j, "preds": test_preds}
        if sc.train_accuracy == 1.0:
            break
        critique = "\n".join(sc.errors or ["Outputs incorrect, improve rule."])
        prompt = build_critique_prompt(last_json, critique, train_pairs, test_inputs)

    # Print comparison
    print(f"\n=== Best for Task {task_id} ===")
    print(f"Train accuracy: {best['score']:.2f}")
    if best["preds"] is not None:
        print("Predicted test output:")
        for grid in best["preds"]:
            print(pretty(grid)); print()
    if ground_truth:
        print("Ground truth test output:")
        for grid in ground_truth:
            print(pretty(grid)); print()
        print(f"Test match: {best['preds'] == ground_truth}")
    else:
        print("No ground-truth available for this task ID in solutions file.")

    # Return structured results (also useful for saving later)
    return {
        "task_id": task_id,
        "best_train_accuracy": best["score"],
        "best_json": best["json"],
        "test_predictions": best["preds"],
        "ground_truth": ground_truth,
        "test_match": (best["preds"] == ground_truth) if ground_truth is not None else None,
    }

def plot_grid(grid, ax, title):
    ax.imshow(np.array(grid), cmap='viridis', vmin=0, vmax=9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def display_task_results(results_list,challenges):
    count=0
    for task_result in results_list:
        task_id = task_result["task_id"]
        best_json = task_result["best_json"]
        test_predictions = task_result["test_predictions"]
        ground_truth = task_result["ground_truth"]
        test_match = task_result["test_match"]
        if(test_match):
            count+=1
        train_pairs = challenges[task_id]["train"]
        test_inputs = challenges[task_id]["test"]


        print(f"--- Task {task_id} ---")
        if best_json and "explanation" in best_json:
            print("LLM Explanation:")
            print(best_json["explanation"])
            print("-" * 20)

        # Display training pairs
        if train_pairs:
            print("Training Pairs:")
            fig, axes = plt.subplots(len(train_pairs), 2, figsize=(6, len(train_pairs) * 3))
            if len(train_pairs) == 1: # Handle single pair case
                axes = [axes] # Make it iterable
            for i, pair in enumerate(train_pairs):
                plot_grid(pair["input"], axes[i][0], f"Train {i+1} Input")
                plot_grid(pair["output"], axes[i][1], f"Train {i+1} Output")
            plt.tight_layout()
            plt.show()
            print("-" * 20)

        # Display test results
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        if test_inputs:
             plot_grid(test_inputs[0]['input'], axes[0], "Test Input") # Assuming one test input
        else:
             plot_grid([[]], axes[0], "Test Input (Not Available)") # Display empty grid with title


        if test_predictions is not None and test_predictions:
            plot_grid(test_predictions[0], axes[1], "Predicted Output") # Assuming one test output
        else:
             plot_grid([[]], axes[1], "Predicted Output (Not Available)") # Display empty grid with title


        if ground_truth is not None and ground_truth:
             plot_grid(ground_truth[0], axes[2], "Ground Truth") # Assuming one ground truth output
        else:
            plot_grid([[]], axes[2], "Ground Truth (Not Available)") # Display empty grid with title


        plt.tight_layout()
        plt.show()
        print("=" * 50)
    print(f"Succes Count: {count}")
    print(f"Total Count: {len(results_list)}")
    print(f"Accuracy: {count/len(results_list)}")
