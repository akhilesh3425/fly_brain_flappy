"""
inference.py — LLM-based agent for the Wildfire ICS OpenEnv environment.

Uses the HuggingFace Inference Router (OpenAI-compatible API) to drive
a language model through the wildfire suppression tasks.

Usage:
    python inference.py [--task 1|2|3]
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from app.env import WildfireEnv
from app.tasks import TaskGrader

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def get_action_from_llm(obs: dict) -> dict:
    """
    Ask the LLM to choose a fire suppression action based on the current observation.
    Returns a dict with division_id, crew_id, action_type, target_cell.
    """
    # Summarise grid to save tokens — just count burning cells per division
    grid = obs["grid"]
    division_summary = {}
    for zone_info in obs["division_zones"]:
        label = zone_info["label"]
        burning = sum(
            1 for r in range(zone_info["row_start"], zone_info["row_end"] + 1)
            for c in range(zone_info["col_start"], zone_info["col_end"] + 1)
            if grid[r][c] == 1
        )
        division_summary[label] = {"id": zone_info["id"], "burning_cells": burning}

    # Find a burning cell to target (first one found in busiest division)
    busiest = max(division_summary.items(), key=lambda x: x[1]["burning_cells"])
    busiest_div_id = busiest[1]["id"]
    busiest_label = busiest[0]
    zone_meta = next(z for z in obs["division_zones"] if z["label"] == busiest_label)
    target_cell = [zone_meta["row_start"] + 2, zone_meta["col_start"] + 2]
    for r in range(zone_meta["row_start"], zone_meta["row_end"] + 1):
        for c in range(zone_meta["col_start"], zone_meta["col_end"] + 1):
            if grid[r][c] == 1:
                target_cell = [r, c]
                break
        else:
            continue
        break

    prompt = f"""You are an Incident Commander for a wildfire suppression operation.
Current timestep: {obs['timestep']}
Containment: {obs['containment_pct']*100:.1f}%
Wind: {obs['wind']['direction']} at speed {obs['wind']['speed']}

Division fire summary (burning cell counts):
{json.dumps(division_summary, indent=2)}

Assets at risk:
{json.dumps([{a['name']: a['status']} for a in obs.get('assets_at_risk', [])], indent=2)}

Choose ONE action to direct a crew boss. Respond ONLY with a JSON object.

JSON Keys:
- "division_id": integer 0-3 (0=A, 1=B, 2=C, 3=D)
- "crew_id": integer 0 or 1 (two crews per division)
- "action_type": "SUPPRESS", "FIREBREAK", "MOVE", or "HOLD"
- "target_cell": [row, col] (integers 0-19)

Busiest division is {busiest_label} (id={busiest_div_id}) — suggested target: {target_cell}

Reply with ONLY the JSON object:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:].strip()
        text = text.strip()

        action = json.loads(text)
        # Validate required keys
        for key in ["division_id", "crew_id", "action_type", "target_cell"]:
            if key not in action:
                raise ValueError(f"Missing key: {key}")
        return action

    except Exception as e:
        print(f"  [LLM fallback] {e}")
        # Fallback: suppress nearest burning cell in busiest division
        return {
            "division_id": busiest_div_id,
            "crew_id": 0,
            "action_type": "SUPPRESS",
            "target_cell": target_cell,
        }


def run_inference(task: int = 1, max_steps: int = None):
    print(f"\n{'='*50}")
    print(f"  Wildfire ICS — LLM Inference  |  Task {task}")
    print(f"{'='*50}")

    env = WildfireEnv()
    obs = env.reset(task=task)
    print(f"Environment reset. Max steps: {env.max_steps}")

    max_steps = max_steps or env.max_steps
    steps = 0
    total_reward = 0.0

    while not env.done and steps < max_steps:
        print(f"\n--- Step {steps + 1} | Containment: {obs['containment_pct']*100:.1f}% ---")
        action = get_action_from_llm(obs)
        print(f"  Action: {action}")

        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Reward: {reward:.4f} | Done: {done}")
        if "error" in info:
            print(f"  Error: {info['error']}")
        steps += 1

    print(f"\n{'='*50}")
    print(f"Finished in {steps} steps. Total reward: {total_reward:.4f}")

    grader = TaskGrader(env)
    scores = {
        "task_1": grader.grade_task_1(),
        "task_2": grader.grade_task_2(),
        "task_3": grader.grade_task_3(),
    }
    print(f"Final Task Scores: {json.dumps(scores, indent=2)}")
    print("="*50)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM inference on Wildfire ICS OpenEnv")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3],
                        help="Task number (1=easy, 2=medium, 3=hard)")
    args = parser.parse_args()
    run_inference(task=args.task)
    sys.exit(0)
