from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.env import WildfireEnv
from app.agent import BaselineAgent
from app.tasks import TASKS, TaskGrader
from app.models import (
    Action, ResetRequest, StepResponse, ResetResponse,
    StateResponse, TaskResponse, GraderResponse, BaselineResponse,
)
from typing import List
import os

DESCRIPTION = """
## Wildfire Incident Command — OpenEnv HTTP API

Simulates real-world ICS (Incident Command System) wildfire suppression on a 20×20 grid
with hierarchical agents (Division Supervisors + Crew Bosses).

### Endpoints
- **POST /reset** — Start a new episode (optionally pass `task`: 1, 2, or 3)
- **POST /step** — Send an action and get observation + reward
- **GET /state** — Inspect full internal state
- **GET /tasks** — List all tasks
- **GET /grader** — Get deterministic task scores
- **GET /baseline** — Run rule-based baseline agent

### Workflow
1. `POST /reset` with `{"task": 1}` to start
2. `POST /step` repeatedly with actions
3. Episode ends when `done: true`
4. `GET /grader` to score the episode

[OpenEnv Hackathon 2026] | BSD-3-Clause
"""

app = FastAPI(
    title="Wildfire Incident Command — OpenEnv",
    version="1.0.0",
    description=DESCRIPTION,
)

# Serve static frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global environment instance
env = WildfireEnv()


@app.get("/", include_in_schema=False)
def serve_frontend():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Wildfire ICS OpenEnv API is running. Visit /docs for the API."}


@app.api_route("/reset", methods=["GET", "POST"], response_model=ResetResponse)
def reset_env(body: ResetRequest = None):
    task = (body.task if body else None) or 1
    obs = env.reset(task=task)
    return {"observation": obs, "info": {"task": task}}


@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    action_dict = action.model_dump()
    obs, reward, done, info = env.step(action_dict)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state", response_model=StateResponse)
def get_state():
    return {"state": env.state()}


@app.get("/tasks", response_model=List[TaskResponse])
def list_tasks():
    return [
        {"name": t.name, "difficulty": t.difficulty, "description": t.description}
        for t in TASKS
    ]


@app.get("/grader", response_model=GraderResponse)
def get_grader_scores():
    grader = TaskGrader(env)
    return {
        "task_scores": {
            "task_1": grader.grade_task_1(),
            "task_2": grader.grade_task_2(),
            "task_3": grader.grade_task_3(),
        }
    }


@app.get("/baseline", response_model=BaselineResponse)
def run_baseline():
    """Run the rule-based baseline agent on the current task and return scores."""
    current_task = env.task if hasattr(env, "task") else 1
    env.reset(task=current_task)
    agent = BaselineAgent(env)
    total_reward = 0.0
    steps = 0

    while not env.done:
        obs = env._get_obs()
        action = agent.select_action(obs)
        _, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

    grader = TaskGrader(env)
    return {
        "task_scores": {
            "task_1": grader.grade_task_1(),
            "task_2": grader.grade_task_2(),
            "task_3": grader.grade_task_3(),
        },
        "total_reward": round(total_reward, 4),
        "steps_taken": steps,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
