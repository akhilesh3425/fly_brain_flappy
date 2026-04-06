---
title: Wildfire Incident Command
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
license: mit
---

# 🔥 Wildfire Incident Command — OpenEnv

> **Meta Hackathon 2026** | Multi-Agent Hierarchical RL Environment

A real-world wildfire suppression simulation built on the **ICS (Incident Command System)** used by agencies like CAL FIRE and NWCG. Agents coordinate Division Supervisors and Crew Bosses across a 20×20 dynamic fire grid.

---

## 🗺️ Environment Overview

Fire spreads dynamically each timestep based on wind direction and fuel probability. A hierarchical agent structure mirrors real field operations.

| Level | Role | Scope | Responsibility |
|-------|------|-------|----------------|
| Level 1 | Incident Commander | Entire map | Allocates divisions, sets overall strategy |
| Level 2 | Division Supervisor | One division zone | Directs crew bosses within their zone |
| Level 3 | Crew Boss | Local fire line | Direct suppression, firebreaks, movement |

**Grid:** 20×20 divided into 4 Division zones (A, B, C, D)

| Cell Value | State |
|------------|-------|
| `0` | Clear |
| `1` | Burning 🔥 |
| `2` | Burnt ⬛ |
| `3` | Firebreak 🟫 |

---

## 📁 File Structure

```
wildfire-incident-command/
├── app/
│   ├── __init__.py         # Package init
│   ├── env.py              # Core environment: reset(), step(), state()
│   ├── models.py           # Pydantic v2 request/response models
│   ├── tasks.py            # Task definitions + TaskGrader
│   ├── agent.py            # Rule-based baseline agent
│   └── main.py             # FastAPI app + all HTTP endpoints
├── static/
│   └── index.html          # Interactive dashboard frontend
├── inference.py            # LLM inference script (OpenAI-compatible)
├── test_env.py             # Smoke tests
├── openenv.yaml            # OpenEnv metadata (required)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container setup for HF Spaces
└── README.md               # This file
```

---

## 🚀 Quickstart

### Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Open in browser
open http://localhost:7860
# API docs at http://localhost:7860/docs
```

### Run with Docker

```bash
docker build -t wildfire-ics .
docker run -p 7860:7860 wildfire-ics
```

### Run smoke tests

```bash
python test_env.py
```

### Run LLM inference

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Run on Task 1 (easy)
python inference.py --task 1

# Run on Task 2 (medium)
python inference.py --task 2

# Run on Task 3 (hard)
python inference.py --task 3
```

---

## 🌐 API Reference

All endpoints follow the **OpenEnv HTTP API standard**.

### `POST /reset`
Start a new episode.

**Request body:**
```json
{ "task": 1 }
```
`task`: `1` (easy), `2` (medium), or `3` (hard)

**Response:**
```json
{
  "observation": {
    "grid": [[0, 0, ...], ...],
    "division_zones": [...],
    "crew_positions": [...],
    "wind": { "direction": "E", "speed": 0.3 },
    "assets_at_risk": [],
    "timestep": 0,
    "containment_pct": 0.0
  },
  "info": { "task": 1 }
}
```

---

### `POST /step`
Send an action to the environment.

**Request body:**
```json
{
  "division_id": 0,
  "crew_id": 0,
  "action_type": "SUPPRESS",
  "target_cell": [3, 3]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `division_id` | int | Division to command: `0`=A, `1`=B, `2`=C, `3`=D |
| `crew_id` | int | Crew boss within division: `0` or `1` |
| `action_type` | string | `SUPPRESS` \| `FIREBREAK` \| `MOVE` \| `HOLD` |
| `target_cell` | [int, int] | Grid coordinates `[row, col]` (0–19) |

**Action types:**

| Action | Effect |
|--------|--------|
| `SUPPRESS` | Extinguishes a burning cell (BURNING → BURNT) |
| `FIREBREAK` | Clears a cell to block spread (CLEAR → FIREBREAK) |
| `MOVE` | Repositions a crew boss |
| `HOLD` | No action (penalised if fire is nearby) |

**Response:**
```json
{
  "observation": { ... },
  "reward": 0.312,
  "done": false,
  "info": { "action": "SUPPRESS", "efficiency": 1.0, "containment_pct": 0.45 }
}
```

---

### `GET /state`
Returns full internal state snapshot (used by graders).

---

### `GET /tasks`
Returns task list with name, difficulty, and description.

---

### `GET /grader`
Returns deterministic task scores for the current episode state.

```json
{
  "task_scores": {
    "task_1": 0.87,
    "task_2": 0.61,
    "task_3": 0.44
  }
}
```

---

### `GET /baseline`
Runs the rule-based baseline agent on the current task and returns scores.

---

## 🎯 Tasks

### Task 1 — Single Front Containment `[EASY]` · 30 steps
Fire starts as a 5×5 cluster in Division A. Contain all fire within Division A.

- **Wind:** Fixed East, speed 0.3
- **Score:** `(cells_in_A / total_burning) × speed_bonus`
- **1.0** = Fire fully contained in A | **0.0** = Spread to 2+ other divisions

---

### Task 2 — Asset Protection `[MEDIUM]` · 50 steps
Fire starts on the north edge and spreads toward 3 named assets.

- **Assets:** Town `(15,10)`, Water Tower `(12,5)`, Road Junction `(14,15)`
- **Wind:** Fixed South, speed 0.5
- **Score:** `containment_ratio × (assets_saved / 3)`
- **1.0** = Contained + all 3 assets safe | **0.0** = All assets burned

---

### Task 3 — Multi-Front Outbreak `[HARD]` · 75 steps
Fire ignites simultaneously in Divisions A, B, and C. Coordinate all crews across the entire map.

- **Assets:** 2 per active division (6 total)
- **Wind:** Randomised each episode, speed 0.3–0.7
- **Score:** `avg_containment × speed_factor × asset_factor − merge_penalty`
- **1.0** = All fronts contained, assets safe, no merges

---

## 📊 Reward Function

Reward fires every step to provide dense learning signal.

```
reward = (0.4 × containment_delta)
       + (0.3 × asset_safety_ratio)
       + (0.2 × crew_efficiency)
       − (0.1 × idle_penalty)
       − (0.5 × assets_destroyed)
       − (0.8 × front_merge)
```

| Component | Weight | Trigger |
|-----------|--------|---------|
| `containment_delta` | +0.4 | Each step containment % increases |
| `asset_safety_ratio` | +0.3 | Each step based on assets still safe |
| `crew_efficiency` | +0.2 | Crew action directly reduces burning cells |
| `idle_penalty` | −0.1 | HOLD used when fire is nearby |
| `asset_destroyed` | −0.5 | Per asset burned this step |
| `front_merge_penalty` | −0.8 | Task 3 only: two fire fronts merge |

---

## 🤖 LLM Inference

`inference.py` connects to the HuggingFace Inference Router using an OpenAI-compatible client. It feeds the current observation to the LLM and parses the JSON action response, with a rule-based fallback if the model returns invalid output.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model to use |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | API endpoint |

---

## 🏗️ Build Order

| Step | What | Day |
|------|------|-----|
| 1 | `app/env.py` — grid, fire spread, ICS hierarchy | Day 1 |
| 2 | `app/models.py` — Pydantic models | Day 1 |
| 3 | `app/tasks.py` — task definitions + graders | Day 1 |
| 4 | `app/agent.py` — baseline agent | Day 1 |
| 5 | `app/main.py` — FastAPI endpoints | Day 1–2 |
| 6 | `static/index.html` — dashboard frontend | Day 2 |
| 7 | `inference.py` — LLM inference script | Day 2 |
| 8 | `test_env.py` — smoke tests | Day 2 |
| 9 | `Dockerfile` + HF Space deploy | Day 3 |
| 10 | `README.md` + `openenv.yaml` | Day 3 |

---

## 📄 License

BSD-3-Clause — Meta Hackathon 2026
