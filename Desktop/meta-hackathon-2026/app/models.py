from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union, Tuple


class Action(BaseModel):
    division_id: int                          # 0=A, 1=B, 2=C, 3=D
    crew_id: int                              # 0 or 1 within that division
    action_type: str                          # SUPPRESS | FIREBREAK | MOVE | HOLD
    target_cell: List[int]                    # [row, col]


class ResetRequest(BaseModel):
    task: Optional[int] = 1                   # 1, 2, or 3


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}


class StateResponse(BaseModel):
    state: Dict[str, Any]


class TaskResponse(BaseModel):
    name: str
    difficulty: str
    description: str


class GraderResponse(BaseModel):
    task_scores: Dict[str, float]


class BaselineResponse(BaseModel):
    task_scores: Dict[str, float]
    total_reward: float
    steps_taken: int
