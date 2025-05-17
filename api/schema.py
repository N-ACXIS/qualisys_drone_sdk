from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict


class TargetRequest(BaseModel):
    target_x: float
    target_y: float
    target_z: float
    cur_x: float
    cur_y: float
    cur_z: float
    rot_mat: List[List[float]]  # 3x3行列


class ControlStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"


class TargetResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: float
    y: float
    z: float
    yaw: float
    status: ControlStatus
