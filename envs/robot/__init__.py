import os

from .robot import Robot

if os.environ.get("ROBOTWIN_DISABLE_PLANNER", "").strip().lower() not in {"1", "true", "yes", "on"}:
    from .planner import *
