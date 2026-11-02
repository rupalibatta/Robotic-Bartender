'''arm_planning_config.py

   Single source of truth for Chad arm planning: chain definition and
   waiting pose. Used by brain_node (planning -> SegmentArray) and
   touchtable_bottles (planning + execution in one node).
'''

import numpy as np
from math import pi

# 3-DOF arm: base, shoulder, elbow. Tip frame must match URDF so the chain has exactly these joints.
ARM_JOINTNAMES = ['base', 'shoulder', 'elbow']
CHAIN_BASE_FRAME = 'world'
CHAIN_TIP_FRAME = 'lowerarm'  # use 'lowerarm' for 3-DOF; 'tip' would include wrist in URDF

# Waiting pose (rad): base 0, shoulder 0, elbow -90° (elbow up)
WAITING_ARM = np.array([0.0, 0.0, -pi / 2], dtype=float)

# Segment duration (seconds)
T_MOVE = 2.0

# IK (used by both nodes when computing touch poses)
IK_TOL = 1e-4
IK_MAX_ITERS = 50
IK_MU = 0.02
TOUCH_Z = 0.0
