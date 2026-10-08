from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "configs" / "push_scene.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)


# -------------------------
# Recorded manual push poses
# -------------------------

q_start = np.array([
     0.000,
     0.0353,
     0.319,
    -1.840,
    -0.203,
     1.870,
    -0.785
])

q_end = np.array([
    -0.724,
     0.0353,
     0.319,
    -1.840,
    -0.203,
     1.870,
    -0.785
])

gripper_ctrl = 121


# -------------------------
# Find Panda joints
# -------------------------

arm_joint_ids = [
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"panda_joint{i}"
    )
    for i in range(1, 8)
]

arm_qpos_addresses = [
    model.jnt_qposadr[joint_id]
    for joint_id in arm_joint_ids
]


# Start robot directly at q_start
for address, value in zip(arm_qpos_addresses, q_start):
    data.qpos[address] = value


# Open gripper
for name in ["panda_finger_joint1", "panda_finger_joint2"]:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    address = model.jnt_qposadr[joint_id]
    data.qpos[address] = 0.04


# -------------------------
# Find actuators
# -------------------------

arm_actuator_ids = [
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"panda_actuator{i}"
    )
    for i in range(1, 8)
]

gripper_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "panda_actuator8"
)


# Initial controller targets
data.ctrl[arm_actuator_ids] = q_start
data.ctrl[gripper_actuator_id] = 121


mujoco.mj_forward(model, data)


# Box position, so we can prove it moved
box_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "box"
)

start_box_position = data.xpos[box_id].copy()

print("Box start position:", start_box_position)


# -------------------------
# Run simulation
# -------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Let you see starting configuration
    time.sleep(2)

    push_duration = 3.0
    elapsed = 0.0

    while viewer.is_running() and elapsed < push_duration:

        step_start = time.time()

        alpha = elapsed / push_duration

        # Smooth start and stop
        smooth_alpha = alpha * alpha * (3 - 2 * alpha)

        q_target = (
            q_start
            + smooth_alpha * (q_end - q_start)
        )

        data.ctrl[arm_actuator_ids] = q_target
        data.ctrl[gripper_actuator_id] = 121

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed += model.opt.timestep

        sleep_time = (
            model.opt.timestep
            - (time.time() - step_start)
        )

        if sleep_time > 0:
            time.sleep(sleep_time)


    # Hold final pose briefly
    hold_time = 2.0
    elapsed = 0.0

    while viewer.is_running() and elapsed < hold_time:

        step_start = time.time()

        data.ctrl[arm_actuator_ids] = q_end
        data.ctrl[gripper_actuator_id] = 121

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed += model.opt.timestep

        sleep_time = (
            model.opt.timestep
            - (time.time() - step_start)
        )

        if sleep_time > 0:
            time.sleep(sleep_time)


    end_box_position = data.xpos[box_id].copy()

    print("Box end position:", end_box_position)
    print(
        "Box displacement:",
        end_box_position - start_box_position
    )

    time.sleep(2)