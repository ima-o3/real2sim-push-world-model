from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "configs" / "push_scene.xml"
CSV_PATH = ROOT / "data" / "processed" / "pilots" / "P01_center.csv"


# =========================================================
# Existing successful Panda push configuration
# =========================================================

Q_START = np.array([
    0.000,
    0.0353,
    0.319,
    -1.840,
    -0.203,
    1.870,
    -0.785,
])

Q_END_FULL = np.array([
    -0.724,
    0.0353,
    0.319,
    -1.840,
    -0.203,
    1.870,
    -0.785,
])

GRIPPER_TARGET = 121.0


# =========================================================
# Extract real-world action from tracked video
# =========================================================

def extract_real_action(csv_path):

    df = pd.read_csv(csv_path)

    # Fill small tracking gaps.
    df = df.interpolate(
        limit_direction="both"
    )

    t = df["time_s"].to_numpy()

    # Smooth tracking noise slightly.
    box = (
        df[["box_x_mm", "box_y_mm"]]
        .rolling(
            7,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    pusher = (
        df[["pusher_x_mm", "pusher_y_mm"]]
        .rolling(
            7,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    dt = np.median(
        np.diff(t)
    )

    # -----------------------------------------------------
    # Detect when the box is actually moving
    # -----------------------------------------------------

    box_speed = np.zeros(
        len(box)
    )

    box_speed[1:] = (
        np.linalg.norm(
            np.diff(
                box,
                axis=0,
            ),
            axis=1,
        )
        / dt
    )

    # Threshold in mm/s.
    raw_active = (
        box_speed > 20.0
    )

    # Smooth the active/inactive classification.
    active = (
        pd.Series(
            raw_active.astype(float)
        )
        .rolling(
            9,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
        > 0.35
    )

    # -----------------------------------------------------
    # Find contiguous motion events
    # -----------------------------------------------------

    groups = []

    start = None

    for i, is_active in enumerate(active):

        if (
            is_active
            and start is None
        ):
            start = i

        ending = (
            start is not None
            and (
                not is_active
                or i == len(active) - 1
            )
        )

        if ending:

            end = (
                i
                if is_active
                else i - 1
            )

            # Ignore tiny noisy events.
            if end - start >= 10:

                motion_score = np.sum(
                    box_speed[
                        start:end + 1
                    ]
                )

                groups.append(
                    (
                        start,
                        end,
                        motion_score,
                    )
                )

            start = None

    if not groups:

        raise RuntimeError(
            "Could not identify "
            "the main push."
        )

    # Choose the largest box-motion event.
    start, end, _ = max(
        groups,
        key=lambda g: g[2],
    )

    # -----------------------------------------------------
    # Extract action + outcome
    # -----------------------------------------------------

    pusher_delta = (
        pusher[end]
        - pusher[start]
    )

    pusher_distance_mm = (
        np.linalg.norm(
            pusher_delta
        )
    )

    duration = (
        t[end]
        - t[start]
    )

    box_delta = (
        box[end]
        - box[start]
    )

    box_distance_mm = (
        np.linalg.norm(
            box_delta
        )
    )

    print(
        "\nREAL PILOT ACTION"
    )

    print(
        "-----------------"
    )

    print(
        f"Detected frames: "
        f"{start} -> {end}"
    )

    print(
        f"Duration: "
        f"{duration:.3f} s"
    )

    print(
        f"Pusher dx: "
        f"{pusher_delta[0]:.1f} mm"
    )

    print(
        f"Pusher dy: "
        f"{pusher_delta[1]:.1f} mm"
    )

    print(
        f"Pusher distance: "
        f"{pusher_distance_mm:.1f} mm"
    )

    print(
        f"Box displacement during push: "
        f"{box_distance_mm:.1f} mm"
    )

    return (
        pusher_distance_mm / 1000.0,
        duration,
        box_distance_mm,
    )


# =========================================================
# MuJoCo utility functions
# =========================================================

def find_ee(model, data):

    # Prefer a site if available.
    site_candidates = [
        "attachment_site",
        "grasp_site",
        "ee_site",
    ]

    for name in site_candidates:

        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if site_id >= 0:

            print(
                f"Using EE site: "
                f"{name}"
            )

            return (
                "site",
                site_id,
            )

    # Otherwise use the Panda hand body.
    body_candidates = [
        "hand",
        "panda_hand",
    ]

    for name in body_candidates:

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        if body_id >= 0:

            print(
                f"Using EE body: "
                f"{name}"
            )

            return (
                "body",
                body_id,
            )

    raise RuntimeError(
        "Could not find Panda "
        "end-effector site/body."
    )


def ee_position(data, ee):

    kind, idx = ee

    if kind == "site":

        return (
            data.site_xpos[idx]
            .copy()
        )

    return (
        data.xpos[idx]
        .copy()
    )


def find_box_body(model):

    candidates = [
        "cube",
        "box",
        "push_box",
        "object",
    ]

    for name in candidates:

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        if body_id >= 0:

            print(
                f"Using object body: "
                f"{name}"
            )

            return body_id

    raise RuntimeError(
        "Could not automatically "
        "find cube body."
    )


# =========================================================
# Read real pilot
# =========================================================

(
    real_distance_m,
    real_duration,
    real_box_distance_mm,
) = extract_real_action(
    CSV_PATH
)


# =========================================================
# Load MuJoCo model
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


# =========================================================
# Locate Panda joints
# =========================================================

joint_ids = [
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"panda_joint{i}",
    )
    for i in range(1, 8)
]

if any(
    joint_id < 0
    for joint_id in joint_ids
):
    raise RuntimeError(
        "Could not find all "
        "Panda joints."
    )


qpos_addresses = [
    model.jnt_qposadr[j]
    for j in joint_ids
]


# =========================================================
# Locate Panda actuators
# =========================================================

arm_actuator_ids = [
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"panda_actuator{i}",
    )
    for i in range(1, 8)
]

if any(
    actuator_id < 0
    for actuator_id
    in arm_actuator_ids
):
    raise RuntimeError(
        "Could not find all "
        "Panda arm actuators."
    )


gripper_actuator_id = (
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "panda_actuator8",
    )
)

if gripper_actuator_id < 0:

    raise RuntimeError(
        "Could not find "
        "Panda gripper actuator."
    )


# =========================================================
# Initialise robot at known start pose
# =========================================================

for adr, value in zip(
    qpos_addresses,
    Q_START,
):

    data.qpos[adr] = value


mujoco.mj_forward(
    model,
    data,
)


# =========================================================
# Find EE and object
# =========================================================

ee = find_ee(
    model,
    data,
)

box_body_id = find_box_body(
    model
)


# =========================================================
# Measure the geometric EE path of the original
# successful joint-1 sweep
# =========================================================

sample_q1 = np.linspace(
    Q_START[0],
    Q_END_FULL[0],
    500,
)

positions = []


for q1 in sample_q1:

    data.qpos[
        qpos_addresses[0]
    ] = q1

    for i in range(
        1,
        7,
    ):

        data.qpos[
            qpos_addresses[i]
        ] = Q_START[i]

    mujoco.mj_forward(
        model,
        data,
    )

    positions.append(
        ee_position(
            data,
            ee,
        )
    )


positions = np.array(
    positions
)


# Only measure horizontal XY motion.
segment_lengths = np.linalg.norm(
    np.diff(
        positions[:, :2],
        axis=0,
    ),
    axis=1,
)

cumulative_path = np.concatenate(
    [
        [0.0],
        np.cumsum(
            segment_lengths
        ),
    ]
)

full_path_length = (
    cumulative_path[-1]
)


print(
    "\nSIMULATION REFERENCE"
)

print(
    "--------------------"
)

print(
    f"Full successful Panda path: "
    f"{full_path_length * 1000:.1f} mm"
)

print(
    f"Requested video-derived path: "
    f"{real_distance_m * 1000:.1f} mm"
)


# =========================================================
# Convert real pusher path length into joint-1 target
# =========================================================

if (
    real_distance_m
    >= full_path_length
):

    target_q1 = (
        Q_END_FULL[0]
    )

    print(
        "WARNING: real action is "
        "longer than the reference "
        "robot path."
    )

else:

    target_q1 = np.interp(
        real_distance_m,
        cumulative_path,
        sample_q1,
    )


q_target = Q_START.copy()

q_target[0] = (
    target_q1
)


print(
    f"Target joint 1: "
    f"{target_q1:.4f} rad"
)

print(
    f"Replay duration: "
    f"{real_duration:.3f} s"
)


# =========================================================
# Reset before actual dynamic experiment
# =========================================================

mujoco.mj_resetData(
    model,
    data,
)

for adr, value in zip(
    qpos_addresses,
    Q_START,
):

    data.qpos[adr] = value


mujoco.mj_forward(
    model,
    data,
)


# =========================================================
# Run replay
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    # -----------------------------------------------------
    # Allow everything to settle first
    # -----------------------------------------------------

    settle_duration = 1.5

    settle_steps = int(
        settle_duration
        / model.opt.timestep
    )

    for _ in range(
        settle_steps
    ):

        data.ctrl[
            arm_actuator_ids
        ] = Q_START

        data.ctrl[
            gripper_actuator_id
        ] = GRIPPER_TARGET

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


    # -----------------------------------------------------
    # Record starting states
    # -----------------------------------------------------

    box_start = (
        data.xpos[
            box_body_id
        ].copy()
    )

    ee_start = (
        ee_position(
            data,
            ee,
        )
    )


    # Store full dynamic trajectories.
    ee_positions = []
    box_positions = []
    timestamps = []


    # -----------------------------------------------------
    # Convert real push duration to simulation steps
    # -----------------------------------------------------

    push_steps = max(
        1,
        int(
            real_duration
            / model.opt.timestep
        ),
    )


    # -----------------------------------------------------
    # Execute video-derived push
    # -----------------------------------------------------

    for step in range(
        push_steps
    ):

        u = (
            step
            / max(
                push_steps - 1,
                1,
            )
        )

        # Smoothstep interpolation:
        #
        # alpha(0) = 0
        # alpha(1) = 1
        #
        # with zero slope at each end.
        alpha = (
            3 * u**2
            - 2 * u**3
        )

        q_command = (
            Q_START
            + alpha
            * (
                q_target
                - Q_START
            )
        )

        data.ctrl[
            arm_actuator_ids
        ] = q_command

        data.ctrl[
            gripper_actuator_id
        ] = GRIPPER_TARGET

        mujoco.mj_step(
            model,
            data,
        )

        # ---------------------------------------------
        # NEW:
        # record actual dynamically achieved motion
        # ---------------------------------------------

        ee_positions.append(
            ee_position(
                data,
                ee,
            )
        )

        box_positions.append(
            data.xpos[
                box_body_id
            ].copy()
        )

        timestamps.append(
            data.time
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


    # -----------------------------------------------------
    # Record state immediately after commanded push
    # -----------------------------------------------------

    ee_push_end = (
        ee_position(
            data,
            ee,
        )
    )

    box_push_end = (
        data.xpos[
            box_body_id
        ].copy()
    )


    # -----------------------------------------------------
    # Hold after push to allow cube to settle
    # -----------------------------------------------------

    hold_steps = int(
        2.0
        / model.opt.timestep
    )

    for _ in range(
        hold_steps
    ):

        data.ctrl[
            arm_actuator_ids
        ] = q_target

        data.ctrl[
            gripper_actuator_id
        ] = GRIPPER_TARGET

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


    # -----------------------------------------------------
    # Final settled cube position
    # -----------------------------------------------------

    box_end = (
        data.xpos[
            box_body_id
        ].copy()
    )


# =========================================================
# Convert recorded trajectories to arrays
# =========================================================

ee_positions = np.array(
    ee_positions
)

box_positions = np.array(
    box_positions
)

timestamps = np.array(
    timestamps
)


# =========================================================
# Analyse actual Panda EE execution
# =========================================================

ee_segment_lengths = np.linalg.norm(
    np.diff(
        ee_positions[:, :2],
        axis=0,
    ),
    axis=1,
)

actual_ee_path_mm = (
    np.sum(
        ee_segment_lengths
    )
    * 1000
)

actual_ee_displacement_mm = (
    np.linalg.norm(
        ee_push_end[:2]
        - ee_start[:2]
    )
    * 1000
)


# =========================================================
# Analyse simulated box response
# =========================================================

box_delta_during_push = (
    box_push_end[:2]
    - box_start[:2]
)

box_motion_during_push_mm = (
    np.linalg.norm(
        box_delta_during_push
    )
    * 1000
)


box_delta_final = (
    box_end[:2]
    - box_start[:2]
)

sim_distance_mm = (
    np.linalg.norm(
        box_delta_final
    )
    * 1000
)


# =========================================================
# Coupling / execution metrics
# =========================================================

if actual_ee_path_mm > 1e-6:

    sim_coupling_ratio = (
        box_motion_during_push_mm
        / actual_ee_path_mm
    )

else:

    sim_coupling_ratio = (
        np.nan
    )


if real_distance_m > 1e-9:

    action_execution_ratio = (
        actual_ee_path_mm
        / (
            real_distance_m
            * 1000
        )
    )

else:

    action_execution_ratio = (
        np.nan
    )


# =========================================================
# Print results
# =========================================================

print(
    "\nACTION EXECUTION"
)

print(
    "----------------"
)

print(
    f"Requested pusher path: "
    f"{real_distance_m * 1000:.1f} mm"
)

print(
    f"Actual Panda EE path: "
    f"{actual_ee_path_mm:.1f} mm"
)

print(
    f"Actual Panda EE net displacement: "
    f"{actual_ee_displacement_mm:.1f} mm"
)

print(
    f"Action execution ratio: "
    f"{action_execution_ratio:.3f}"
)


print(
    "\nOBJECT RESPONSE"
)

print(
    "---------------"
)

print(
    f"Real box motion during push: "
    f"{real_box_distance_mm:.1f} mm"
)

print(
    f"Cube motion during push: "
    f"{box_motion_during_push_mm:.1f} mm"
)

print(
    f"Sim coupling ratio: "
    f"{sim_coupling_ratio:.3f}"
)


print(
    "\nFINAL RESULT"
)

print(
    "------------"
)

print(
    f"Sim box dx: "
    f"{box_delta_final[0] * 1000:.1f} mm"
)

print(
    f"Sim box dy: "
    f"{box_delta_final[1] * 1000:.1f} mm"
)

print(
    f"Sim box final distance: "
    f"{sim_distance_mm:.1f} mm"
)

print(
    f"Real box distance: "
    f"{real_box_distance_mm:.1f} mm"
)

print(
    f"Magnitude error: "
    f"{abs(sim_distance_mm - real_box_distance_mm):.1f} mm"
)