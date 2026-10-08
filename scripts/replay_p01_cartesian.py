from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "configs" / "push_scene.xml"

CSV_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pilots"
    / "P01_center.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "metrics"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Existing validated Panda configuration
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

Q_END_REFERENCE = np.array([
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
# IK tuning
# =========================================================

IK_MAX_ITERATIONS = 80

IK_DAMPING = 0.015

IK_GAIN = 0.8

MAX_JOINT_STEP = 0.05

POSITION_TOLERANCE = 0.00020     # 0.2 mm

ORIENTATION_TOLERANCE = np.deg2rad(
    0.75
)

# Converts orientation error in radians into approximately
# position-equivalent task weighting.
ORIENTATION_SCALE = 0.12

NULLSPACE_GAIN = 0.002


# =========================================================
# Extract real P01 trajectory
# =========================================================

def extract_real_trajectory(csv_path):

    df = pd.read_csv(csv_path)

    # Fill any small tracking gaps.
    df = df.interpolate(
        limit_direction="both"
    )

    t = df["time_s"].to_numpy()

    box = (
        df[[
            "box_x_mm",
            "box_y_mm",
        ]]
        .rolling(
            7,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    pusher = (
        df[[
            "pusher_x_mm",
            "pusher_y_mm",
        ]]
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
    # Find the main box-motion event
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

    raw_active = (
        box_speed > 20.0
    )

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

    groups = []

    start = None

    for i, is_active in enumerate(
        active
    ):

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

            if (
                end - start
                >= 10
            ):

                score = np.sum(
                    box_speed[
                        start:end + 1
                    ]
                )

                groups.append(
                    (
                        start,
                        end,
                        score,
                    )
                )

            start = None

    if not groups:

        raise RuntimeError(
            "Could not identify "
            "the main P01 push."
        )

    start, end, _ = max(
        groups,
        key=lambda g: g[2],
    )

    # -----------------------------------------------------
    # Crop to active push
    # -----------------------------------------------------

    push_t = (
        t[start:end + 1]
        - t[start]
    )

    push_pusher = (
        pusher[
            start:end + 1
        ]
    )

    push_box = (
        box[
            start:end + 1
        ]
    )

    # Relative trajectory starts at zero.
    pusher_relative_mm = (
        push_pusher
        - push_pusher[0]
    )

    box_relative_mm = (
        push_box
        - push_box[0]
    )

    # -----------------------------------------------------
    # Trajectory metrics
    # -----------------------------------------------------

    pusher_net_delta_mm = (
        pusher_relative_mm[-1]
    )

    pusher_net_distance_mm = (
        np.linalg.norm(
            pusher_net_delta_mm
        )
    )

    pusher_path_mm = np.sum(
        np.linalg.norm(
            np.diff(
                pusher_relative_mm,
                axis=0,
            ),
            axis=1,
        )
    )

    box_delta_mm = (
        box_relative_mm[-1]
    )

    box_distance_mm = np.linalg.norm(
        box_delta_mm
    )

    duration = (
        push_t[-1]
    )

    print(
        "\nREAL P01 TRAJECTORY"
    )

    print(
        "-------------------"
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
        f"Pusher net dx: "
        f"{pusher_net_delta_mm[0]:.1f} mm"
    )

    print(
        f"Pusher net dy: "
        f"{pusher_net_delta_mm[1]:.1f} mm"
    )

    print(
        f"Pusher net displacement: "
        f"{pusher_net_distance_mm:.1f} mm"
    )

    print(
        f"Pusher trajectory path length: "
        f"{pusher_path_mm:.1f} mm"
    )

    print(
        f"Real box displacement: "
        f"{box_distance_mm:.1f} mm"
    )

    return {
        "time": push_t,
        "pusher_relative_mm":
            pusher_relative_mm,
        "box_relative_mm":
            box_relative_mm,
        "duration":
            duration,
        "box_distance_mm":
            box_distance_mm,
        "pusher_net_distance_mm":
            pusher_net_distance_mm,
        "pusher_path_mm":
            pusher_path_mm,
    }


# =========================================================
# MuJoCo helpers
# =========================================================

def find_ee(model):
    name = "panda_pusher_contact"

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Required pusher site '{name}' not found. "
            "Check the Panda XML and attached model."
        )

    print(f"Using EE site: {name}")

    return ("site", idx, name)


def get_ee_pose(
    data,
    ee,
):

    kind, idx, _ = ee

    if kind == "site":

        position = (
            data.site_xpos[idx]
            .copy()
        )

        rotation = (
            data.site_xmat[idx]
            .reshape(3, 3)
            .copy()
        )

    else:

        position = (
            data.xpos[idx]
            .copy()
        )

        rotation = (
            data.xmat[idx]
            .reshape(3, 3)
            .copy()
        )

    return (
        position,
        rotation,
    )


def get_ee_jacobian(
    model,
    data,
    ee,
    dof_indices,
):

    jacp = np.zeros(
        (
            3,
            model.nv,
        )
    )

    jacr = np.zeros(
        (
            3,
            model.nv,
        )
    )

    kind, idx, _ = ee

    if kind == "site":

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            idx,
        )

    else:

        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            idx,
        )

    return (
        jacp[
            :,
            dof_indices,
        ],
        jacr[
            :,
            dof_indices,
        ],
    )


def orientation_error(
    current_rotation,
    desired_rotation,
):

    # Small-angle orientation error
    # expressed in world coordinates.
    error = 0.5 * (
        np.cross(
            current_rotation[:, 0],
            desired_rotation[:, 0],
        )
        +
        np.cross(
            current_rotation[:, 1],
            desired_rotation[:, 1],
        )
        +
        np.cross(
            current_rotation[:, 2],
            desired_rotation[:, 2],
        )
    )

    return error


def find_box_body(
    model,
):

    candidates = [
        "box",
        "cube",
        "push_box",
        "object",
    ]

    for name in candidates:

        idx = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        if idx >= 0:

            print(
                f"Using object body: "
                f"{name}"
            )

            return idx

    raise RuntimeError(
        "Could not find "
        "push object."
    )


# =========================================================
# IK solver
# =========================================================

def solve_ik(
    model,
    data,
    ee,
    qpos_addresses,
    dof_indices,
    joint_ids,
    q_seed,
    target_position,
    target_rotation,
):

    q = q_seed.copy()

    q_reference = (
        q_seed.copy()
    )

    final_position_error = np.inf

    final_orientation_error = np.inf

    for _ in range(
        IK_MAX_ITERATIONS
    ):

        # Write current estimate.
        for adr, value in zip(
            qpos_addresses,
            q,
        ):

            data.qpos[adr] = value

        mujoco.mj_forward(
            model,
            data,
        )

        (
            current_position,
            current_rotation,
        ) = get_ee_pose(
            data,
            ee,
        )

        position_error = (
            target_position
            - current_position
        )

        rotation_error = (
            orientation_error(
                current_rotation,
                target_rotation,
            )
        )

        final_position_error = (
            np.linalg.norm(
                position_error
            )
        )

        final_orientation_error = (
            np.linalg.norm(
                rotation_error
            )
        )

        if (
            final_position_error
            < POSITION_TOLERANCE
            and
            final_orientation_error
            < ORIENTATION_TOLERANCE
        ):

            break

        jacp, jacr = (
            get_ee_jacobian(
                model,
                data,
                ee,
                dof_indices,
            )
        )

        task_error = np.concatenate(
            [
                position_error,
                (
                    ORIENTATION_SCALE
                    * rotation_error
                ),
            ]
        )

        task_jacobian = np.vstack(
            [
                jacp,
                (
                    ORIENTATION_SCALE
                    * jacr
                ),
            ]
        )

        # Damped least-squares inverse.
        A = (
            task_jacobian
            @ task_jacobian.T
            +
            (
                IK_DAMPING ** 2
            )
            * np.eye(6)
        )

        dq = (
            task_jacobian.T
            @ np.linalg.solve(
                A,
                task_error,
            )
        )

        dq *= IK_GAIN

        # Gentle regularisation discourages
        # unnecessary joint excursions.
        dq += (
            NULLSPACE_GAIN
            * (
                q_reference
                - q
            )
        )

        # Prevent enormous individual
        # solver jumps.
        max_step = np.max(
            np.abs(dq)
        )

        if (
            max_step
            > MAX_JOINT_STEP
        ):

            dq *= (
                MAX_JOINT_STEP
                / max_step
            )

        q += dq

        # Respect joint limits.
        for i, joint_id in enumerate(
            joint_ids
        ):

            if (
                model.jnt_limited[
                    joint_id
                ]
            ):

                lower, upper = (
                    model.jnt_range[
                        joint_id
                    ]
                )

                q[i] = np.clip(
                    q[i],
                    lower,
                    upper,
                )

    return (
        q,
        final_position_error,
        final_orientation_error,
    )


# =========================================================
# Interpolation helper
# =========================================================

def interpolate_waypoint(
    t,
    waypoint_times,
    values,
):

    if (
        t <= waypoint_times[0]
    ):

        return values[0].copy()

    if (
        t >= waypoint_times[-1]
    ):

        return values[-1].copy()

    upper = np.searchsorted(
        waypoint_times,
        t,
        side="right",
    )

    lower = (
        upper - 1
    )

    t0 = (
        waypoint_times[lower]
    )

    t1 = (
        waypoint_times[upper]
    )

    alpha = (
        (t - t0)
        / (t1 - t0)
    )

    return (
        (1.0 - alpha)
        * values[lower]
        +
        alpha
        * values[upper]
    )


# =========================================================
# Read real experiment
# =========================================================

real = extract_real_trajectory(
    CSV_PATH
)


# =========================================================
# Load MuJoCo
# =========================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)


# =========================================================
# Panda joint / actuator IDs
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
        "seven Panda joints."
    )


qpos_addresses = [
    model.jnt_qposadr[
        joint_id
    ]
    for joint_id in joint_ids
]


dof_indices = [
    model.jnt_dofadr[
        joint_id
    ]
    for joint_id in joint_ids
]


arm_actuator_ids = [
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"panda_actuator{i}",
    )
    for i in range(1, 8)
]


gripper_actuator_id = (
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "panda_actuator8",
    )
)


# =========================================================
# Identify EE and object
# =========================================================

ee = find_ee(
    model
)

box_body_id = find_box_body(
    model
)


# =========================================================
# Find a collision-free starting pose for the pusher
# =========================================================

setup_data = mujoco.MjData(model)

# Start from the old robot configuration as an IK seed.
for adr, value in zip(qpos_addresses, Q_START):
    setup_data.qpos[adr] = value

mujoco.mj_forward(model, setup_data)

old_tip, old_rotation = get_ee_pose(setup_data, ee)
box_centre = setup_data.xpos[box_body_id].copy()

print("\nSAFE START CONFIGURATION")
print("------------------------")
print("Old pusher tip:", np.round(old_tip, 4))
print("Box centre:", np.round(box_centre, 4))

# The box is 80 mm wide in Y, so its incoming
# face is at y = 0.040 m.
#
# Position the pusher 12 mm beyond that face,
# giving approximately 6 mm of clearance
# after accounting for the capsule radius.
#
# Position its contact tip at box mid-height.

safe_target = np.array([
    box_centre[0],
    box_centre[1] + 0.052,
    box_centre[2],
])

print("Target tip:", np.round(safe_target, 4))

q_safe, pos_error, rot_error = solve_ik(
    model=model,
    data=setup_data,
    ee=ee,
    qpos_addresses=qpos_addresses,
    dof_indices=dof_indices,
    joint_ids=joint_ids,
    q_seed=Q_START.copy(),
    target_position=safe_target,
    target_rotation=old_rotation,
)

print(f"Start IK error: {pos_error * 1000:.3f} mm")

if pos_error > 0.002:
    raise RuntimeError(
        "Could not find an accurate collision-free start pose."
    )

# Install the new robot start configuration.
Q_START = q_safe.copy()

# Use a joint-1 sweep from the NEW start pose only
# to establish the camera-to-robot push direction.
Q_END_REFERENCE = Q_START.copy()
Q_END_REFERENCE[0] -= 0.25

# Verify that the initial geometry no longer
# has substantial physical penetration.
for adr, value in zip(qpos_addresses, Q_START):
    setup_data.qpos[adr] = value

mujoco.mj_forward(model, setup_data)

actual_tip, _ = get_ee_pose(setup_data, ee)
print("Solved tip:", np.round(actual_tip, 4))
print(
    "Tip height above table:",
    round((actual_tip[2] - 0.375) * 1000, 2),
    "mm",
)

for i in range(setup_data.ncon):
    contact = setup_data.contact[i]

    a = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
    ) or ""

    b = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
    ) or ""

    robot_environment_contact = (
        (a.startswith("panda_") and b in ("table", "box_geom"))
        or
        (b.startswith("panda_") and a in ("table", "box_geom"))
    )

    if robot_environment_contact and contact.dist < -0.002:
        raise RuntimeError(
            f"Initial penetration: {a} vs {b}, "
            f"{contact.dist * 1000:.2f} mm"
        )

print("Safe starting configuration prepared.")


# =========================================================
# Establish start pose
# =========================================================

reference_data = (
    mujoco.MjData(
        model
    )
)

for adr, value in zip(
    qpos_addresses,
    Q_START,
):

    reference_data.qpos[
        adr
    ] = value


mujoco.mj_forward(
    model,
    reference_data,
)


(
    ee_start_position,
    ee_start_rotation,
) = get_ee_pose(
    reference_data,
    ee,
)


# =========================================================
# Determine direction of previously validated Panda push
# =========================================================

for adr, value in zip(
    qpos_addresses,
    Q_END_REFERENCE,
):

    reference_data.qpos[
        adr
    ] = value


mujoco.mj_forward(
    model,
    reference_data,
)


(
    ee_reference_end,
    _,
) = get_ee_pose(
    reference_data,
    ee,
)


sim_reference_vector = (
    ee_reference_end[:2]
    - ee_start_position[:2]
)


sim_reference_distance = (
    np.linalg.norm(
        sim_reference_vector
    )
)


if (
    sim_reference_distance
    < 1e-6
):

    raise RuntimeError(
        "Reference Panda push "
        "has almost no XY motion."
    )


# =========================================================
# Rotate camera trajectory into robot/table frame
# =========================================================

real_relative_m = (
    real[
        "pusher_relative_mm"
    ]
    / 1000.0
)


real_final_vector = (
    real_relative_m[-1]
)


if (
    np.linalg.norm(
        real_final_vector
    )
    < 1e-6
):

    raise RuntimeError(
        "Real push trajectory "
        "has almost no displacement."
    )


real_angle = np.arctan2(
    real_final_vector[1],
    real_final_vector[0],
)


sim_angle = np.arctan2(
    sim_reference_vector[1],
    sim_reference_vector[0],
)


rotation_angle = (
    sim_angle
    - real_angle
)


c = np.cos(
    rotation_angle
)

s = np.sin(
    rotation_angle
)


camera_to_sim_rotation = np.array([
    [c, -s],
    [s,  c],
])


mapped_relative_xy = (
    camera_to_sim_rotation
    @ real_relative_m.T
).T


desired_positions = np.column_stack(
    [
        (
            ee_start_position[0]
            + mapped_relative_xy[:, 0]
        ),
        (
            ee_start_position[1]
            + mapped_relative_xy[:, 1]
        ),
        np.full(
            len(mapped_relative_xy),
            ee_start_position[2],
        ),
    ]
)


print(
    "\nCARTESIAN MAPPING"
)

print(
    "-----------------"
)

print(
    f"Camera push angle: "
    f"{np.degrees(real_angle):.1f} deg"
)

print(
    f"Reference Panda angle: "
    f"{np.degrees(sim_angle):.1f} deg"
)

print(
    f"Applied rotation: "
    f"{np.degrees(rotation_angle):.1f} deg"
)

print(
    f"Mapped final XY displacement: "
    f"{np.linalg.norm(mapped_relative_xy[-1]) * 1000:.1f} mm"
)


# =========================================================
# Solve entire 2-D trajectory offline
# =========================================================

ik_data = mujoco.MjData(
    model
)


q_waypoints = []

position_errors = []

orientation_errors = []


q_seed = (
    Q_START.copy()
)


for i, target_position in enumerate(
    desired_positions
):

    (
        q_solution,
        position_error,
        orientation_error_value,
    ) = solve_ik(
        model=model,
        data=ik_data,
        ee=ee,
        qpos_addresses=qpos_addresses,
        dof_indices=dof_indices,
        joint_ids=joint_ids,
        q_seed=q_seed,
        target_position=
            target_position,
        target_rotation=
            ee_start_rotation,
    )

    q_waypoints.append(
        q_solution.copy()
    )

    position_errors.append(
        position_error
    )

    orientation_errors.append(
        orientation_error_value
    )

    # Seed next frame from previous solution.
    q_seed = (
        q_solution.copy()
    )


q_waypoints = np.array(
    q_waypoints
)

position_errors = np.array(
    position_errors
)

orientation_errors = np.array(
    orientation_errors
)


print(
    "\nIK SOLUTION"
)

print(
    "-----------"
)

print(
    f"Waypoints solved: "
    f"{len(q_waypoints)}"
)

print(
    f"Mean IK position error: "
    f"{np.mean(position_errors) * 1000:.3f} mm"
)

print(
    f"Max IK position error: "
    f"{np.max(position_errors) * 1000:.3f} mm"
)

print(
    f"Max IK orientation error: "
    f"{np.degrees(np.max(orientation_errors)):.3f} deg"
)


if (
    np.max(
        position_errors
    )
    > 0.010
):

    raise RuntimeError(
        "IK trajectory contains a waypoint "
        "with >10 mm position error. "
        "Do not execute it."
    )


# =========================================================
# Dynamic simulation
# =========================================================

data = mujoco.MjData(
    model
)


mujoco.mj_resetData(
    model,
    data,
)


for adr, value in zip(
    qpos_addresses,
    Q_START,
):

    data.qpos[
        adr
    ] = value


mujoco.mj_forward(
    model,
    data,
)


# =========================================================
# Hold-pose calibration and dynamic replay
# =========================================================
# MuJoCo's Panda position actuators can have a finite static
# joint offset under gravity. Identify a SMALL feed-forward
# setpoint correction using the stationary robot, rather than
# permitting large tool drift or silently relaxing safeguards.

with mujoco.viewer.launch_passive(model, data) as viewer:
    hold_command = Q_START.copy()
    MAX_HOLD_BIAS_RAD = 0.035     # 2.0 degrees per joint maximum
    MAX_BIAS_STEP_RAD = 0.012    # 0.69 degrees per iteration
    HOLD_ERROR_LIMIT_MM = 2.0    # Strict start-pose tolerance
    MAX_SETTLING_ROUNDS = 8
    SETTLING_SECONDS_PER_ROUND = 0.7

    def ensure_valid_arm_command(command):
        for actuator_id, requested in zip(arm_actuator_ids, command):
            if model.actuator_ctrllimited[actuator_id]:
                lower, upper = model.actuator_ctrlrange[actuator_id]
                if requested < lower or requested > upper:
                    raise RuntimeError(
                        f"Position actuator {actuator_id} setpoint "
                        f"{requested:.4f} outside [{lower:.4f}, {upper:.4f}]"
                    )

    def reject_unintended_contacts():
        """Abort if table contact or premature box contact occurs."""
        for i in range(data.ncon):
            contact = data.contact[i]
            a = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            b = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            if a.startswith("panda_") and b in {"table", "box_geom"} or (
                b.startswith("panda_") and a in {"table", "box_geom"}
            ):
                raise RuntimeError(
                    "Unintended robot/environment contact while settling: "
                    f"{a} vs {b}, distance {contact.dist * 1000:.2f} mm. "
                    "Check pusher placement before replay."
                )

    print("\nGRAVITY / STATIC SETPOINT CALIBRATION")
    print("-------------------------------------")

    for round_id in range(1, MAX_SETTLING_ROUNDS + 1):
        ensure_valid_arm_command(hold_command)
        steps = int(
            SETTLING_SECONDS_PER_ROUND / model.opt.timestep
        )
        for _ in range(steps):
            data.ctrl[arm_actuator_ids] = hold_command
            data.ctrl[gripper_actuator_id] = GRIPPER_TARGET
            mujoco.mj_step(model, data)
            reject_unintended_contacts()
            viewer.sync()
            time.sleep(model.opt.timestep)

        actual_ee_start, _ = get_ee_pose(data, ee)
        q_actual = np.array([data.qpos[adr] for adr in qpos_addresses])
        tip_error_mm = (actual_ee_start - ee_start_position) * 1000.0
        start_drift_mm = float(np.linalg.norm(tip_error_mm))
        joint_error = Q_START - q_actual
        velocity_norm = float(np.linalg.norm(data.qvel[dof_indices]))

        print(
            f"Round {round_id}: tip drift {start_drift_mm:.2f} mm; "
            f"XYZ error {np.round(tip_error_mm, 2)} mm; "
            f"joint velocity {velocity_norm:.4f} rad/s"
        )

        if start_drift_mm <= HOLD_ERROR_LIMIT_MM and velocity_norm < 0.03:
            print("Static starting pose validated.")
            break

        # A position servo under gravity sags, so command slightly
        # beyond the desired joint position to compensate. This is
        # a measured, bounded setpoint offset, not an IK modification.
        hold_command += np.clip(
            0.8 * joint_error,
            -MAX_BIAS_STEP_RAD,
            MAX_BIAS_STEP_RAD,
        )
        if np.max(np.abs(hold_command - Q_START)) > MAX_HOLD_BIAS_RAD:
            raise RuntimeError(
                "Static gravity compensation exceeded 2 degrees. "
                "Check actuator gains, saturation and contact geometry."
            )
    else:
        raise RuntimeError(
            f"Start pose not stable after {MAX_SETTLING_ROUNDS} rounds: "
            f"tip drift {start_drift_mm:.2f} mm. "
            "No replay was executed."
        )

    # Apply the same experimentally determined joint-space offset
    # to every subsequent position-actuator command. Offline IK
    # waypoints remain unchanged, and the actual trajectory error
    # will still be measured and reported.
    actuator_bias = hold_command - Q_START
    print("Final start-pose drift:", round(start_drift_mm, 2), "mm")
    print("Joint setpoint bias (deg):", np.round(np.degrees(actuator_bias), 3))

    box_start = data.xpos[box_body_id].copy()
    sim_times = []
    desired_log = []
    actual_log = []
    box_log = []

    push_steps = int(np.ceil(real["duration"] / model.opt.timestep)) + 1
    simulation_push_start_time = data.time

    for step in range(push_steps):
        elapsed = min(step * model.opt.timestep, real["duration"])
        q_nominal = interpolate_waypoint(elapsed, real["time"], q_waypoints)
        q_command = q_nominal + actuator_bias
        ensure_valid_arm_command(q_command)
        desired_position = interpolate_waypoint(
            elapsed, real["time"], desired_positions
        )

        data.ctrl[arm_actuator_ids] = q_command
        data.ctrl[gripper_actuator_id] = GRIPPER_TARGET
        mujoco.mj_step(model, data)
        actual_position, _ = get_ee_pose(data, ee)

        sim_times.append(data.time - simulation_push_start_time)
        desired_log.append(desired_position.copy())
        actual_log.append(actual_position.copy())
        box_log.append(data.xpos[box_body_id].copy())

        viewer.sync()
        time.sleep(model.opt.timestep)

    ee_push_end, _ = get_ee_pose(data, ee)
    box_push_end = data.xpos[box_body_id].copy()

    # Hold final joint targets after the push; cube may keep sliding.
    end_command = q_waypoints[-1] + actuator_bias
    ensure_valid_arm_command(end_command)
    hold_steps = int(2.0 / model.opt.timestep)
    for _ in range(hold_steps):
        data.ctrl[arm_actuator_ids] = end_command
        data.ctrl[gripper_actuator_id] = GRIPPER_TARGET
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

    box_final = data.xpos[box_body_id].copy()


# =========================================================
# Analyse execution
# =========================================================

sim_times = np.array(
    sim_times
)

desired_log = np.array(
    desired_log
)

actual_log = np.array(
    actual_log
)

box_log = np.array(
    box_log
)


# ---------------------------------------------------------
# Requested Cartesian path length
# ---------------------------------------------------------

requested_path_mm = (
    np.sum(
        np.linalg.norm(
            np.diff(
                desired_log[:, :2],
                axis=0,
            ),
            axis=1,
        )
    )
    * 1000
)


# ---------------------------------------------------------
# Actual Cartesian EE path
# ---------------------------------------------------------

actual_path_mm = (
    np.sum(
        np.linalg.norm(
            np.diff(
                actual_log[:, :2],
                axis=0,
            ),
            axis=1,
        )
    )
    * 1000
)


actual_net_mm = (
    np.linalg.norm(
        ee_push_end[:2]
        - actual_ee_start[:2]
    )
    * 1000
)


# ---------------------------------------------------------
# Cartesian tracking error
# ---------------------------------------------------------

tracking_error = np.linalg.norm(
    (
        actual_log[:, :2]
        - desired_log[:, :2]
    ),
    axis=1,
)


tracking_rmse_mm = (
    np.sqrt(
        np.mean(
            tracking_error ** 2
        )
    )
    * 1000
)


tracking_max_mm = (
    np.max(
        tracking_error
    )
    * 1000
)


endpoint_error_mm = (
    np.linalg.norm(
        actual_log[-1, :2]
        - desired_log[-1, :2]
    )
    * 1000
)


# ---------------------------------------------------------
# Object response
# ---------------------------------------------------------

box_push_delta = (
    box_push_end[:2]
    - box_start[:2]
)


box_push_distance_mm = (
    np.linalg.norm(
        box_push_delta
    )
    * 1000
)


box_final_delta = (
    box_final[:2]
    - box_start[:2]
)


box_final_distance_mm = (
    np.linalg.norm(
        box_final_delta
    )
    * 1000
)


magnitude_error_mm = abs(
    box_final_distance_mm
    - real["box_distance_mm"]
)


# =========================================================
# Print results
# =========================================================

print(
    "\nCARTESIAN ACTION EXECUTION"
)

print(
    "--------------------------"
)

print(
    f"Requested EE path: "
    f"{requested_path_mm:.1f} mm"
)

print(
    f"Actual Panda EE path: "
    f"{actual_path_mm:.1f} mm"
)

print(
    f"Actual EE net displacement: "
    f"{actual_net_mm:.1f} mm"
)

print(
    f"Cartesian tracking RMSE: "
    f"{tracking_rmse_mm:.2f} mm"
)

print(
    f"Maximum tracking error: "
    f"{tracking_max_mm:.2f} mm"
)

print(
    f"Endpoint tracking error: "
    f"{endpoint_error_mm:.2f} mm"
)


print(
    "\nOBJECT RESPONSE"
)

print(
    "---------------"
)

print(
    f"Real box displacement: "
    f"{real['box_distance_mm']:.1f} mm"
)

print(
    f"Sim cube displacement "
    f"during push: "
    f"{box_push_distance_mm:.1f} mm"
)

print(
    f"Sim cube final displacement: "
    f"{box_final_distance_mm:.1f} mm"
)

print(
    f"Real-vs-sim magnitude error: "
    f"{magnitude_error_mm:.1f} mm"
)


# =========================================================
# Save detailed replay data
# =========================================================

result_df = pd.DataFrame({

    "time_s":
        sim_times,

    "desired_ee_x":
        desired_log[:, 0],

    "desired_ee_y":
        desired_log[:, 1],

    "desired_ee_z":
        desired_log[:, 2],

    "actual_ee_x":
        actual_log[:, 0],

    "actual_ee_y":
        actual_log[:, 1],

    "actual_ee_z":
        actual_log[:, 2],

    "box_x":
        box_log[:, 0],

    "box_y":
        box_log[:, 1],

    "box_z":
        box_log[:, 2],

    "tracking_error_mm":
        tracking_error * 1000,

})


output_path = (
    RESULTS_DIR
    / "P01_cartesian_replay.csv"
)


result_df.to_csv(
    output_path,
    index=False,
)


print(
    "\nSaved detailed replay data:"
)

print(
    output_path
)