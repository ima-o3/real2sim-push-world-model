from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "data" / "raw" / "pilots" / "P01_center.mp4"

# Provisional calibration.
# Replace this with your accurately measured tape separation later.
TAPE_DISTANCE_MM = 300.0


def blue_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return cv2.inRange(
        hsv,
        np.array([80, 20, 80]),
        np.array([130, 220, 255]),
    )


def box_mask(frame):
    """
    Works specifically for the pale blue/white box in the pilot footage.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([80, 20, 90]),
        np.array([130, 170, 255]),
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    return mask


def red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(
        hsv,
        np.array([0, 150, 100]),
        np.array([8, 255, 255]),
    )

    upper_red = cv2.inRange(
        hsv,
        np.array([172, 150, 100]),
        np.array([179, 255, 255]),
    )

    mask = lower_red | upper_red

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    return mask


def get_tape_centres(frame):
    mask = blue_mask(frame)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates = []

    for i in range(1, n):

        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter out the box and tiny image noise.
        if (
            150 < area < 2500
            and 8 < width < 70
            and 8 < height < 80
        ):
            candidates.append(centroids[i])

    candidates = sorted(
        candidates,
        key=lambda p: p[0],
    )

    if len(candidates) < 2:
        raise RuntimeError(
            "Could not find both blue calibration markers."
        )

    return (
        np.array(candidates[0]),
        np.array(candidates[-1]),
    )


def get_box_centre(frame, previous=None):
    mask = box_mask(frame)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates = []

    for i in range(1, n):

        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if (
            2500 < area < 20000
            and 50 < width < 180
            and 40 < height < 180
        ):
            candidates.append(
                (
                    centroids[i],
                    area,
                )
            )

    if not candidates:
        return None

    if previous is None:

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

    else:

        candidates.sort(
            key=lambda item:
            np.linalg.norm(
                item[0] - previous
            )
        )

    return np.array(candidates[0][0])


def get_red_centroid(frame):
    mask = red_mask(frame)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) > 1200
    ]

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    return np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ]
    )


cap = cv2.VideoCapture(str(VIDEO_PATH))

fps = cap.get(cv2.CAP_PROP_FPS)

ok, first_frame = cap.read()

if not ok:
    raise RuntimeError("Could not open video.")

tape_a, tape_b = get_tape_centres(first_frame)

pixels_between_tapes = np.linalg.norm(
    tape_b - tape_a
)

mm_per_pixel = (
    TAPE_DISTANCE_MM
    / pixels_between_tapes
)

print(
    f"Tape spacing: "
    f"{pixels_between_tapes:.1f} px"
)

print(
    f"Calibration: "
    f"{mm_per_pixel:.4f} mm/px"
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0,
)

rows = []

previous_box = None

frame_number = 0

while True:

    ok, frame = cap.read()

    if not ok:
        break

    box = get_box_centre(
        frame,
        previous_box,
    )

    if box is not None:
        previous_box = box

    pusher = get_red_centroid(frame)

    rows.append(
        {
            "frame": frame_number,
            "time_s": frame_number / fps,

            "box_u": (
                box[0]
                if box is not None
                else np.nan
            ),

            "box_v": (
                box[1]
                if box is not None
                else np.nan
            ),

            "pusher_u": (
                pusher[0]
                if pusher is not None
                else np.nan
            ),

            "pusher_v": (
                pusher[1]
                if pusher is not None
                else np.nan
            ),
        }
    )

    frame_number += 1


cap.release()

df = pd.DataFrame(rows)

df["box_x_mm"] = (
    df["box_u"] - tape_a[0]
) * mm_per_pixel

df["box_y_mm"] = (
    df["box_v"] - tape_a[1]
) * mm_per_pixel

df["pusher_x_mm"] = (
    df["pusher_u"] - tape_a[0]
) * mm_per_pixel

df["pusher_y_mm"] = (
    df["pusher_v"] - tape_a[1]
) * mm_per_pixel


output_dir = (
    ROOT
    / "data"
    / "processed"
    / "pilots"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    output_dir
    / "P01_center.csv"
)

df.to_csv(
    output_path,
    index=False,
)

print(
    f"Saved trajectory to:\n"
    f"{output_path}"
)

valid_box = df.dropna(
    subset=["box_x_mm", "box_y_mm"]
)

start = valid_box[
    ["box_x_mm", "box_y_mm"]
].iloc[:30].mean()

end = valid_box[
    ["box_x_mm", "box_y_mm"]
].iloc[-30:].mean()

delta = end - start

distance = np.linalg.norm(delta)

print()
print("Real box displacement:")
print(
    f"dx = {delta.iloc[0]:.1f} mm"
)
print(
    f"dy = {delta.iloc[1]:.1f} mm"
)
print(
    f"|d| = {distance:.1f} mm"
)