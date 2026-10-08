from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]

SCENE_PATH = (
    ROOT
    / "external"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

print(f"Loading: {SCENE_PATH}")

model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)