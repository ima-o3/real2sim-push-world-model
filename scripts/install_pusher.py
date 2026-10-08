
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[1]

PANDA_XML = (
    ROOT / "external"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)

SCENE_XML = ROOT / "configs" / "push_scene.xml"

if not PANDA_XML.exists():
    raise FileNotFoundError(f"Panda XML missing: {PANDA_XML}")

# Preserve existing XML comments.
parser = ET.XMLParser(
    target=ET.TreeBuilder(insert_comments=True)
)

tree = ET.parse(PANDA_XML, parser=parser)
root = tree.getroot()

# Find the actual robot hand, NOT the world body.
hand = root.find(".//body[@name='hand']")

if hand is None:
    raise RuntimeError("Cannot find Panda hand body.")

tool = hand.find("./body[@name='pusher_tool']")

if tool is None:
    tool = ET.SubElement(
        hand, "body", {"name": "pusher_tool"}
    )

def ensure_element(parent, tag, name, attributes):
    element = parent.find(f"./{tag}[@name='{name}']")

    if element is None:
        element = ET.SubElement(parent, tag)

    element.attrib.update({
        "name": name,
        **attributes,
    })

    return element

# Rigid capsule along the hand's local +Z axis.
ensure_element(
    tool,
    "geom",
    "pusher_rod",
    {
        "type": "capsule",
        "fromto": "0 0 0.070 0 0 0.145",
        "size": "0.006",
        "mass": "0.03",
        "friction": "0.8 0.005 0.0001",
        "rgba": "0.9 0.1 0.1 1",
    },
)

# Put the site at the rounded outer tip.
ensure_element(
    tool,
    "site",
    "pusher_contact",
    {
        "pos": "0 0 0.151",
        "size": "0.004",
        "rgba": "0 1 0 1",
    },
)

# Back up the original once.
backup = PANDA_XML.with_suffix(".xml.bak")

if not backup.exists():
    shutil.copy2(PANDA_XML, backup)

ET.indent(tree, space="  ")
tree.write(
    PANDA_XML,
    encoding="utf-8",
    xml_declaration=True,
)

print("Pusher geometry installed.")

# Check the FULL assembled scene.
model = mujoco.MjModel.from_xml_path(
    str(SCENE_XML.resolve())
)

objects = [
    (mujoco.mjtObj.mjOBJ_BODY, "panda_pusher_tool"),
    (mujoco.mjtObj.mjOBJ_GEOM, "panda_pusher_rod"),
    (mujoco.mjtObj.mjOBJ_SITE, "panda_pusher_contact"),
]

for obj_type, name in objects:
    idx = mujoco.mj_name2id(
        model, obj_type, name
    )

    print(f"{name}: {idx}")

    if idx < 0:
        raise RuntimeError(f"Missing compiled object: {name}")

print("SUCCESS: All pusher components loaded.")
