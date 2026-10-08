import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
    <option timestep="0.002"/>

    <worldbody>
        <light pos="0 0 2"/>

        <geom name="floor"
              type="plane"
              size="1 1 0.1"
              rgba="0.8 0.8 0.8 1"/>

        <body name="box" pos="0 0 0.5">
            <freejoint/>

            <geom type="box"
                  size="0.05 0.05 0.05"
                  mass="0.2"
                  friction="0.8 0.1 0.1"
                  rgba="0.2 0.5 0.8 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Give you time to actually see the cube before it falls
    time.sleep(2)

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        # Run approximately in real time
        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)