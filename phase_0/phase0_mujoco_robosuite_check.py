"""
Verification script for MuJoCo and Robosuite installations.
Runs basic checks to confirm both libraries work correctly.
"""

import numpy as np

# ──────────────────────────────────────────────
# 1. MuJoCo — direct API check
# ──────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: MuJoCo")
print("=" * 60)

import mujoco

print(f"  MuJoCo version : {mujoco.__version__}")

# Build a minimal model in XML (a free-floating box)
XML = """
<mujoco model="verify">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1"
            mass="1" rgba="0.2 0.6 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)

print(f"  Model name     : verify")  # MuJoCo 3.x removed model.model attr
print(f"  nq / nv        : {model.nq} / {model.nv}")
print(f"  Bodies         : {model.nbody}")
print(f"  Geoms          : {model.ngeom}")

# Run 500 steps and track the box falling under gravity
initial_z = data.qpos[2]          # z of freejoint
steps = 500
for _ in range(steps):
    mujoco.mj_step(model, data)

final_z = data.qpos[2]
sim_time = data.time

print(f"\n  Simulated {steps} steps  →  sim_time = {sim_time:.3f} s")
print(f"  Box z: {initial_z:.3f} m  →  {final_z:.3f} m  (fell {initial_z - final_z:.3f} m)")

assert final_z < initial_z, "Box should have fallen under gravity!"
print("  ✓  Gravity / integration check PASSED")

# Check that mj_forward updates site/body xpos
mujoco.mj_forward(model, data)
box_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
box_xpos = data.xpos[box_id]
print(f"  Box world pos  : {box_xpos}")
print("  ✓  mj_forward / kinematics check PASSED")


# ──────────────────────────────────────────────
# 2. Robosuite — wrapper check
# ──────────────────────────────────────────────
print()
print("=" * 60)
print("SECTION 2: Robosuite")
print("=" * 60)

import robosuite as suite
from robosuite import load_controller_config

print(f"  Robosuite version : {suite.__version__}")

# Create a simple Lift environment (Panda robot, OSC_POSE controller)
controller_config = load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,          # headless
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    controller_configs=controller_config,
)

print(f"  Environment     : {env.__class__.__name__}")
print(f"  Robot           : {env.robots[0].name}")
print(f"  Action dim      : {env.action_dim}")

obs = env.reset()
print(f"  Observation keys: {sorted(obs.keys())}")

# Take 50 random actions
rewards = []
for step in range(50):
    action = np.random.uniform(-0.1, 0.1, env.action_dim)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        obs = env.reset()

print(f"\n  Ran 50 random steps")
print(f"  Reward range    : [{min(rewards):.3f}, {max(rewards):.3f}]")

# Confirm the underlying MuJoCo sim is accessible
sim_time = env.sim.data.time
print(f"  MuJoCo sim time : {sim_time:.3f} s")
print("  ✓  Robosuite environment step/reset PASSED")
print("  ✓  MuJoCo sim accessible through robosuite PASSED")

env.close()

print()
print("=" * 60)
print("ALL CHECKS PASSED — MuJoCo + Robosuite are working correctly.")
print("=" * 60)