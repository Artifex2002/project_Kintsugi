# Kintsugi
An experimental fracture of VLA models: exposing universal vulnerabilities so they may be mended with gold.

---

This repo is a reproducible research environment (provides a ready-to-use **conda(mamba)** environment) for running and experimenting with my theorized universal attack on Vision-Language-Action models, specifically on **SmolVLA**, in simulation using the **LIBERO** benchmark suite on the **MuJoCo** physics engine. 

**Stack:**
- **SmolVLA** — A compact (450M params), small Vision-Language-Action model (HuggingFace / LeRobot)
- **LIBERO** — 130 language-conditioned robot manipulation tasks (LeRobot/Libero)
- **MuJoCo 3.x** — Physics engine for robotics simulation
- **Robosuite 1.4.1** — Robot simulation framework (pinned for LIBERO compatibility)
- **PyTorch with MPS** — GPU acceleration via Apple Metal Performance Shaders
- **pyav** — Video decoding backend (replaces torchcodec, broken on macOS ARM?)

---

## 🚀 Getting Started

### Prerequisites

- macOS 12.3 or later (required for MPS support)
- [Homebrew](https://brew.sh)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) + [Mamba](https://mamba.readthedocs.io)

### Step 1: Install System Dependencies

```bash
# Install Conda if you don't have it. Follow the guide below:
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
# Install Mamba into your base conda env if you haven't already
conda install mamba -n base -c conda-forge

# Install system-level dependencies for MuJoCo rendering
brew install glfw cmake
```

### Step 2: Clone This Repository

```bash
git clone https://github.com/Artifex2002/project_Kintsugi.git

# Feel free to rename this to your desired project name
cd project_Kintsugi
```

### Step 3: Create the Conda Environment

```bash
# Creates the environment with all pinned dependencies
# Replace my_new_env_name with desired environment name
mamba env create -f environment.yml -n project_Kintsugi

# Activate it
conda activate project_Kintsugi
```

### Step 4: Verify Installation

```bash
# 1. Confirm MuJoCo and Robosuite are working
cd phase_0
python phase0_mujoco_robosuite_check.py

# 2. Confirm the full SmolVLA inference pipeline works end-to-end
python phase0_smolvla_minimal_inference_check.py
```

Both scripts should complete without errors and print a final `✓` summary.

🎉 **Environment is now good to go. Time to run some attacks!**

---

## 📂 Project Structure

```
project_Kintsugi/
├── environment.yml                                  # Conda environment spec (pinned versions)
├── phase_0/                                         # phase0 scripts
│   └── phase0_smolvla_minimal_inference_check.py    # Verifies full SmolVLA inference pipeline
│   └── phase0_mujoco_robosuite_check.py             # Verifies MuJoCo + Robosuite installation
│   └── check_script_outputs/                        # JSON outputs from stack verification runs
│       └── phase0_1_results.json
└── README.md                                        # This file
```

---

## ⚠️ Important Notes

### Why Robosuite 1.4.1?

This project pins Robosuite to **1.4.1** specifically.

| Version | Status |
|---|---|
| Robosuite 1.4.1 | ✅ Compatible with LIBERO |
| Robosuite 1.5.0+ | ❌ Breaks LIBERO (`SingleArmEnv` was removed) |

`environment.yml` ensures you always get the correct version.

### Why pyav instead of torchcodec?

torchcodec requires FFmpeg shared libraries (`.dylib`) that are not correctly linked in conda environments on macOS ARM, even when FFmpeg itself is installed. pyav wraps the same underlying FFmpeg but resolves cleanly. On a CUDA machine you would switch back to torchcodec for faster video decoding during training.

### SmolVLA + LIBERO Action Space Mismatch

`smolvla_base` (the pre-trained checkpoint) was not trained on LIBERO data. Running it zero-shot on LIBERO observations produces a 6-DOF output instead of the expected 7-DOF, because the action and state spaces don't align. This is expected and confirms that **fine-tuning smolvla on LIBERO data is required** before outputs are meaningful. The verification script confirms the pipeline runs end-to-end; it does not validate action quality.

---

## 🤝 Contributing & Collaboration

If you install any new packages into the environment, **always update `environment.yml` before pushing** so others can avail your exact setup:

```bash
mamba env export --no-builds > environment.yml
```
The **--no-builds** flag strips platform-specific build strings (e.g. py310h1234_0) that would cause the environment to fail on someone else's machine. Without it, the export is often not reproducible across different OS versions or hardware.