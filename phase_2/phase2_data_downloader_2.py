import os
from pathlib import Path
from datasets import load_dataset

BASE_DIR = "datasets_2"
FRAMES_PER_TASK = 50
SKIP_BETWEEN = 49   # skip 49 → save every 50th
INTERVAL = SKIP_BETWEEN + 1

TASKS = list(range(0, 40, 2))  # 0,2,4,...,38


def main():

    os.makedirs(BASE_DIR, exist_ok=True)

    print("Loading dataset stream...")
    dataset = load_dataset(
        "HuggingFaceVLA/libero",
        split="train",
        streaming=True
    )

    # counters per task
    saved = {t: 0 for t in TASKS}
    seen = {t: 0 for t in TASKS}

    for item in dataset:

        task = item.get("task_index")

        if task not in TASKS:
            continue

        if saved[task] >= FRAMES_PER_TASK:
            continue

        seen[task] += 1

        # enforce skipping
        if seen[task] % INTERVAL != 0:
            continue

        img = item.get("observation.images.image") or item.get("image")
        if img is None:
            continue

        task_dir = Path(BASE_DIR) / f"task_{task}"
        task_dir.mkdir(parents=True, exist_ok=True)

        path = task_dir / f"frame_{saved[task]:03d}.png"
        img.save(path)

        saved[task] += 1

        print(f"Task {task}: saved {saved[task]}/50")

        # stop if everything finished
        if all(saved[t] >= FRAMES_PER_TASK for t in TASKS):
            break

    print("Done.")


if __name__ == "__main__":
    main()