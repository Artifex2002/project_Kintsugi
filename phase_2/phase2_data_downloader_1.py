import os
from pathlib import Path
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

SUITES = {
    "libero_spatial??": range(0, 10),
    "libero_object??": range(10, 20),
    "libero_goal??": range(20, 30),
    "libero_10??": range(30, 40),
}

def get_suite_name(task_index):
    for suite, r in SUITES.items():
        if task_index in r:
            return suite
    return None


def save_image(img, path):
    img.save(path)


def download_libero_fast(base_dir="datasets_1", per_suite=250, interval=35, workers=8):

    os.makedirs(base_dir, exist_ok=True)

    suite_counts = {k: 0 for k in SUITES}
    total_target = per_suite * len(SUITES)
    total_saved = 0
    stream_index = 0

    print("Loading dataset stream...")

    dataset = load_dataset(
        "HuggingFaceVLA/libero",
        split="train",
        streaming=True
    ).shuffle(buffer_size=10000)

    executor = ThreadPoolExecutor(max_workers=workers)

    for item in dataset:

        if total_saved >= total_target:
            break

        stream_index += 1

        # enforce frame spacing
        if stream_index % interval != 0:
            continue

        task_idx = item.get("task_index", None)
        if task_idx is None:
            continue

        suite = get_suite_name(task_idx)

        if suite is None:
            continue

        if suite_counts[suite] >= per_suite:
            continue

        img = item.get("observation.images.image") or item.get("image")

        if img is None:
            continue

        suite_dir = Path(base_dir) / suite
        suite_dir.mkdir(parents=True, exist_ok=True)

        frame_id = suite_counts[suite]

        img_path = suite_dir / f"frame_{frame_id:04d}.png"

        executor.submit(save_image, img, img_path)

        suite_counts[suite] += 1
        total_saved += 1

        if total_saved % 50 == 0:
            print(f"Saved {total_saved}/{total_target}")

    executor.shutdown(wait=True)

    print("\nDownload complete.")
    print("Final distribution:")

    for k, v in suite_counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    download_libero_fast(
        base_dir="datasets",
        per_suite=250,
        interval=15,
        workers=8
    )