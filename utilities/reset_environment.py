import os
import shutil
from pathlib import Path


# ==========================
# SETTINGS
# ==========================

CLEAR_INPUTS = True
CLEAR_RESULTS = True
CLEAR_IMAGES = True
CLEAR_MEMORY = True


ROOT = Path(__file__).parent.parent


TARGETS = [

    ROOT / "results",
    ROOT / "images"

]

if CLEAR_INPUTS:

    TARGETS.extend([

        ROOT / "data" / "generated",
        ROOT / "data" / "hidden"

    ])


if CLEAR_MEMORY:

    TARGETS.extend([

        ROOT / "results" / "novel_memory.csv",
        ROOT / "results" / "candidates.csv"

    ])


# ==========================
# DELETE
# ==========================

print()
print("="*60)
print("RESETTING ENVIRONMENT")
print("="*60)

for target in TARGETS:

    if not target.exists():
        continue


    # file
    if target.is_file():

        try:

            target.unlink()

            print(
                f"Removed file:"
                f" {target.name}"
            )

        except Exception as e:

            print(
                f"Failed:"
                f" {e}"
            )


    # directory
    else:

        try:

            shutil.rmtree(
                target
            )

            target.mkdir(
                parents=True,
                exist_ok=True
            )

            print(
                f"Reset folder:"
                f" {target.name}"
            )

        except Exception as e:

            print(
                f"Failed:"
                f" {e}"
            )

print()
print("Environment reset complete.")
print()