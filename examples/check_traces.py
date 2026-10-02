"""Checks that every example's trace shows progress to its end: the frames after the last one
in which what the page plots changes are cut, so no trace ends with frames that repeat it."""

import glob
import json
import sys

# the examples whose numbers depend on the machine
SKIP = {"asynchronous", "bo_asynchronous"}

failed = []
for path in sorted(glob.glob("examples/*/trace.json")):
    name = path.split("/")[-2]
    if name in SKIP:
        continue
    with open(path, encoding="utf-8") as file:
        frames = json.load(file)["frames"]
    plotted = [
        json.dumps([frame.get("best"), frame.get("median"), frame.get("state"), frame.get("series")])
        for frame in frames
    ]
    repeats = 0
    while repeats + 1 < len(plotted) and plotted[-2 - repeats] == plotted[-1]:
        repeats += 1
    if repeats:
        failed.append(f"{name}: the last {repeats + 1} frames plot the same")
if failed:
    print("traces with a frozen end (regenerate them with GENOXIDE_TRACE):")
    print("\n".join(failed))
    sys.exit(1)
print(f"{len(glob.glob('examples/*/trace.json')) - len(SKIP)} traces end where their plot stops changing")
