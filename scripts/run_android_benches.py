#!/usr/bin/env python3
"""Run the example benchmarks on an Android device and pull one bench.json per run.

Needs the RELEASE build of the app installed (README-android.md) and the device
awake and unlocked. Conditions are the ones in docs/benchmarks.md ("Measurement
conditions"): each run starts the app with no viewer, waits for thermal status 0,
and rests afterwards. Completion comes from logcat, so nothing polls the UI.

Next to each bench.json, device_state.json records the power source, thermal
status, and battery temperature at the start of the run.

Usage:
    python scripts/run_android_benches.py [-s serial] [-o out_root]
        [--reps 3] [--only label,label] [--cool 90] [-l]
    python scripts/bench_summary.py bench_results/android
"""

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PACKAGE = "org.dressi.examples"
# out-dir name, timeout in seconds
ENTRIES = [
    ("image_fitting", 120),
    ("texture_optimization", 400),
    ("silhouette_optimization_hardsoftras", 200),
    ("silhouette_optimization_aa", 200),
    ("pbr_shading", 120),
    ("pbr_material_optimization", 400),
    ("pbr_envmap_optimization", 1100),
]
THERMAL_WAIT_S = 600
FINISHED = re.compile(r"== (\w+) finished with exit code (-?\d+) ==")


def find_adb():
    adb = shutil.which("adb")
    if adb:
        return adb
    for root in (os.environ.get("ANDROID_HOME"),
                 os.environ.get("ANDROID_SDK_ROOT"),
                 os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk")):
        if root:
            for name in ("adb.exe", "adb"):
                path = Path(root) / "platform-tools" / name
                if path.exists():
                    return str(path)
    sys.exit("error: adb not found")


class Device:
    def __init__(self, serial):
        self.base = [find_adb()] + (["-s", serial] if serial else [])

    def shell(self, *args):
        return subprocess.run(self.base + ["shell", *args], capture_output=True,
                              text=True, errors="replace", timeout=60).stdout

    def thermal_status(self):
        m = re.search(r"Thermal Status: (\d+)",
                      self.shell("dumpsys", "thermalservice"))
        return int(m.group(1)) if m else -1

    def state(self):
        battery = self.shell("dumpsys", "battery")
        power = [name for name, key in (("ac", "AC powered"),
                                        ("usb", "USB powered"),
                                        ("wireless", "Wireless powered"))
                 if re.search(key + r": true", battery)]
        temp = re.search(r"temperature: (\d+)", battery)
        return {"power": power[0] if power else "battery",
                "thermal_status": self.thermal_status(),
                "battery_temp_c": int(temp.group(1)) / 10 if temp else None}

    def wait_cool(self):
        deadline = time.time() + THERMAL_WAIT_S
        while self.thermal_status() > 0:
            if time.time() > deadline:
                sys.exit("error: device did not return to thermal status 0")
            time.sleep(15)

    def logcat(self):
        """Follows dressi's log from now on; lines arrive on the queue."""
        now = self.shell("date", "+%m-%d %H:%M:%S.000").strip()
        proc = subprocess.Popen(self.base + ["logcat", "-T", now, "-s", "dressi"],
                                stdout=subprocess.PIPE, text=True,
                                errors="replace")
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout],
                         daemon=True).start()
        return proc, lines


def run(device, root, entry, tag, timeout):
    device.wait_cool()
    device.shell("am", "force-stop", PACKAGE)
    device.shell("run-as", PACKAGE, "rm", "-f", f"files/out/{entry}/bench.json")
    state = device.state()
    start = time.time()
    proc, lines = device.logcat()
    device.shell("am", "start", "-n", f"{PACKAGE}/.MainActivity",
                 "-e", "run", entry, "--ez", "no_view", "true")
    code = "timeout"
    while time.time() - start < timeout:
        try:
            m = FINISHED.search(lines.get(timeout=5))
        except queue.Empty:
            if not device.shell("pidof", PACKAGE).strip():
                code = "crashed"
                break
            continue
        if m:
            code = int(m.group(2))
            break
    proc.terminate()

    dst = root / tag
    shutil.rmtree(dst, ignore_errors=True)
    if code == 0:
        dst.mkdir(parents=True)
        bench = device.shell("run-as", PACKAGE, "cat",
                             f"files/out/{entry}/bench.json")
        (dst / "bench.json").write_text(bench, encoding="utf-8")
        (dst / "device_state.json").write_text(json.dumps(state) + "\n",
                                               encoding="utf-8")
    print(f"{tag}: exit {code}, {time.time() - start:.0f}s, {state}", flush=True)
    return code == 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-s", "--serial", default="",
                   help="adb serial (default: the only connected device)")
    p.add_argument("-o", "--out-root", default="bench_results/android")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only", default="",
                   help="comma-separated entry substrings to keep")
    p.add_argument("--cool", type=int, default=90,
                   help="seconds to rest after each run")
    p.add_argument("-l", "--list", action="store_true",
                   help="list the entries and exit")
    args = p.parse_args()

    pats = [s for s in args.only.split(",") if s]
    entries = [e for e in ENTRIES
               if not pats or any(s in e[0] for s in pats)]
    if args.list:
        print("\n".join(entry for entry, _ in entries))
        return 0

    device = Device(args.serial)
    root = Path(args.out_root)
    ok = True
    for rep in range(1, args.reps + 1):
        for entry, timeout in entries:
            ok &= run(device, root, entry, f"run{rep}/{entry}", timeout)
            time.sleep(args.cool)
    device.shell("am", "force-stop", PACKAGE)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
