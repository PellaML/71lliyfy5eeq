#!/usr/bin/env python3
"""Bounded standard-library-only collector for this synthetic native Metal probe."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs():
    rows = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    if not 1 <= len(rows) <= 128 or len({c["id"] for c in rows}) != len(rows):
        raise ValueError("case count or duplicate identifiers")
    counts = {"ordered": 0, "equal": 0, "reversed": 0}
    for c in rows:
        if set(c) != {"id", "x", "lower", "upper"}:
            raise ValueError("unknown case field")
        for key in ("x", "lower", "upper"):
            v = c[key]
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError("input must be finite numeric data")
            if any(struct.unpack("<" + fmt, struct.pack("<" + fmt, v))[0] != v for fmt in ("e", "f")):
                raise ValueError("input must be exactly representable in both f16 and f32")
        kind = "ordered" if c["lower"] < c["upper"] else "equal" if c["lower"] == c["upper"] else "reversed"
        counts[kind] += 1
    if not all(counts.values()) or not any(c["lower"] > c["upper"] and c["upper"] < 0 for c in rows):
        raise ValueError("required finite domains absent")
    return {"status": "STATIC_INPUT_CHECK_PASSED", "native_execution": False,
            "cases": len(rows), "bounds_counts": counts,
            "inputs_exact_in_f16_and_f32": True,
            "expected_observation_rows_if_all_native_runs_complete": len(rows) * 2 * 2 * (4 + 2),
            "cases_sha256": sha(HERE / "cases.json")}


def collect(out):
    out = out.resolve()
    out.mkdir(parents=False, exist_ok=False)
    logs = out / "logs"
    source = out / "source"
    work = out / "work"
    for p in (logs, source, work):
        p.mkdir()
    for name in ("probe.metal", "probe.swift", "cases.json", "collect.py"):
        (source / name).write_bytes((HERE / name).read_bytes())
    (source / "observe.yml").write_bytes(
        (HERE.parent / ".github" / "workflows" / "observe.yml").read_bytes())
    result = {"schema": "probe-collection-v1", "native_execution": False,
              "status": "STARTED", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commands": [], "source_sha256": {p.name: sha(p) for p in source.iterdir()},
              "runner": {k: os.environ[k] for k in (
                  "GITHUB_SHA", "GITHUB_REF", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
                  "RUNNER_ARCH", "ImageOS", "ImageVersion") if k in os.environ},
              "compiler_note": "The executed Metal library uses the OS runtime compiler. CLI compiler "
                               "version is metadata only and is not asserted to identify that compiler."}

    def record(label, args, timeout=30, required=False):
        row = {"label": label, "argv": args, "timeout_seconds": timeout}
        with (logs / (label + ".stdout")).open("wb") as stdout, (logs / (label + ".stderr")).open("wb") as stderr:
            try:
                p = subprocess.Popen(args, cwd=work, stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    row["exit_code"] = p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    row.update(exit_code=124, timed_out=True)
            except OSError as error:
                stderr.write(str(error).encode("utf-8"))
                row["exit_code"] = 127
        result["commands"].append(row)
        (logs / (label + ".command.json")).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        if required and row["exit_code"] != 0:
            raise RuntimeError(label + " failed; see its preserved stdout/stderr/command receipt")
        return row["exit_code"]

    try:
        result["input_check"] = check_inputs()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("BLOCKED_NATIVE_HOST_REQUIRED: Darwin arm64 required, no CPU/software fallback")
        for label, cmd in (
            ("os", ["/usr/bin/sw_vers"]),
            ("architecture", ["/usr/bin/uname", "-m"]),
            ("cpu", ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
            ("model", ["/usr/sbin/sysctl", "-n", "hw.model"]),
            ("translated", ["/usr/sbin/sysctl", "-n", "sysctl.proc_translated"]),
            ("xcode", ["/usr/bin/xcodebuild", "-version"]),
            ("swift", ["/usr/bin/xcrun", "swiftc", "--version"]),
            ("sdk", ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"]),
            ("sdk_build", ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-build-version"]),
            ("metal_cli", ["/usr/bin/xcrun", "--sdk", "macosx", "metal", "--version"]),
        ):
            record(label, cmd)
        if not (logs / "cpu.stdout").read_text().strip().startswith("Apple"):
            raise RuntimeError("Apple CPU identity gate failed")
        if (logs / "translated.stdout").read_text().strip() not in ("", "0"):
            raise RuntimeError("Rosetta translation refused")
        host = work / "probe-host"
        record("host_build", ["/usr/bin/xcrun", "swiftc", "-O", "-target", "arm64-apple-macos14.0",
                              str(source / "probe.swift"), "-o", str(host)], timeout=120, required=True)
        result["host_executable_sha256"] = sha(host)
        rc = record("native", [str(host), str(source / "probe.metal"), str(source / "cases.json")], timeout=180)
        if (logs / "native.stdout").stat().st_size:
            try:
                native = json.loads((logs / "native.stdout").read_text())
                result["native_execution"] = native.get("native_execution") is True
                result["observation_count"] = len(native.get("observations", []))
                result["validation_failures"] = native.get("validation_failures", [])
                (out / "native.json").write_text(json.dumps(native, indent=2, sort_keys=True) + "\n")
            except (ValueError, TypeError) as error:
                raise RuntimeError("Native stdout is not a complete JSON receipt") from error
        if rc != 0:
            raise RuntimeError("Native probe failed; exact failure is retained in logs/native.stderr")
        if not result["native_execution"] or result.get("observation_count") != result["input_check"]["expected_observation_rows_if_all_native_runs_complete"]:
            raise RuntimeError("Native observation count/admission mismatch")
        result["status"] = "OBSERVED_NATIVE_FINITE_CASES_NOT_PORTABLE_SEMANTICS"
    except Exception as error:
        result.update(status="BLOCKED_OR_FAILED", error=str(error))
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "collection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Artifact paths exclude work/: no linked libraries, SDK, or host binary upload.
        allowed = sorted([p for d in (source, logs) for p in d.iterdir() if p.is_file()] +
                         [p for p in out.glob("*.json") if p.is_file()])
        manifest = {str(p.relative_to(out)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in allowed}
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "native_execution": result["native_execution"],
                          "error": result.get("error"), "evidence": str(out)}, indent=2))
    return 0 if result["status"] == "OBSERVED_NATIVE_FINITE_CASES_NOT_PORTABLE_SEMANTICS" else 1


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(check_inputs(), indent=2))
        return 0
    return collect(args.output)


if __name__ == "__main__":
    sys.exit(main())
