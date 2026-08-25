#!/usr/bin/env python3
"""Summarise instrumented test failures for a commit comment.

The emulator step often dies before Gradle writes its text reports, so the
JUnit XML is the reliable source: it names the failing test and carries the
stack. Falls back to the raw log when there is no XML at all.
"""
import glob
import sys
import xml.etree.ElementTree as ET

def main() -> None:
    failures = 0
    total = 0
    for path in glob.glob("app/build/outputs/**/TEST-*.xml", recursive=True):
        try:
            root = ET.parse(path).getroot()
        except Exception:
            continue
        for case in root.iter("testcase"):
            total += 1
            bad = list(case.findall("failure")) + list(case.findall("error"))
            for problem in bad:
                failures += 1
                cls = case.get("classname", "?").rsplit(".", 1)[-1]
                print(f"{cls}.{case.get('name', '?')}")
                text = (problem.text or problem.get("message", "")).strip()
                print(text[:700])
                print("-" * 50)

    if total == 0:
        print("No test XML found — the run failed before tests started.")
        try:
            with open("/tmp/instr.log", encoding="utf-8", errors="replace") as fh:
                tail = fh.read().splitlines()[-60:]
            print("\n".join(tail))
        except OSError:
            print("(no gradle log either)")
    else:
        print(f"{failures} failed of {total} tests")

if __name__ == "__main__":
    main()
