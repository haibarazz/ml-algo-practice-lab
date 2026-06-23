#!/usr/bin/env python3
"""Run every module-level tests.py script."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = [
    ROOT / "ml_dl_handwriting",
    ROOT / "projects" / "minimind",
]


def iter_tests() -> list[Path]:
    tests: list[Path] = []
    for root in TEST_ROOTS:
        for path in sorted(root.rglob("tests.py")):
            if "source" in path.relative_to(ROOT).parts:
                continue
            tests.append(path)
    return tests


def run_test(path: Path, timeout: int) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode == 0, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all module tests.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per tests.py in seconds.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failing test.")
    args = parser.parse_args()

    tests = iter_tests()
    failures: list[tuple[Path, str]] = []

    for index, path in enumerate(tests, start=1):
      rel = path.relative_to(ROOT)
      print(f"[{index}/{len(tests)}] {rel.as_posix()}")
      try:
          ok, output = run_test(path, args.timeout)
      except subprocess.TimeoutExpired:
          ok, output = False, f"Timed out after {args.timeout}s"

      if not ok:
          failures.append((rel, output))
          print(output)
          if args.fail_fast:
              break

    if failures:
        print("\nFailing module tests:")
        for rel, output in failures:
            print(f"- {rel.as_posix()}")
            if output:
                print(output)
        print(f"\nPassed: {len(tests) - len(failures)}/{len(tests)}")
        return 1

    print(f"\nAll module tests passed: {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
