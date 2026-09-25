"""Run unittest with a machine-readable inventory of successes and skips."""
import argparse
import time
import unittest
from pathlib import Path

from training.common.files import write_json


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover")
    parser.add_argument("--test")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    suite = (unittest.defaultTestLoader.discover(args.discover) if args.discover
             else unittest.defaultTestLoader.loadTestsFromName(args.test))
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    write_json(args.report, {"tests": result.testsRun, "passed": result.passed,
        "skipped": [{"test": t.id(), "reason": why} for t, why in result.skipped],
        "failures": [{"test": t.id(), "traceback": why} for t, why in result.failures + result.errors],
        "seconds": time.monotonic() - started, "successful": result.wasSuccessful()})
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
