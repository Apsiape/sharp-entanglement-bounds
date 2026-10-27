"""Runs every check in turn, one process at a time, and writes its output to logs/<name>.log.
Usage: python -I run_all.py"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = ["check_constants", "check_lemmas", "check_count", "check_chsh", "check_return", "check_compiler",
          "check_universal", "check_rates", "check_additive", "check_coefficients", "check_linear_games",
          "check_transport", "check_reservoir", "check_tree_filling"]


def main():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    failed = []
    for name in CHECKS:
        start = time.monotonic()
        res = subprocess.run([sys.executable, "-I", os.path.join(HERE, name + ".py")], capture_output=True,
                             text=True, env=env, cwd=HERE, timeout=600)
        with open(os.path.join(HERE, "logs", name + ".log"), "w", encoding="utf-8", newline="\n") as f:
            f.write(res.stdout)
            if res.stderr:
                f.write("\n[stderr]\n" + res.stderr)
        status = "passed" if res.returncode == 0 else "FAILED"
        if res.returncode:
            failed.append(name)
        print(f"{name:18s} {status}  ({time.monotonic() - start:.1f} s)")
    print("all checks passed" if not failed else "failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
