"""Replays the checks of the larger exported objects in verify/artifacts, one process at a time.

Usage: python -I run_artifacts.py

Single-threaded and moderate: about eighty seconds in all on a laptop. Each step
runs in its own process with the BLAS/OpenMP thread counts set to one; its output goes to
logs/artifacts/<step>.log, and the checkers write their result files beside themselves.

  1. every input file is compared with its SHA-256 in artifacts/SHA256SUMS;
  2. the point-Pauli Boolean game (point_game/BOOLEAN-GAME.json): its source relators are compared
     with the 22 relators printed in the paper, its counts are checked (6960 variables, 5713 contexts,
     arity at most three, amplification 12), and VERIFY-BOOLEAN.py checks exactly, against those
     relators, every variable square, context commutator and forbidden assignment and the six
     recovery maps;
  3. the L4M8 linear game (l4m8/L4M8-LINEAR-GAME.json): 6798 game variables, the central J being
     declared but in no row, 5653 rows, maximum arity five, amplification 16; L4M8-EXACT.py checks the
     archived file exactly against its source presentation (every square, row and row commutator, with
     the normal-form engine of check_linear_games.py, and a negative control); L4M8-PLAY.py plays complete
     Fourier-lamp strategies on every context at four sizes, in floating point, and reports the observed
     rank exponents;
  4. the two-copy linear game (two_copy/TWO-COPY-FINAL-GAME.json): 199129 game variables, all used,
     with no J, 164608 rows, maximum arity nine, amplification 64; EXACT.py checks every square, row and
     row commutator in the negative source algebra, and PLAY.py plays every context and raw outcome
     at two sizes;
  5. the selector construction of the undecidability theorem on two small inputs, with the L4M8 game
     in place of Slofstra's fallback game: BUILD-SELECTOR.py builds the two games and AUDIT-SELECTOR.py
     checks every local predicate, the incidence weights and one finite perfect strategy;
  6. exact-source presentations: CHECK-NATIVE.py exports the subset-lamp source for k = 2 (four
     generators, 23 relators) and runs bounded exact controls of its words, addresses and cocycles;
     CHECK-COMPACT.py reconstructs the literal 21-relator replication source and runs bounded exact
     block-action and cocycle controls.

Finite checks support the proofs in the paper; they do not replace them.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
LOGS = os.path.join(HERE, "logs", "artifacts")

# The 22 relators of the point source as printed in the paper (section on the point-Pauli game), with
# generators a, b, s, x, z, J numbered 1..6 and inverses negative; [u, v] = u v u^-1 v^-1.
A, B, S, X, Z, J = 1, 2, 3, 4, 5, 6


def inv(w):
    return [-g for g in reversed(w)]


def comm(u, v):
    return u + v + inv(u) + inv(v)


PAPER_RELATORS = (
    [[S, S], [S, A, S, -A] * 3, comm([S], [A, A, S, -A, -A]), [A, B, -A, -B, -S], [A, S, -A, B, -S, -B],
     [J, J], [X, X], [Z, Z]]
    + [comm([J], [g]) for g in (A, B, S, X, Z)]
    + [comm([X], [Z]) + [-J]]
    + [comm([u], [S, g]) for u in (X, Z) for g in (A, B)]
    + [comm([u], [A, v, -A]) for u in (X, Z) for v in (X, Z)]
)


class Failure(Exception):
    pass


def need(cond, message):
    if not cond:
        raise Failure(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(rel):
    with open(os.path.join(ART, rel), encoding="utf-8-sig") as f:
        return json.load(f)


def run(step, rel, args=()):
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    path = os.path.join(ART, rel)
    res = subprocess.run([sys.executable, "-I", "-B", path, *args], capture_output=True, text=True, env=env,
                         cwd=os.path.dirname(path), timeout=1200)
    with open(os.path.join(LOGS, step + ".log"), "a", encoding="utf-8", newline="\n") as f:
        f.write(f"$ python -I -B {rel} {' '.join(args)}\n")
        f.write(res.stdout)
        if res.stderr:
            f.write("\n[stderr]\n" + res.stderr)
        f.write(f"[exit code {res.returncode}]\n\n")
    need(res.returncode == 0, f"{rel} exited with code {res.returncode}")


def step_hashes():
    lines = open(os.path.join(ART, "SHA256SUMS"), encoding="utf-8").read().split("\n")
    n = 0
    for line in lines:
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        need(sha256(os.path.join(ART, rel.strip())) == digest, f"hash mismatch: {rel}")
        n += 1
    return f"{n} input files match artifacts/SHA256SUMS"


def step_point_game():
    g = load("point_game/BOOLEAN-GAME.json")
    need(g["source_relators"] == PAPER_RELATORS, "exported relators differ from the paper's 22 relators")
    need(g["source_negative_generator"] == J, "marked generator is not J")
    need(len(g["variables"]) == 6960 and len(g["constraints"]) == 5713, "variable or context count")
    need(max(len(c["variables"]) for c in g["constraints"]) <= 3, "arity")
    need(g["amplification"] == 12, "amplification")
    run("point_game", "point_game/VERIFY-BOOLEAN.py")
    r = load("point_game/VERIFY-BOOLEAN.json")
    need(r["status"] == "PASS" and r["variables"] == 6960 and r["contexts"] == 5713, "VERIFY-BOOLEAN result")
    c = r["counts"]
    return (f"relators equal the paper's 22; 6960 variables, 5713 contexts, arity <= 3, amplification 12; "
            f"exact: {c['variable_squares']} squares, {c['context_commutators']} context commutators, "
            f"{c['forbidden_assignments']} forbidden assignments, {c['whole_state_recovery_generators']} recovery maps")


def step_l4m8():
    g = load("l4m8/L4M8-LINEAR-GAME.json")
    used = set()
    for row in g["rows"]:
        used.update(row["variables"])
    declared = set(g["variables"])
    need(declared - used == {"J"}, "declared but unused variables other than J")
    need(len(used) == 6798 and len(g["rows"]) == 5653, "L4M8 counts")
    need(max(len(row["variables"]) for row in g["rows"]) == 5 and g["amplification"] == 16, "L4M8 arity/amplification")
    run("l4m8", "l4m8/L4M8-EXACT.py")
    x = load("l4m8/L4M8-EXACT-CHECKS.json")
    need(x["status"] == "PASS" and x["counts"] == {"squares": 6798, "rows": 5653, "row_commutators": 16964}
         and x["negative_control"]["detected"], "L4M8 exact result")
    run("l4m8", "l4m8/L4M8-PLAY.py")
    r = load("l4m8/L4M8-PLAY-CHECKS.json")
    need(r["status"] == "PASS" and len(r["records"]) == 4, "L4M8 play result")
    ex = [round(x, 3) for x in r["empirical_rank_exponents"]]
    need(ex == [0.318, 0.331, 0.333], f"L4M8 exponents {ex}")
    return (f"6798 game variables (J declared, in no row), 5653 rows, maximum arity 5, amplification 16; "
            f"exact: 6798 squares, 5653 rows, 16964 row commutators, negative control detected; "
            f"complete play at four sizes, rank exponents {ex}")


def step_two_copy():
    g = load("two_copy/TWO-COPY-FINAL-GAME.json")
    names = g["variable_names"]
    used = set()
    for row in g["rows"]:
        used.update(row[1:])
    need(len(names) == 199129 and len(used) == 199129 and "J" not in names, "two-copy variable count")
    need(len(g["rows"]) == 164608 and max(len(row) - 1 for row in g["rows"]) == 9, "two-copy rows/arity")
    need(g["counts"]["amplification"] == 64, "two-copy amplification")
    run("two_copy", "two_copy/EXACT.py")
    r = load("two_copy/EXACT-CHECKS.json")
    need(r["status"] == "PASS" and r["counts"] == {"squares": 199129, "rows": 164608, "row_commutators": 493855},
         "two-copy exact result")
    run("two_copy", "two_copy/PLAY.py")
    p = load("two_copy/PLAY-CHECKS.json")
    need(p["status"] == "PASS" and p["all_final_contexts_visited"], "two-copy play result")
    return (f"199129 game variables (all used, no J), 164608 rows, maximum arity 9, amplification 64; exact: "
            f"199129 squares, 164608 rows, 493855 row commutators; play: {p['raw_outcome_visits']} raw outcomes "
            f"at word lengths {p['lengths']}, losses {[x['actual_game_loss'] for x in p['records']]}")


def step_selector():
    run("selector", "selector/BUILD-SELECTOR.py")
    run("selector", "selector/AUDIT-SELECTOR.py")
    r = load("selector/AUDIT-SELECTOR.json")
    need(r["status"] == "PASS" and r["fallback_sha256"] == sha256(os.path.join(ART, "l4m8/L4M8-LINEAR-GAME.json")),
         "selector audit")
    objs = ", ".join(f"{o['object']} ({o['contexts']} contexts)" for o in r["objects"])
    return f"built and audited {objs}; negative control passed"


def step_exact_sources():
    run("exact_sources", "exact_sources/CHECK-NATIVE.py")
    r = load("exact_sources/NATIVE-RESULTS.json")
    need(r["status"] == "PASS" and r["native_generators"] == 4 and r["native_relators"] == 23, "subset k=2 checks")
    need(r["source_export"]["sha256"] == sha256(os.path.join(ART, "exact_sources/SOURCE-K2.json")), "subset export hash")
    run("exact_sources", "exact_sources/CHECK-COMPACT.py")
    c = load("exact_sources/COMPACT-RESULTS.json")
    need(c["status"] == "PASS" and c["counts"]["actual_twenty_one_relators"] == 1, "replication checks")
    return (f"subset source k=2: 4 generators, 23 relators, {sum(r['exact_check_counts'].values())} exact checks; "
            f"replication source: 21 relators, {c['exact_comparisons']} exact comparisons")


STEPS = [("hashes", step_hashes), ("point_game", step_point_game), ("l4m8", step_l4m8), ("two_copy", step_two_copy),
         ("selector", step_selector), ("exact_sources", step_exact_sources)]


def main():
    os.makedirs(LOGS, exist_ok=True)
    for name, _ in STEPS:
        open(os.path.join(LOGS, name + ".log"), "w").close()
    failed = []
    for name, step in STEPS:
        start = time.monotonic()
        try:
            summary = step()
            print(f"{name:14s} passed  ({time.monotonic() - start:.1f} s)  {summary}", flush=True)
        except (Failure, KeyError, subprocess.TimeoutExpired) as exc:
            failed.append(name)
            print(f"{name:14s} FAILED  ({time.monotonic() - start:.1f} s)  {exc}", flush=True)
    print("all artifact checks passed" if not failed else "failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
