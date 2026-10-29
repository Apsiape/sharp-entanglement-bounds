"""Exact check of the archived L4M8 linear game against its source presentation.

The script loads the archived
L4M8-LINEAR-GAME.json itself and checks, exactly, in the negative quotient of the source L48 (four lamps, m = 8), that every
variable used in a row squares to the identity, that every row multiplies to its sign, and that the variables of every row
commute; a negative control flips the sign of one block of one variable and must be detected. Each variable is stored as a block-monomial matrix (`row_to_column`, entries with a sign and a word in the source
generators); products are formed with the block-monomial arithmetic of verify/check_linear_games.py, and a word is decided in
the negative quotient by the normal-form engine `l48_scalar` of that file (normal forms in the base group and Britton's lemma
for the HNN extension by x). The check therefore shares that engine with run_all.py; it is not an independent implementation
of the normal form. Its result is written to L4M8-EXACT-CHECKS.json beside this file.

Usage: python -I -B L4M8-EXACT.py
"""
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, VERIFY)
import check_linear_games as exact  # noqa: E402


def identity(m):
    return m.p == tuple(range(m.n)) and all(exact.l48_scalar(w) == sign for sign, w in m.w)


def main():
    start = time.monotonic()
    with open(os.path.join(HERE, "L4M8-LINEAR-GAME.json"), encoding="utf-8-sig") as f:
        game = json.load(f)
    mats = {name: exact.Mono(rec["row_to_column"], [(e["sign"], e["word"]) for e in rec["entries"]])
            for name, rec in game["variables"].items()}
    used = sorted({name for row in game["rows"] for name in row["variables"]})
    counts = {"squares": 0, "rows": 0, "row_commutators": 0}
    failures = []
    for name in used:
        counts["squares"] += 1
        if not identity(mats[name] @ mats[name]):
            failures.append(["square", name])
    for i, row in enumerate(game["rows"]):
        product = exact.prod([mats[name] for name in row["variables"]], game["amplification"])
        if row["product_sign"] < 0:
            product = product.neg()
        counts["rows"] += 1
        if not identity(product):
            failures.append(["row", i])
        for a, b in itertools.combinations(row["variables"], 2):
            counts["row_commutators"] += 1
            if not identity(mats[a] @ mats[b] @ mats[a].star() @ mats[b].star()):
                failures.append(["commutator", i, a, b])
    # negative control: flip the sign of one block of one used variable; the square of that variable must then fail
    victim = used[0]
    rec = game["variables"][victim]
    entries = [(e["sign"], e["word"]) for e in rec["entries"]]
    entries[0] = (-entries[0][0], entries[0][1])
    bad = exact.Mono(rec["row_to_column"], entries)
    control_detected = not identity(bad @ bad) or any(
        not identity(exact.prod([bad if v == victim else mats[v] for v in row["variables"]], game["amplification"]).neg()
                     if row["product_sign"] < 0 else
                     exact.prod([bad if v == victim else mats[v] for v in row["variables"]], game["amplification"]))
        for row in game["rows"] if victim in row["variables"])
    if not control_detected:
        failures.append(["negative control not detected", victim])
    result = {"status": "PASS" if not failures else "FAIL", "counts": counts, "failures": failures[:50],
              "negative_control": {"variable": victim, "detected": control_detected},
              "number_of_failures": len(failures), "input": "L4M8-LINEAR-GAME.json",
              "normal_form_engine": "verify/check_linear_games.py, l48_scalar",
              "seconds": round(time.monotonic() - start, 2)}
    with open(os.path.join(HERE, "L4M8-EXACT-CHECKS.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
