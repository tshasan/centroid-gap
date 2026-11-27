#!/usr/bin/env python3
"""Replay the pinned Lean 4.24 IVF proof export in Nanoda, with a negative control."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def replay(exporter, checker, directory):
    project = Path(__file__).resolve().parents[1]
    modules = ["IndexRecall.IVFGap", "IndexRecall.PlanarIVF"]
    roots = [
        line.split()[-1]
        for module in modules
        for line in (project / (module.replace(".", "/") + ".lean")).read_text().splitlines()
        if line.startswith("#print axioms ")
    ] + ["IndexRecall.PlanarGap.database_size"]
    if len(roots) != 16 or len(set(roots)) != len(roots):
        raise ValueError("expected 16 distinct theorem roots")
    valid, invalid = (directory / name for name in ("proof.ndjson", "control.ndjson"))
    with valid.open("w") as output:
        subprocess.run(["lake", "env", exporter, *modules, "--", *roots],
                       cwd=project, stdout=output, check=True)

    allowed = {"propext", "Classical.choice", "Quot.sound"}
    names, seen_roots, axioms = {0: ""}, set(), set()
    true_intro = None
    replacements = 0
    with valid.open() as source, invalid.open("w") as control:
        for line in source:
            record = json.loads(line)
            if "str" in record or "num" in record:
                part = record.get("str", record.get("num"))
                prefix = names[part["pre"]]
                suffix = str(part.get("str", part.get("i")))
                names[record["in"]] = f"{prefix}.{suffix}" if prefix else suffix
            elif "const" in record and names[record["const"]["name"]] == "True.intro":
                true_intro = record["ie"]
            elif "axiom" in record:
                axioms.add(names[record["axiom"]["name"]])
            elif "thm" in record:
                theorem = names[record["thm"]["name"]]
                if theorem in roots:
                    seen_roots.add(theorem)
                if theorem == "IndexRecall.FlatScan.balanced_work_cap":
                    if true_intro is None:
                        raise ValueError("missing True.intro in export")
                    record["thm"]["value"] = true_intro
                    line = json.dumps(record, separators=(",", ":")) + "\n"
                    replacements += 1
            control.write(line)
    if seen_roots != set(roots) or axioms != allowed or replacements != 1:
        raise ValueError("unexpected proof roots, axioms or negative control")

    config = directory / "nanoda.json"
    pretty_output = directory / "axioms.txt"
    pretty_output.touch()
    for path in (valid, invalid):
        config.write_text(json.dumps({
            "export_file_path": str(path),
            "permitted_axioms": sorted(allowed),
            "unpermitted_axiom_hard_error": True,
            "nat_extension": True,
            "string_extension": True,
            "print_success_message": True,
            "pp_output_path": str(pretty_output),
        }))
        result = subprocess.run([checker, config], capture_output=True, text=True)
        if path == valid:
            if result.returncode or "declarations with no errors" not in result.stdout:
                raise RuntimeError(f"Nanoda rejected the valid export:\n{result.stdout}{result.stderr}")
            print(result.stdout.strip(), flush=True)
        elif result.returncode == 0 or "self.def_eq(u, v)" not in result.stderr:
            raise RuntimeError("the corrupted cap did not fail the expected type check")
    print(f"{len(roots)} roots; corrupted cap rejected; SHA-256 {hashlib.sha256(valid.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lean4export", type=Path)
    parser.add_argument("nanoda_bin", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as scratch:
        replay(args.lean4export.resolve(), args.nanoda_bin.resolve(), Path(scratch))
