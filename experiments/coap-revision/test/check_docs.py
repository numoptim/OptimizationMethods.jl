#!/usr/bin/env python3
"""Gate-4 check: every repo path and script mentioned in CHTC_RUNBOOK.md / ANALYSIS.md exists.
Run from experiments/coap-revision:  python3 test/check_docs.py"""
import os, re, subprocess, sys

DOCS = ["CHTC_RUNBOOK.md", "ANALYSIS.md"]
# files that are created by the workflow itself, not in the repository
GENERATED = re.compile(r"^(results/(raw|merged|paper|raw_invalid)|chtc/(depot\.tar\.gz|code\.tar\.gz|julia-|logs|resubmit|jobs_\d+_todo)|merged\.tar\.gz|depot\.tar\.gz|build_env\.sh$)")
PATH_RE = re.compile(r"(?<![\w/.-])((?:chtc|src|config|test|results|analysis)/[A-Za-z0-9_./-]+|(?:CHTC_RUNBOOK|ANALYSIS|OURS_vs_Algorithm1|STATUS|PROMPT)\.md)")
missing, checked = [], 0
for doc in DOCS:
    text = open(doc).read()
    for m in sorted(set(PATH_RE.findall(text))):
        p = m.rstrip(".,:;)`")
        if "<" in p or "*" in p or p.endswith("/"):
            continue
        if GENERATED.match(p):
            continue
        checked += 1
        if not os.path.exists(p):
            missing.append((doc, p))
# every `bash chtc/x.sh` / `julia ... src/x.jl` command must refer to an existing file; shell scripts must parse
for doc in DOCS:
    for line in open(doc):
        for m in re.finditer(r"bash (chtc/[\w.]+\.sh)", line):
            f = m.group(1)
            if f == "chtc/build_env.sh":
                continue
            if not os.path.isfile(f):
                missing.append((doc, f))
            elif subprocess.run(["bash", "-n", f]).returncode != 0:
                missing.append((doc, f + " (syntax error)"))
for sub in ["chtc/build.sub", "chtc/jobs_1.sub", "chtc/pilot.sub"]:
    for line in open(sub):
        m = re.match(r"(executable|transfer_input_files)\s*=\s*(.*)", line)
        if m:
            for f in [x.strip() for x in m.group(2).split(",")]:
                if f and not GENERATED.match(f) and not os.path.exists(f):
                    missing.append((sub, f))
print(f"checked {checked} paths in {DOCS}")
if missing:
    for d, p in missing:
        print("MISSING:", d, p)
    sys.exit(1)
print("all paths and scripts exist")
