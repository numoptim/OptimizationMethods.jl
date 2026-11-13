# CHTC runbook: COAP revision experiments

Everything below runs on the **CHTC submit node** unless it says "on your laptop".
Replace only the two placeholders `<NETID>` and `<SUBMIT_HOST>` (your CHTC login). All other
commands are copy-paste. Unless stated otherwise, commands run from the directory

```
~/OptimizationMethods.jl/experiments/coap-revision
```

**What you are about to run:** 120,000 runs (120 problems x 10 starts x 100 algorithms), packed
into **535 jobs**, 1 CPU each. Worst-case job time 19.6 h (median 14.3 h; these are
upper bounds, most runs stop earlier). Worst-case total 6,600 core-hours. Requests per job:
3000 MB memory, 4 GB disk. Jobs are pinned to one CPU level (`Microarch == "x86_64-v3"`).

---------------------------------------------------------------------------------------------

## 1. Clone the branch

```bash
ssh <NETID>@<SUBMIT_HOST>
cd ~
git clone --branch cv/experiments-quasi-likelihood --single-branch https://github.com/numoptim/OptimizationMethods.jl.git
cd ~/OptimizationMethods.jl/experiments/coap-revision
git log --oneline | head -1        # write this commit id down: it identifies the code that produced the results
mkdir -p chtc/logs results/raw
```

## 2. Build the environment tarballs (once)

CHTC machines have no Julia. We ship Julia 1.13.3 and a pre-built package depot with every job.

```bash
bash chtc/make_tarballs.sh         # downloads Julia 1.13.3, packs the code of this commit
condor_submit -i chtc/build.sub    # interactive build job; wait a few minutes for the prompt
```

You are now **inside the build job**. Run:

```bash
bash build_env.sh                  # installs and precompiles the packages (about 10-15 min), prints the tarball size
exit
```

Back on the submit node:

```bash
ls -lh depot.tar.gz && mv depot.tar.gz chtc/
ls -lh chtc/*.tar.gz
```

*If `depot.tar.gz` is larger than 100 MB*, CHTC wants it served through SQUID: email
chtc@cs.wisc.edu ("Julia depot tarball for HTC jobs, needs SQUID access") and use the location
they give you in `transfer_input_files` of the `.sub` files; otherwise continue.

## 3. CHTC pilot (14 jobs, full budgets)

The pilot jobs are ordinary jobs of the final list (smallest, a middle and the largest problem), so
their output is kept and **not rerun** in step 4. Their worst case is 15.2 h, so this takes up to a day.

First check that the pinned CPU level exists in the pool (you want many `x86_64-v3` slots):

```bash
condor_status -af Microarch | sort | uniq -c
condor_status -constraint 'Microarch == "x86_64-v3"' -af CpuModel | sort | uniq -c
```

If `x86_64-v3` is missing or has few slots, pick another level that has many slots (for example
`x86_64-v2`) and switch it in the submit files with one command (replace `x86_64-v2` by your choice):

```bash
sed -i 's/Microarch == "x86_64-v3"/Microarch == "x86_64-v2"/' chtc/jobs_1.sub chtc/pilot.sub
grep -h Microarch chtc/jobs_1.sub chtc/pilot.sub
```

Every run records its CPU model (`cpu_model` in `runs.csv`), so the choice can be audited later.

Submit and wait:

```bash
condor_submit chtc/pilot.sub       # prints "14 job(s) submitted to cluster <CLUSTER_ID>"
condor_q
```

When `condor_q` shows no jobs (they finished or went on hold), check runtime, memory and disk
against the requests (`<CLUSTER_ID>` is the number printed by `condor_submit`):

```bash
bash chtc/job_runtimes.sh <CLUSTER_ID>     # prints max/mean runtime in hours, checked against 24 h
bash chtc/collect.sh --pilot               # merges and validates the finished pilot jobs
```

Pass criteria: `RUNTIME OK (<= 24 h)`, peak memory below 3000 MB, peak disk below 4 GB, and
`collect.sh --pilot` ends with `VALID`. If a criterion fails, do **not** continue; send the output
to whoever maintains the branch (the sizes are set in `src/jobs.jl`).

## 4. Submit everything

```bash
bash chtc/submit_all.sh --dry-run  # shows how many jobs will be queued (535 minus the finished pilot jobs)
bash chtc/submit_all.sh
```

## 5. Monitor, release and resubmit

```bash
condor_q                                               # running / idle / held
ls results/raw | wc -l                                 # finished job directories (of 535 in total)
condor_q -hold -af ClusterId ProcId HoldReason | head  # why jobs are held
```

* **Held, transient reason** (for example the node went away): `condor_release -all`.
* **Held for exceeding memory or disk**: send the `HoldReason` to the branch maintainer (the sizes are in `src/jobs.jl`); do not keep releasing.
* **Jobs that finished but are missing or invalid in step 6** are resubmitted from the list
  `collect` writes (below); nothing else needs to be done by hand.

## 6. Collect and copy the results off CHTC

When `condor_q` is empty:

```bash
bash chtc/collect.sh               # merges results/raw/job_* into results/merged and validates; ends with VALID
cat results/merged/resubmit_jobs.txt
```

Expected: last line `VALID`, `runs: 120000`, `hit rows: 480000`, and an empty `resubmit_jobs.txt`.
Then pack and copy (on the submit node, then on your laptop):

```bash
tar -czf merged.tar.gz results/merged
```

```bash
# on your laptop
scp <NETID>@<SUBMIT_HOST>:OptimizationMethods.jl/experiments/coap-revision/merged.tar.gz .
tar -xzf merged.tar.gz              # creates results/merged/{runs,hits,ours_events,ours_events_summary}.csv
```

Give `results/merged/` (and the commit id from step 1) to a fresh session; it follows `ANALYSIS.md`.

## 7. If `collect.sh` fails

It prints lines starting with `FAIL:` and exits with an error; it also writes
`results/merged/resubmit_jobs.txt`.

1. **Jobs missing or invalid** (the FAIL line says "job(s) missing", "missing output file(s)", "do not match
   the job specification", NaN/Inf in counts or times): rerun them and collect again.
   ```bash
   bash chtc/resubmit.sh results/merged/resubmit_jobs.txt
   # wait until condor_q is empty, then
   bash chtc/collect.sh
   ```
   Output of the old attempt is kept in `results/raw_invalid/`.
2. **A job failed with a "checksums ... differ" message in `chtc/logs/job_<id>.err`**: the random
   numbers differ from the ones used to create `config/problems.csv` (different Julia version?). Check
   `grep julia_version results/raw/job_*/runs.csv | sort | uniq -c | head`; every run must say 1.13.3.
   Do not resubmit blindly; send the message to the branch maintainer.
3. **Logical failures** ("gradient-only methods with n_obj != 0", "hit data not monotone",
   "status ... disagrees with the 1e-9 hit", "exceed their evaluation budget"): these indicate a code
   problem, not a job problem. Do not resubmit. Send the full `collect.sh` output to the branch maintainer.
4. **Rows with status `error`** are not a collection failure. Look at the messages and keep them:
   ```bash
   cat results/raw/job_*/errors.log | head -20
   ```
5. Statuses `budget_grad`, `budget_obj`, `timeout` and `stalled` are expected: they are runs that did
   not reach 1e-9 within the budget. They count as failures at the tolerances they did not reach.

## Command reference (all in `chtc/`)

| File | Purpose |
|---|---|
| `make_tarballs.sh` | Julia 1.13.3 tarball and code tarball of the current commit |
| `build.sub`, `build_env.sh` | interactive build job that creates `depot.tar.gz` |
| `run_job.sh`, `jobs_1.sub`, `jobs_1.txt` | what each of the 535 jobs runs / their list |
| `pilot.sub`, `pilot_jobs.txt` | the 14 pilot jobs |
| `submit_all.sh` | queues all jobs that do not have output yet |
| `job_runtimes.sh` | runtime, memory, disk of a finished cluster |
| `collect.sh` | merge and validate (runs `src/collect.jl` with the portable Julia) |
| `resubmit.sh` | rerun the jobs listed in `results/merged/resubmit_jobs.txt` |
