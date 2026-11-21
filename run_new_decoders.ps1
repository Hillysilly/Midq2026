# Benchmarks for the decoders we wrote from scratch, against the library baselines.
# Writes out/ours_circuit/ and out/ours_code_capacity/ (resumable, like run_experiments.ps1).
# Usage:  powershell -ExecutionPolicy Bypass -File run_new_decoders.ps1      (~1 hour on 12 cores)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

# 1. Circuit-level noise, d = 3, 5, 7: our decoders vs PyMatching (plain and correlated)
& $py bench.py --name ours_circuit --noise circuit --distances 3 5 7 `
    --ps 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.01 0.012 `
    --decoders pymatching pymatching_correlated union_find union_find_weighted bp_uf bp_osd `
    --max-shots 40000 --max-errors 200

# 2. Code capacity (compare with the review's Fig. 27: BP+OSD 13.9%, MWPM 14.0%, UF 11.6%)
& $py bench.py --name ours_code_capacity --noise code-capacity --distances 5 9 13 `
    --ps 0.08 0.10 0.12 0.13 0.14 0.15 0.16 0.18 `
    --decoders pymatching union_find union_find_weighted bp_uf bp_osd --max-shots 30000 --max-errors 300

# 3. Metrics table + plots for the talk
& $py metrics_report.py
