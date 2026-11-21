# Re-run every experiment behind RESULTS.md. Each writes to out/<name>/ and resumes if interrupted
# (finished points are read back from out/<name>/stats.csv and not re-simulated).
# Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File run_experiments.ps1
# Total time on a 12-core laptop: roughly 1-2 hours.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

# 1. Code-capacity thresholds (compare with the review's Fig. 27)
& $py bench.py --name code_capacity --noise code-capacity --distances 5 9 13 `
    --ps 0.08 0.10 0.12 0.13 0.14 0.15 0.16 0.18 `
    --decoders pymatching pymatching_correlated union_find union_find_weighted --max-shots 100000 --max-errors 500

# 2. Phenomenological noise (noisy measurements, no circuit details)
& $py bench.py --name phenomenological --noise phenomenological --distances 5 9 13 `
    --ps 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 `
    --decoders pymatching union_find_weighted --max-shots 100000 --max-errors 400

# 3. Circuit-level noise: decoder shoot-out (belief-matching is slow, so only d <= 5 and fewer shots)
& $py bench.py --name circuit_decoders --noise circuit --distances 3 5 7 `
    --decoders pymatching pymatching_correlated union_find union_find_weighted --max-shots 50000 --max-errors 300
& $py bench.py --name circuit_decoders --noise circuit --distances 3 5 `
    --decoders belief_matching --max-shots 20000 --max-errors 200

# 4. Biased circuit-level noise (eta = 100): CSS vs XZZX, both memory bases
& $py bench.py --name bias100 --noise circuit --bias 100 --variants css xzzx --basis both `
    --distances 3 5 7 --ps 0.002 0.003 0.004 0.005 0.006 0.008 0.01 0.012 0.015 `
    --decoders pymatching --max-shots 200000 --max-errors 300

# 5. Threshold vs bias (code capacity)
& $py bias_sweep.py

# 6. Superconducting-like noise (SI1000): Lambda at realistic error rates, compare with Google's 2024 Lambda ~ 2.1
& $py bench.py --name si1000 --noise si1000 --distances 3 5 7 --ps 0.001 0.0015 0.002 0.003 0.004 0.005 `
    --decoders pymatching pymatching_correlated --max-shots 1000000 --max-errors 300

# 7. Sliding-window (real-time) decoding, then decoder speed. Both time things, so they run last on an idle CPU.
& $py realtime.py
& $py speed.py

# 8. Copy the results into the interactive explainer
& $py tools\export_explainer_data.py
