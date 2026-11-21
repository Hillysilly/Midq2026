# Extra points that bracket the thresholds found in run_experiments.ps1 (results merge into the same out/ folders).
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

& $py bench.py --name circuit_decoders --noise circuit --distances 3 5 7 --ps 0.014 0.016 0.018 `
    --decoders pymatching pymatching_correlated union_find union_find_weighted --max-shots 50000 --max-errors 300
& $py bench.py --name bias100 --noise circuit --bias 100 --variants css xzzx --basis both `
    --distances 3 5 7 --ps 0.02 0.025 0.03 0.035 --decoders pymatching --max-shots 200000 --max-errors 300

# timing-sensitive runs last, on an idle CPU
& $py realtime.py
& $py speed.py
& $py tools\export_explainer_data.py
