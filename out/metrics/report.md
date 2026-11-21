Circuit-level noise p = 0.30%, d = 5, 5 rounds, Z memory. Timing: 1 thread(s).

| decoder | ours? | logical error / round | 1-fault fail | 2-fault fail | µs / shot | µs / round | parallel depth (mean / max) |
|---|---|---|---|---|---|---|---|
| PyMatching corr. | baseline | 4.21e-04 (42/20000) | 0.0% | 0.0% | 4.6 | 0.93 | – |
| BP+UF (ours) | yes | 5.61e-04 (56/20000) | 0.0% | 0.1% | 374.4 | 74.89 | 13.2 / 42 |
| BP+OSD (ours) | yes | 6.11e-04 (61/20000) | 0.0% | 0.3% | 748.9 | 149.79 | 57.2 / 150 |
| PyMatching | baseline | 6.92e-04 (69/20000) | 0.0% | 0.0% | 1.8 | 0.36 | – |
| MWPM (ours) | yes | 7.35e-04 (11/3000) | 0.0% | 0.0% | 1016.4 | 203.27 | – |
| Sparse Blossom (ours) | yes | 7.35e-04 (11/3000) | 0.0% | 0.0% | 200.1 | 40.02 | – |
| UF weighted | yes | 7.82e-04 (78/20000) | 0.0% | 0.0% | 8.8 | 1.75 | 4.2 / 12 |
| BP+OSD (ldpc) | baseline | 1.00e-03 (15/3000) | 0.1% | 1.3% | 1669.9 | 333.98 | – |
| Union-Find | yes | 1.10e-03 (110/20000) | 0.0% | 0.0% | 4.8 | 0.95 | 1.5 / 4 |
| Greedy+2opt | yes | 1.75e-03 (26/3000) | 0.0% | 0.7% | 332.3 | 66.45 | – |
| Greedy | yes | 3.38e-03 (50/3000) | 0.0% | 2.1% | 302.5 | 60.49 | – |
| BP | yes | 1.76e-02 (1640/20000) | 3.3% | 8.9% | 346.5 | 69.30 | 12.5 / 30 |
