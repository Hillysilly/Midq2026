Circuit-level noise p = 0.30%, d = 5, 5 rounds, Z memory. Timing: 1 thread(s).

| decoder | ours? | logical error / round | 1-fault fail | 2-fault fail | µs / shot | µs / round | parallel depth (mean / max) |
|---|---|---|---|---|---|---|---|
| PyMatching corr. | baseline | 4.21e-04 (42/20000) | 0.0% | 0.0% | 4.7 | 0.93 | – |
| BP+UF (ours) | yes | 5.61e-04 (56/20000) | 0.0% | 0.1% | 394.9 | 78.99 | 13.2 / 42 |
| BP+OSD (ours) | yes | 6.11e-04 (61/20000) | 0.0% | 0.3% | 763.5 | 152.69 | 57.2 / 150 |
| PyMatching | baseline | 6.92e-04 (69/20000) | 0.0% | 0.0% | 1.7 | 0.34 | – |
| MWPM (ours) | yes | 7.35e-04 (11/3000) | 0.0% | 0.0% | 1008.9 | 201.78 | – |
| UF weighted | yes | 7.82e-04 (78/20000) | 0.0% | 0.0% | 10.1 | 2.03 | 4.2 / 12 |
| BP+OSD (ldpc) | baseline | 1.00e-03 (15/3000) | 0.1% | 1.3% | 1672.8 | 334.56 | – |
| Union-Find | yes | 1.10e-03 (110/20000) | 0.0% | 0.0% | 5.6 | 1.12 | 1.5 / 4 |
| Greedy+2opt | yes | 1.75e-03 (26/3000) | 0.0% | 0.7% | 324.2 | 64.84 | – |
| Greedy | yes | 3.38e-03 (50/3000) | 0.0% | 2.1% | 302.7 | 60.53 | – |
| BP | yes | 1.76e-02 (1640/20000) | 3.3% | 8.9% | 336.6 | 67.32 | 12.5 / 30 |
