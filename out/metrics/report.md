Circuit-level noise p = 0.30%, d = 5, 5 rounds, Z memory. Timing: 1 thread(s).

| decoder | ours? | logical error / round | 1-fault fail | 2-fault fail | µs / shot | µs / round | parallel depth (mean / max) |
|---|---|---|---|---|---|---|---|
| PyMatching corr. | baseline | 4.21e-04 (42/20000) | 0.0% | 0.0% | 5.0 | 1.00 | – |
| BP+UF (ours) | yes | 5.61e-04 (56/20000) | 0.0% | 0.1% | 370.4 | 74.07 | 13.2 / 42 |
| BP+OSD (ours) | yes | 6.11e-04 (61/20000) | 0.0% | 0.3% | 756.9 | 151.38 | 57.2 / 150 |
| PyMatching | baseline | 6.92e-04 (69/20000) | 0.0% | 0.0% | 1.8 | 0.36 | – |
| MWPM (ours) | yes | 7.35e-04 (11/3000) | 0.0% | 0.0% | 1023.0 | 204.60 | – |
| Sparse Blossom (ours) | yes | 7.35e-04 (11/3000) | 0.0% | 0.0% | 208.2 | 41.63 | – |
| UF weighted | yes | 7.82e-04 (78/20000) | 0.0% | 0.0% | 8.8 | 1.77 | 4.2 / 12 |
| Local BP+UF (ours) | yes | 8.03e-04 (80/20000) | 0.0% | 0.0% | 163.6 | 32.73 | 5.8 / 33 |
| BP+OSD (ldpc) | baseline | 1.00e-03 (15/3000) | 0.1% | 1.3% | 1642.6 | 328.51 | – |
| Union-Find | yes | 1.10e-03 (110/20000) | 0.0% | 0.0% | 4.7 | 0.94 | 1.5 / 4 |
| Greedy+2opt | yes | 1.75e-03 (26/3000) | 0.0% | 0.7% | 319.1 | 63.82 | – |
| Greedy | yes | 3.38e-03 (50/3000) | 0.0% | 2.1% | 308.5 | 61.69 | – |
| BP | yes | 1.76e-02 (1640/20000) | 3.3% | 8.9% | 331.4 | 66.28 | 12.5 / 30 |
