# Same-grid curriculum development progress

All points use the same 24,516 validation records and16 complete East matches (two consumed logical seeds, two opponent profiles, four seats). Parameters changed; the execution policy and grid stayed fixed.

| Metric | Step0 | Step625 | Step1250 | Step1875 | Change 1250→1875 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation objective | 1.04924 | 1.04242 | 1.02447 | 1.00642 | -0.01805 |
| Validation weighted target CE | 1.03889 | 1.03486 | 1.01787 | 1.00033 | -0.01754 |
| Validation natural target CE | 0.93195 | 0.93345 | 0.91758 | 0.89855 | -0.01904 |
| New-route weighted objective contribution | 0.18541 | 0.17971 | 0.17105 | 0.16657 | -0.00448 |
| Top-two rate | 87.50000% | 81.25000% | 93.75000% | 93.75000% | +0.00000 pp |
| Net points per match | 4306.25000 | 10412.50000 | 6350.00000 | 10037.50000 | +3687.50000 |
| Hand win rate | 22.44544% | 25.62996% | 33.01091% | 26.83532% | -6.17560 pp |
| Dealin hand rate | 6.42113% | 3.67063% | 5.12897% | 8.77480% | +3.64583 pp |
| First-place rate | 56.25000% | 50.00000% | 56.25000% | 81.25000% | +25.00000 pp |
| Fourth-place rate | 12.50000% | 12.50000% | 0.00000% | 6.25000% | +6.25000 pp |
| Match reward (net /8000) | 0.53828 | 1.30156 | 0.79375 | 1.25469 | +0.46094 |

This consumed development grid has only two seed groups. It does not establish independent strength, rare-yaku mastery or superiority to frozen v8. No confidence interval or checkpoint-selection claim is made here. New-route objective is a fixed-weight validation loss contribution; match reward is net points divided by8000. Natural CE is retained to disclose tradeoffs.

This report reads completed artifacts only. It generates no games, changes no training recipe or serving policy, and does not open final confirmation or replace production weights. Raw matches and validation menus remain private.
