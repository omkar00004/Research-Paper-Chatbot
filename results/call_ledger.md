# Call ledger


## run = full

| stage | provider | model | HTTP calls | cache hits | hit rate | retries | 429s | 5xx | truncations (length retries) | failed | invalid JSON | prompt tok | completion tok | mean latency s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | freellmapi | nemotron-3-super-120b | 206 | 17 | 8% | 159 | 0 | 159 | 8 | 28 | 0 | 93996 | 142137 | 4.51 |
| S3 | freellmapi | gpt-oss-20b | 258 | 25 | 9% | 0 | 0 | 0 | 25 | 18 | 0 | 516939 | 24043 | 3.38 |
| S3 | freellmapi | nemotron-3-super-120b | 240 | 8 | 3% | 124 | 0 | 125 | 9 | 18 | 0 | 470706 | 111179 | 3.47 |
| S4 | freellmapi | gpt-oss-20b | 33 | 31 | 48% | 0 | 0 | 0 | 1 | 0 | 0 | 52672 | 2811 | 3.68 |
| S4 | freellmapi | nemotron-3-super-120b | 288 | 292 | 50% | 50 | 28 | 22 | 1 | 7 | 0 | 313628 | 115410 | 3.95 |
| S5 | freellmapi | nemotron-3-super-120b | 30 | 3 | 9% | 12 | 0 | 12 | 0 | 2 | 0 | 26557 | 13654 | 4.73 |
| preflight | freellmapi | gpt-oss-20b | 0 | 1 | 100% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| preflight | freellmapi | nemotron-3-super-120b | 0 | 2 | 100% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| **total** | | | 1055 | 379 | | 345 | 28 | 318 | 44 | 73 | 0 | 1474498 | 409234 | |

## run = dry

| stage | provider | model | HTTP calls | cache hits | hit rate | retries | 429s | 5xx | truncations (length retries) | failed | invalid JSON | prompt tok | completion tok | mean latency s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | freellmapi | nemotron-3-super-120b | 14 | 0 | 0% | 10 | 0 | 10 | 0 | 1 | 0 | 6371 | 9122 | 5.07 |
| S1 | freellmapi | nvidia/nemotron-3-super-120b-a12b | 0 | 2 | 100% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| S3 | freellmapi | gpt-oss-20b | 16 | 0 | 0% | 0 | 0 | 0 | 1 | 0 | 0 | 28029 | 1179 | 2.42 |
| S3 | freellmapi | nemotron-3-super-120b | 15 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | 0 | 25819 | 6148 | 2.16 |
| S4 | freellmapi | gpt-oss-20b | 10 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | 0 | 16037 | 690 | 1.93 |
| S4 | freellmapi | nemotron-3-super-120b | 89 | 0 | 0% | 2 | 0 | 2 | 0 | 0 | 0 | 95599 | 32284 | 2.50 |
| S4 | freellmapi | nvidia/nemotron-3-super-120b-a12b | 0 | 31 | 100% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - |
| S5 | freellmapi | nemotron-3-super-120b | 3 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | 0 | 3040 | 1135 | 2.81 |
| **total** | | | 147 | 33 | | 12 | 0 | 12 | 1 | 1 | 0 | 174895 | 50558 | |

Cost: all models are served by freellmapi free tiers; dollar cost = $0.00 (prices configurable via PRICE_IN_PER_M / PRICE_OUT_PER_M).

