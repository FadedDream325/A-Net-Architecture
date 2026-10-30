# A-Net v2 Stage 3 summary

Successful runs: 288; failed: 0.

## batch=1

| width | model | rate | cache | hit rate | exec ratio | P50 us | P50 ratio | throughput |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 128 | A-Net Reuse (no share) | 0.00 | False | 0.00 | 1.00 | 538.8 | n/a | 1754 |
| 128 | A-Net Reuse (no share) | 0.00 | True | 0.00 | 1.00 | 804.5 | 1.49 | 1157 |
| 128 | A-Net Reuse (no share) | 0.25 | False | 0.00 | 1.00 | 475.8 | n/a | 2013 |
| 128 | A-Net Reuse (no share) | 0.25 | True | 0.25 | 0.75 | 789.8 | 1.66 | 1303 |
| 128 | A-Net Reuse (no share) | 0.50 | False | 0.00 | 1.00 | 478.2 | n/a | 1991 |
| 128 | A-Net Reuse (no share) | 0.50 | True | 0.50 | 0.50 | 753.9 | 1.58 | 1428 |
| 128 | A-Net Reuse (no share) | 0.75 | False | 0.00 | 1.00 | 477.1 | n/a | 1936 |
| 128 | A-Net Reuse (no share) | 0.75 | True | 0.75 | 0.25 | 626.9 | 1.31 | 1469 |
| 128 | A-Net Full (share+cache) | 0.00 | False | 0.00 | 1.00 | 451.5 | n/a | 1995 |
| 128 | A-Net Full (share+cache) | 0.00 | True | 0.20 | 0.80 | 853.8 | 1.89 | 1124 |
| 128 | A-Net Full (share+cache) | 0.25 | False | 0.00 | 1.00 | 398.0 | n/a | 2256 |
| 128 | A-Net Full (share+cache) | 0.25 | True | 0.40 | 0.60 | 674.9 | 1.70 | 1359 |
| 128 | A-Net Full (share+cache) | 0.50 | False | 0.00 | 1.00 | 471.6 | n/a | 1889 |
| 128 | A-Net Full (share+cache) | 0.50 | True | 0.60 | 0.40 | 689.5 | 1.46 | 1297 |
| 128 | A-Net Full (share+cache) | 0.75 | False | 0.00 | 1.00 | 418.5 | n/a | 2037 |
| 128 | A-Net Full (share+cache) | 0.75 | True | 0.80 | 0.20 | 528.4 | 1.26 | 1589 |
| 512 | A-Net Reuse (no share) | 0.00 | False | 0.00 | 1.00 | 469.0 | n/a | 1868 |
| 512 | A-Net Reuse (no share) | 0.00 | True | 0.00 | 1.00 | 818.2 | 1.74 | 1123 |
| 512 | A-Net Reuse (no share) | 0.25 | False | 0.00 | 1.00 | 473.3 | n/a | 1998 |
| 512 | A-Net Reuse (no share) | 0.25 | True | 0.25 | 0.75 | 809.2 | 1.71 | 1257 |
| 512 | A-Net Reuse (no share) | 0.50 | False | 0.00 | 1.00 | 491.7 | n/a | 1909 |
| 512 | A-Net Reuse (no share) | 0.50 | True | 0.50 | 0.50 | 810.9 | 1.65 | 1380 |
| 512 | A-Net Reuse (no share) | 0.75 | False | 0.00 | 1.00 | 490.2 | n/a | 1906 |
| 512 | A-Net Reuse (no share) | 0.75 | True | 0.75 | 0.25 | 548.9 | 1.12 | 1542 |
| 512 | A-Net Full (share+cache) | 0.00 | False | 0.00 | 1.00 | 485.3 | n/a | 1959 |
| 512 | A-Net Full (share+cache) | 0.00 | True | 0.20 | 0.80 | 784.4 | 1.62 | 1192 |
| 512 | A-Net Full (share+cache) | 0.25 | False | 0.00 | 1.00 | 491.0 | n/a | 1941 |
| 512 | A-Net Full (share+cache) | 0.25 | True | 0.40 | 0.60 | 775.0 | 1.58 | 1316 |
| 512 | A-Net Full (share+cache) | 0.50 | False | 0.00 | 1.00 | 492.1 | n/a | 1844 |
| 512 | A-Net Full (share+cache) | 0.50 | True | 0.60 | 0.40 | 759.5 | 1.54 | 1431 |
| 512 | A-Net Full (share+cache) | 0.75 | False | 0.00 | 1.00 | 492.8 | n/a | 1900 |
| 512 | A-Net Full (share+cache) | 0.75 | True | 0.80 | 0.20 | 550.7 | 1.12 | 1567 |
| 2048 | A-Net Reuse (no share) | 0.00 | False | 0.00 | 1.00 | 453.8 | n/a | 2084 |
| 2048 | A-Net Reuse (no share) | 0.00 | True | 0.00 | 1.00 | 817.5 | 1.80 | 1156 |
| 2048 | A-Net Reuse (no share) | 0.25 | False | 0.00 | 1.00 | 468.1 | n/a | 1991 |
| 2048 | A-Net Reuse (no share) | 0.25 | True | 0.25 | 0.75 | 807.4 | 1.73 | 1264 |
| 2048 | A-Net Reuse (no share) | 0.50 | False | 0.00 | 1.00 | 462.1 | n/a | 2047 |
| 2048 | A-Net Reuse (no share) | 0.50 | True | 0.50 | 0.50 | 786.6 | 1.70 | 1391 |
| 2048 | A-Net Reuse (no share) | 0.75 | False | 0.00 | 1.00 | 465.3 | n/a | 2028 |
| 2048 | A-Net Reuse (no share) | 0.75 | True | 0.75 | 0.25 | 559.7 | 1.20 | 1541 |
| 2048 | A-Net Full (share+cache) | 0.00 | False | 0.00 | 1.00 | 461.8 | n/a | 2059 |
| 2048 | A-Net Full (share+cache) | 0.00 | True | 0.20 | 0.80 | 765.1 | 1.66 | 1247 |
| 2048 | A-Net Full (share+cache) | 0.25 | False | 0.00 | 1.00 | 450.1 | n/a | 2080 |
| 2048 | A-Net Full (share+cache) | 0.25 | True | 0.40 | 0.60 | 740.2 | 1.64 | 1322 |
| 2048 | A-Net Full (share+cache) | 0.50 | False | 0.00 | 1.00 | 450.5 | n/a | 2104 |
| 2048 | A-Net Full (share+cache) | 0.50 | True | 0.60 | 0.40 | 721.5 | 1.60 | 1471 |
| 2048 | A-Net Full (share+cache) | 0.75 | False | 0.00 | 1.00 | 445.4 | n/a | 2127 |
| 2048 | A-Net Full (share+cache) | 0.75 | True | 0.80 | 0.20 | 543.5 | 1.22 | 1606 |

## batch=256

| width | model | rate | cache | hit rate | exec ratio | P50 us | P50 ratio | throughput |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 128 | A-Net Reuse (no share) | 0.00 | False | 0.00 | 1.00 | 508.9 | n/a | 474655 |
| 128 | A-Net Reuse (no share) | 0.00 | True | 0.00 | 1.00 | 889.5 | 1.75 | 261358 |
| 128 | A-Net Reuse (no share) | 0.25 | False | 0.00 | 1.00 | 486.8 | n/a | 497908 |
| 128 | A-Net Reuse (no share) | 0.25 | True | 0.25 | 0.75 | 834.6 | 1.71 | 315048 |
| 128 | A-Net Reuse (no share) | 0.50 | False | 0.00 | 1.00 | 488.4 | n/a | 483180 |
| 128 | A-Net Reuse (no share) | 0.50 | True | 0.50 | 0.50 | 800.0 | 1.64 | 342305 |
| 128 | A-Net Reuse (no share) | 0.75 | False | 0.00 | 1.00 | 503.5 | n/a | 475978 |
| 128 | A-Net Reuse (no share) | 0.75 | True | 0.75 | 0.25 | 577.2 | 1.15 | 377627 |
| 128 | A-Net Full (share+cache) | 0.00 | False | 0.00 | 1.00 | 485.5 | n/a | 492028 |
| 128 | A-Net Full (share+cache) | 0.00 | True | 0.20 | 0.80 | 799.7 | 1.65 | 290990 |
| 128 | A-Net Full (share+cache) | 0.25 | False | 0.00 | 1.00 | 488.4 | n/a | 481905 |
| 128 | A-Net Full (share+cache) | 0.25 | True | 0.40 | 0.60 | 786.5 | 1.61 | 330482 |
| 128 | A-Net Full (share+cache) | 0.50 | False | 0.00 | 1.00 | 487.6 | n/a | 471782 |
| 128 | A-Net Full (share+cache) | 0.50 | True | 0.60 | 0.40 | 755.8 | 1.55 | 359974 |
| 128 | A-Net Full (share+cache) | 0.75 | False | 0.00 | 1.00 | 489.7 | n/a | 497152 |
| 128 | A-Net Full (share+cache) | 0.75 | True | 0.80 | 0.20 | 574.7 | 1.17 | 389897 |
| 512 | A-Net Reuse (no share) | 0.00 | False | 0.00 | 1.00 | 512.9 | n/a | 476674 |
| 512 | A-Net Reuse (no share) | 0.00 | True | 0.00 | 1.00 | 1061.9 | 2.07 | 236507 |
| 512 | A-Net Reuse (no share) | 0.25 | False | 0.00 | 1.00 | 514.7 | n/a | 473261 |
| 512 | A-Net Reuse (no share) | 0.25 | True | 0.25 | 0.75 | 856.4 | 1.66 | 309246 |
| 512 | A-Net Reuse (no share) | 0.50 | False | 0.00 | 1.00 | 510.6 | n/a | 476919 |
| 512 | A-Net Reuse (no share) | 0.50 | True | 0.50 | 0.50 | 842.4 | 1.65 | 332493 |
| 512 | A-Net Reuse (no share) | 0.75 | False | 0.00 | 1.00 | 513.4 | n/a | 468436 |
| 512 | A-Net Reuse (no share) | 0.75 | True | 0.75 | 0.25 | 583.0 | 1.14 | 376488 |
| 512 | A-Net Full (share+cache) | 0.00 | False | 0.00 | 1.00 | 514.6 | n/a | 477272 |
| 512 | A-Net Full (share+cache) | 0.00 | True | 0.20 | 0.80 | 979.9 | 1.90 | 259341 |
| 512 | A-Net Full (share+cache) | 0.25 | False | 0.00 | 1.00 | 516.7 | n/a | 465682 |
| 512 | A-Net Full (share+cache) | 0.25 | True | 0.40 | 0.60 | 802.6 | 1.55 | 326180 |
| 512 | A-Net Full (share+cache) | 0.50 | False | 0.00 | 1.00 | 513.7 | n/a | 470095 |
| 512 | A-Net Full (share+cache) | 0.50 | True | 0.60 | 0.40 | 778.7 | 1.52 | 352918 |
| 512 | A-Net Full (share+cache) | 0.75 | False | 0.00 | 1.00 | 512.9 | n/a | 470455 |
| 512 | A-Net Full (share+cache) | 0.75 | True | 0.80 | 0.20 | 584.1 | 1.14 | 384269 |
| 2048 | A-Net Reuse (no share) | 0.00 | False | 0.00 | 1.00 | 872.9 | n/a | 284890 |
| 2048 | A-Net Reuse (no share) | 0.00 | True | 0.00 | 1.00 | 1034.1 | 1.18 | 227215 |
| 2048 | A-Net Reuse (no share) | 0.25 | False | 0.00 | 1.00 | 875.7 | n/a | 283604 |
| 2048 | A-Net Reuse (no share) | 0.25 | True | 0.25 | 0.75 | 1022.6 | 1.17 | 265314 |
| 2048 | A-Net Reuse (no share) | 0.50 | False | 0.00 | 1.00 | 876.7 | n/a | 282145 |
| 2048 | A-Net Reuse (no share) | 0.50 | True | 0.50 | 0.50 | 972.0 | 1.11 | 304928 |
| 2048 | A-Net Reuse (no share) | 0.75 | False | 0.00 | 1.00 | 872.7 | n/a | 284683 |
| 2048 | A-Net Reuse (no share) | 0.75 | True | 0.75 | 0.25 | 585.6 | 0.67 | 354626 |
| 2048 | A-Net Full (share+cache) | 0.00 | False | 0.00 | 1.00 | 861.7 | n/a | 286241 |
| 2048 | A-Net Full (share+cache) | 0.00 | True | 0.20 | 0.80 | 888.5 | 1.03 | 256938 |
| 2048 | A-Net Full (share+cache) | 0.25 | False | 0.00 | 1.00 | 872.8 | n/a | 284058 |
| 2048 | A-Net Full (share+cache) | 0.25 | True | 0.40 | 0.60 | 879.5 | 1.01 | 303109 |
| 2048 | A-Net Full (share+cache) | 0.50 | False | 0.00 | 1.00 | 871.4 | n/a | 285408 |
| 2048 | A-Net Full (share+cache) | 0.50 | True | 0.60 | 0.40 | 840.6 | 0.96 | 335472 |
| 2048 | A-Net Full (share+cache) | 0.75 | False | 0.00 | 1.00 | 870.8 | n/a | 286296 |
| 2048 | A-Net Full (share+cache) | 0.75 | True | 0.80 | 0.20 | 657.6 | 0.76 | 346236 |

