# Scaling summary

| scale | model | params | active | MACs | acc mean ± std | acc T1 | acc T2 | acc T3 | train s | peak GB | b1 µs | b1024 samples/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | A-Net + block (disabled) | 116,742 | 33,922 | 33,536 | 98.03 ± 0.11 | 99.31 | 98.44 | 96.34 | 4.03 | 0.067 | 86.7 | 10253263 |
| medium | A-Net + block (disabled) | 1,412,102 | 405,442 | 404,096 | 98.02 ± 0.30 | 99.27 | 98.41 | 96.38 | 3.94 | 0.087 | 104.6 | 4274245 |
| large | A-Net + block (disabled) | 5,634,054 | 1,613,698 | 1,611,008 | 97.86 ± 0.37 | 99.03 | 98.56 | 96.00 | 4.92 | 0.139 | 97.2 | 2005786 |

Failed runs: 0