# A-Net v2 Stage 2 summary

Successful runs: 315; failed: 0.

## dense

| correlation | width | model | unique params | active params | test acc | grad cos | conflict |
|---|---:|---|---:|---:|---:|---:|---:|
| high | 128 | A-Net Basic | 51,217 | 50,827 | 94.91 ± 0.68 | n/a | n/a |
| high | 128 | A-Net Share | 34,705 | 34,315 | 95.13 ± 0.75 | +0.041 | 0.27 |
| high | 128 | A-Net Share+Adapter | 37,993 | 37,603 | 94.27 ± 1.16 | +0.113 | 0.13 |
| high | 512 | A-Net Basic | 794,641 | 793,099 | 95.26 ± 0.73 | n/a | n/a |
| high | 512 | A-Net Share | 531,985 | 530,443 | 94.73 ± 0.85 | +0.051 | 0.27 |
| high | 512 | A-Net Share+Adapter | 545,065 | 543,523 | 93.43 ± 0.58 | +0.047 | 0.47 |
| high | 2048 | A-Net Basic | 12,615,697 | 12,609,547 | 93.03 ± 2.56 | n/a | n/a |
| high | 2048 | A-Net Share | 8,419,345 | 8,413,195 | 94.17 ± 1.39 | +0.059 | 0.27 |
| high | 2048 | A-Net Share+Adapter | 8,471,593 | 8,465,443 | 93.16 ± 1.06 | +0.079 | 0.20 |
| low | 128 | A-Net Basic | 51,217 | 50,827 | 95.11 ± 1.12 | n/a | n/a |
| low | 128 | A-Net Share | 34,705 | 34,315 | 95.50 ± 1.14 | +0.028 | 0.40 |
| low | 128 | A-Net Share+Adapter | 37,993 | 37,603 | 95.05 ± 1.06 | +0.038 | 0.27 |
| low | 512 | A-Net Basic | 794,641 | 793,099 | 93.29 ± 2.31 | n/a | n/a |
| low | 512 | A-Net Share | 531,985 | 530,443 | 94.02 ± 1.72 | +0.021 | 0.33 |
| low | 512 | A-Net Share+Adapter | 545,065 | 543,523 | 94.71 ± 1.47 | +0.021 | 0.20 |
| low | 2048 | A-Net Basic | 12,615,697 | 12,609,547 | 93.49 ± 1.27 | n/a | n/a |
| low | 2048 | A-Net Share | 8,419,345 | 8,413,195 | 92.97 ± 1.35 | +0.028 | 0.27 |
| low | 2048 | A-Net Share+Adapter | 8,471,593 | 8,465,443 | 94.77 ± 1.54 | +0.003 | 0.40 |
| medium | 128 | A-Net Basic | 51,217 | 50,827 | 93.75 ± 1.76 | n/a | n/a |
| medium | 128 | A-Net Share | 34,705 | 34,315 | 94.24 ± 1.32 | +0.017 | 0.47 |
| medium | 128 | A-Net Share+Adapter | 37,993 | 37,603 | 94.25 ± 1.29 | +0.024 | 0.40 |
| medium | 512 | A-Net Basic | 794,641 | 793,099 | 94.77 ± 1.20 | n/a | n/a |
| medium | 512 | A-Net Share | 531,985 | 530,443 | 94.21 ± 0.49 | +0.034 | 0.40 |
| medium | 512 | A-Net Share+Adapter | 545,065 | 543,523 | 92.61 ± 2.99 | +0.053 | 0.40 |
| medium | 2048 | A-Net Basic | 12,615,697 | 12,609,547 | 93.94 ± 0.98 | n/a | n/a |
| medium | 2048 | A-Net Share | 8,419,345 | 8,413,195 | 91.61 ± 2.45 | +0.015 | 0.33 |
| medium | 2048 | A-Net Share+Adapter | 8,471,593 | 8,465,443 | 93.83 ± 1.44 | -0.013 | 0.47 |

## pair

| correlation | width | model | unique params | active params | test acc | grad cos | conflict |
|---|---:|---|---:|---:|---:|---:|---:|
| high | 128 | Independent | 54,024 | 17,922 | 95.60 ± 0.64 | n/a | n/a |
| high | 128 | A-Net Basic | 51,217 | 34,315 | 95.06 ± 0.87 | n/a | n/a |
| high | 128 | A-Net Share | 34,705 | 26,059 | 95.01 ± 0.82 | +0.074 | 0.27 |
| high | 128 | A-Net Share+Adapter | 37,993 | 28,251 | 94.35 ± 1.41 | +0.209 | 0.20 |
| high | 512 | Independent | 805,896 | 268,290 | 96.01 ± 0.39 | n/a | n/a |
| high | 512 | A-Net Basic | 794,641 | 530,443 | 94.71 ± 1.89 | n/a | n/a |
| high | 512 | A-Net Share | 531,985 | 399,115 | 94.33 ± 1.28 | +0.043 | 0.33 |
| high | 512 | A-Net Share+Adapter | 545,065 | 407,835 | 93.56 ± 1.94 | +0.074 | 0.33 |
| high | 2048 | Independent | 12,660,744 | 4,218,882 | 95.71 ± 0.51 | n/a | n/a |
| high | 2048 | A-Net Basic | 12,615,697 | 8,413,195 | 95.05 ± 0.67 | n/a | n/a |
| high | 2048 | A-Net Share | 8,419,345 | 6,315,019 | 94.07 ± 1.27 | +0.011 | 0.47 |
| high | 2048 | A-Net Share+Adapter | 8,471,593 | 6,349,851 | 92.57 ± 1.76 | +0.059 | 0.53 |
| low | 128 | Independent | 54,024 | 17,922 | 96.33 ± 0.93 | n/a | n/a |
| low | 128 | A-Net Basic | 51,217 | 34,315 | 95.84 ± 0.85 | n/a | n/a |
| low | 128 | A-Net Share | 34,705 | 26,059 | 95.38 ± 1.32 | +0.028 | 0.40 |
| low | 128 | A-Net Share+Adapter | 37,993 | 28,251 | 95.50 ± 1.27 | +0.022 | 0.33 |
| low | 512 | Independent | 805,896 | 268,290 | 96.42 ± 0.56 | n/a | n/a |
| low | 512 | A-Net Basic | 794,641 | 530,443 | 95.35 ± 1.15 | n/a | n/a |
| low | 512 | A-Net Share | 531,985 | 399,115 | 94.68 ± 2.39 | -0.016 | 0.53 |
| low | 512 | A-Net Share+Adapter | 545,065 | 407,835 | 94.57 ± 0.83 | +0.014 | 0.33 |
| low | 2048 | Independent | 12,660,744 | 4,218,882 | 95.49 ± 0.53 | n/a | n/a |
| low | 2048 | A-Net Basic | 12,615,697 | 8,413,195 | 93.64 ± 1.73 | n/a | n/a |
| low | 2048 | A-Net Share | 8,419,345 | 6,315,019 | 94.53 ± 1.33 | +0.022 | 0.33 |
| low | 2048 | A-Net Share+Adapter | 8,471,593 | 6,349,851 | 93.91 ± 1.86 | -0.009 | 0.60 |
| medium | 128 | Independent | 54,024 | 17,922 | 95.49 ± 1.02 | n/a | n/a |
| medium | 128 | A-Net Basic | 51,217 | 34,315 | 94.45 ± 1.35 | n/a | n/a |
| medium | 128 | A-Net Share | 34,705 | 26,059 | 94.11 ± 1.67 | +0.031 | 0.40 |
| medium | 128 | A-Net Share+Adapter | 37,993 | 28,251 | 93.50 ± 2.67 | +0.017 | 0.47 |
| medium | 512 | Independent | 805,896 | 268,290 | 96.05 ± 0.42 | n/a | n/a |
| medium | 512 | A-Net Basic | 794,641 | 530,443 | 95.19 ± 1.02 | n/a | n/a |
| medium | 512 | A-Net Share | 531,985 | 399,115 | 94.97 ± 0.91 | +0.028 | 0.53 |
| medium | 512 | A-Net Share+Adapter | 545,065 | 407,835 | 94.93 ± 0.72 | +0.043 | 0.27 |
| medium | 2048 | Independent | 12,660,744 | 4,218,882 | 95.70 ± 0.33 | n/a | n/a |
| medium | 2048 | A-Net Basic | 12,615,697 | 8,413,195 | 94.95 ± 0.47 | n/a | n/a |
| medium | 2048 | A-Net Share | 8,419,345 | 6,315,019 | 94.97 ± 1.10 | +0.007 | 0.60 |
| medium | 2048 | A-Net Share+Adapter | 8,471,593 | 6,349,851 | 95.29 ± 0.59 | +0.034 | 0.27 |

