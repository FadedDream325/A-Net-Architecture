# A-Net v2 Stage 2 summary

Successful runs: 315; failed: 0.

## dense

| correlation | width | model | unique params | active params | test acc | grad cos | conflict |
|---|---:|---|---:|---:|---:|---:|---:|
| high | 128 | A-Net Basic | 51,217 | 50,827 | 95.49 ± 0.72 | n/a | n/a |
| high | 128 | A-Net Share | 34,705 | 34,315 | 95.41 ± 0.74 | +0.000 | 0.00 |
| high | 128 | A-Net Share+Adapter | 37,993 | 37,603 | 95.35 ± 0.60 | +0.000 | 0.00 |
| high | 512 | A-Net Basic | 794,641 | 793,099 | 95.96 ± 0.41 | n/a | n/a |
| high | 512 | A-Net Share | 531,985 | 530,443 | 96.10 ± 0.43 | +0.000 | 0.00 |
| high | 512 | A-Net Share+Adapter | 545,065 | 543,523 | 96.14 ± 0.42 | +0.000 | 0.00 |
| high | 2048 | A-Net Basic | 12,615,697 | 12,609,547 | 95.89 ± 0.68 | n/a | n/a |
| high | 2048 | A-Net Share | 8,419,345 | 8,413,195 | 95.93 ± 0.55 | +0.000 | 0.00 |
| high | 2048 | A-Net Share+Adapter | 8,471,593 | 8,465,443 | 95.89 ± 0.60 | +0.000 | 0.00 |
| low | 128 | A-Net Basic | 51,217 | 50,827 | 95.72 ± 0.90 | n/a | n/a |
| low | 128 | A-Net Share | 34,705 | 34,315 | 95.68 ± 0.80 | +0.000 | 0.00 |
| low | 128 | A-Net Share+Adapter | 37,993 | 37,603 | 95.57 ± 0.85 | +0.000 | 0.00 |
| low | 512 | A-Net Basic | 794,641 | 793,099 | 96.19 ± 0.72 | n/a | n/a |
| low | 512 | A-Net Share | 531,985 | 530,443 | 96.22 ± 0.74 | +0.000 | 0.00 |
| low | 512 | A-Net Share+Adapter | 545,065 | 543,523 | 96.17 ± 0.72 | +0.000 | 0.00 |
| low | 2048 | A-Net Basic | 12,615,697 | 12,609,547 | 96.17 ± 0.79 | n/a | n/a |
| low | 2048 | A-Net Share | 8,419,345 | 8,413,195 | 96.09 ± 0.61 | +0.000 | 0.00 |
| low | 2048 | A-Net Share+Adapter | 8,471,593 | 8,465,443 | 96.07 ± 0.64 | +0.000 | 0.00 |
| medium | 128 | A-Net Basic | 51,217 | 50,827 | 95.02 ± 1.41 | n/a | n/a |
| medium | 128 | A-Net Share | 34,705 | 34,315 | 94.77 ± 1.18 | +0.000 | 0.00 |
| medium | 128 | A-Net Share+Adapter | 37,993 | 37,603 | 95.01 ± 1.23 | +0.000 | 0.00 |
| medium | 512 | A-Net Basic | 794,641 | 793,099 | 95.74 ± 0.64 | n/a | n/a |
| medium | 512 | A-Net Share | 531,985 | 530,443 | 95.81 ± 0.62 | +0.000 | 0.00 |
| medium | 512 | A-Net Share+Adapter | 545,065 | 543,523 | 95.83 ± 0.49 | +0.000 | 0.00 |
| medium | 2048 | A-Net Basic | 12,615,697 | 12,609,547 | 95.77 ± 0.58 | n/a | n/a |
| medium | 2048 | A-Net Share | 8,419,345 | 8,413,195 | 95.69 ± 0.61 | +0.000 | 0.00 |
| medium | 2048 | A-Net Share+Adapter | 8,471,593 | 8,465,443 | 95.79 ± 0.61 | +0.000 | 0.00 |

## pair

| correlation | width | model | unique params | active params | test acc | grad cos | conflict |
|---|---:|---|---:|---:|---:|---:|---:|
| high | 128 | Independent | 54,024 | 17,922 | 95.60 ± 0.64 | n/a | n/a |
| high | 128 | A-Net Basic | 51,217 | 34,315 | 95.53 ± 0.72 | n/a | n/a |
| high | 128 | A-Net Share | 34,705 | 26,059 | 95.67 ± 0.76 | +0.000 | 0.00 |
| high | 128 | A-Net Share+Adapter | 37,993 | 28,251 | 95.45 ± 0.53 | +0.000 | 0.00 |
| high | 512 | Independent | 805,896 | 268,290 | 96.01 ± 0.39 | n/a | n/a |
| high | 512 | A-Net Basic | 794,641 | 530,443 | 96.09 ± 0.39 | n/a | n/a |
| high | 512 | A-Net Share | 531,985 | 399,115 | 96.11 ± 0.40 | +0.000 | 0.00 |
| high | 512 | A-Net Share+Adapter | 545,065 | 407,835 | 96.13 ± 0.46 | +0.000 | 0.00 |
| high | 2048 | Independent | 12,660,744 | 4,218,882 | 95.71 ± 0.51 | n/a | n/a |
| high | 2048 | A-Net Basic | 12,615,697 | 8,413,195 | 95.55 ± 0.63 | n/a | n/a |
| high | 2048 | A-Net Share | 8,419,345 | 6,315,019 | 95.79 ± 0.55 | +0.000 | 0.00 |
| high | 2048 | A-Net Share+Adapter | 8,471,593 | 6,349,851 | 95.55 ± 0.53 | +0.000 | 0.00 |
| low | 128 | Independent | 54,024 | 17,922 | 96.33 ± 0.93 | n/a | n/a |
| low | 128 | A-Net Basic | 51,217 | 34,315 | 95.95 ± 0.97 | n/a | n/a |
| low | 128 | A-Net Share | 34,705 | 26,059 | 95.92 ± 0.83 | +0.000 | 0.00 |
| low | 128 | A-Net Share+Adapter | 37,993 | 28,251 | 95.81 ± 0.88 | +0.000 | 0.00 |
| low | 512 | Independent | 805,896 | 268,290 | 96.42 ± 0.56 | n/a | n/a |
| low | 512 | A-Net Basic | 794,641 | 530,443 | 96.16 ± 0.63 | n/a | n/a |
| low | 512 | A-Net Share | 531,985 | 399,115 | 96.08 ± 0.64 | +0.000 | 0.00 |
| low | 512 | A-Net Share+Adapter | 545,065 | 407,835 | 96.02 ± 0.70 | +0.000 | 0.00 |
| low | 2048 | Independent | 12,660,744 | 4,218,882 | 95.49 ± 0.53 | n/a | n/a |
| low | 2048 | A-Net Basic | 12,615,697 | 8,413,195 | 95.98 ± 0.64 | n/a | n/a |
| low | 2048 | A-Net Share | 8,419,345 | 6,315,019 | 95.91 ± 0.58 | +0.000 | 0.00 |
| low | 2048 | A-Net Share+Adapter | 8,471,593 | 6,349,851 | 95.95 ± 0.61 | +0.000 | 0.00 |
| medium | 128 | Independent | 54,024 | 17,922 | 95.49 ± 1.02 | n/a | n/a |
| medium | 128 | A-Net Basic | 51,217 | 34,315 | 95.16 ± 1.30 | n/a | n/a |
| medium | 128 | A-Net Share | 34,705 | 26,059 | 95.00 ± 1.22 | +0.000 | 0.00 |
| medium | 128 | A-Net Share+Adapter | 37,993 | 28,251 | 95.04 ± 1.25 | +0.000 | 0.00 |
| medium | 512 | Independent | 805,896 | 268,290 | 96.05 ± 0.42 | n/a | n/a |
| medium | 512 | A-Net Basic | 794,641 | 530,443 | 95.85 ± 0.52 | n/a | n/a |
| medium | 512 | A-Net Share | 531,985 | 399,115 | 95.87 ± 0.53 | +0.000 | 0.00 |
| medium | 512 | A-Net Share+Adapter | 545,065 | 407,835 | 96.01 ± 0.44 | +0.000 | 0.00 |
| medium | 2048 | Independent | 12,660,744 | 4,218,882 | 95.70 ± 0.33 | n/a | n/a |
| medium | 2048 | A-Net Basic | 12,615,697 | 8,413,195 | 95.67 ± 0.64 | n/a | n/a |
| medium | 2048 | A-Net Share | 8,419,345 | 6,315,019 | 95.83 ± 0.67 | +0.000 | 0.00 |
| medium | 2048 | A-Net Share+Adapter | 8,471,593 | 6,349,851 | 95.65 ± 0.63 | +0.000 | 0.00 |

