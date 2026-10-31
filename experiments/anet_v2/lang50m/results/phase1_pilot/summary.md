# Phase 1 pilot summary

1 seed, 6M tokens/run. BPC lower is better; not a final capability claim.

| arch | math ratio | mode | math BPC | code BPC | balanced | math acc | code acc | train s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| shared_depth | 1.00 | batch | 1.172 | 8.139 | 4.656 | 0.687 | 0.007 | 174 |
| shared_depth | 1.00 | boundary | 1.172 | 8.139 | 4.656 | 0.687 | 0.007 | 175 |
| shared_depth | 0.75 | batch | 1.108 | 2.668 | 1.888 | 0.704 | 0.518 | 202 |
| shared_depth | 0.75 | boundary | 1.101 | 2.676 | 1.888 | 0.709 | 0.518 | 175 |
| shared_depth | 0.50 | batch | 1.181 | 2.313 | 1.747 | 0.686 | 0.579 | 198 |
| shared_depth | 0.50 | boundary | 1.224 | 2.291 | 1.757 | 0.673 | 0.586 | 176 |
| shared_depth | 0.25 | batch | 1.262 | 2.142 | 1.702 | 0.665 | 0.609 | 203 |
| shared_depth | 0.25 | boundary | 1.298 | 2.100 | 1.699 | 0.658 | 0.615 | 174 |
| shared_depth | 0.00 | batch | 8.315 | 2.004 | 5.159 | 0.001 | 0.635 | 171 |
| shared_depth | 0.00 | boundary | 8.315 | 2.004 | 5.159 | 0.001 | 0.635 | 175 |
| transformer12 | 1.00 | batch | 1.044 | 17.579 | 9.311 | 0.719 | 0.013 | 191 |
| transformer12 | 1.00 | boundary | 1.044 | 17.579 | 9.311 | 0.719 | 0.013 | 189 |
| transformer12 | 0.75 | batch | 1.122 | 2.916 | 2.019 | 0.702 | 0.482 | 204 |
| transformer12 | 0.75 | boundary | 1.121 | 2.919 | 2.020 | 0.701 | 0.473 | 160 |
| transformer12 | 0.50 | batch | 1.215 | 2.481 | 1.848 | 0.674 | 0.556 | 175 |
| transformer12 | 0.50 | boundary | 1.210 | 2.469 | 1.840 | 0.676 | 0.552 | 188 |
| transformer12 | 0.25 | batch | 1.278 | 2.290 | 1.784 | 0.662 | 0.587 | 210 |
| transformer12 | 0.25 | boundary | 1.301 | 2.287 | 1.794 | 0.658 | 0.588 | 188 |
| transformer12 | 0.00 | batch | 7.693 | 2.168 | 4.931 | 0.081 | 0.608 | 185 |
| transformer12 | 0.00 | boundary | 7.693 | 2.168 | 4.931 | 0.081 | 0.608 | 177 |
