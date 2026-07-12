# Concurrency sweep C1–C12 · ablit-mida · 1M

- API: `http://10.100.10.1:8000`
- Weights: balance L10–42 λ3.5 no-mtp
- Serve: max_model_len=1048576, max_num_seqs=12, UTIL=0.85, DSpark k=5, nvfp4_ds_mla
- Method: static batch, ignore_eos, max_tokens=256, code CRUD prompt, best of 2
- Timestamp: 2026-07-12T14:21:28Z

| C | server agg tok/s | per-stream (agg/C) | DSpark accept | client mean pure |
|---:|---:|---:|---:|---:|
| 1 | 50.2 | 50.2 | 0.629 | 12.7 |
| 2 | 77.9 | 39.0 | 0.6192 | 10.1 |
| 3 | 92.3 | 30.8 | 0.5749 | 8.6 |
| 4 | 113.4 | 28.4 | 0.5931 | 7.6 |
| 5 | 105.0 | 21.0 | 0.5781 | 6.0 |
| 6 | 149.5 | 24.9 | 0.601 | 6.8 |
| 7 | 160.4 | 22.9 | 0.5643 | 6.7 |
| 8 | 167.7 | 21.0 | 0.5635 | 6.0 |
| 9 | 158.9 | 17.7 | 0.563 | 5.2 |
| 10 | 197.8 | 19.8 | 0.5654 | 5.6 |
| 11 | 214.9 | 19.5 | 0.5921 | 5.5 |
| 12 | 216.1 | 18.0 | 0.5842 | 5.2 |
