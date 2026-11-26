# California Housing: capped20k SGDM confirmation

Complete five-seed confirmation: **True**; audited runs **5/5**. All64 development prefixes were reused without further training.

Frozen winner: **sgdm_11**, `{"dampening": 0, "dropout": 0.3690073279296028, "lr": 0.01, "momentum": 0.9, "nesterov": false, "optimizer": "sgdm", "weight_decay": 0.001}`.

All comparisons below use20,000 updates. New SGDM uses fresh seeds3340–3344; original controls use seeds2340–2344. Tuning and possibly dropout differ. These independent-seed comparisons do not isolate an optimizer-only causal effect.

Lower RMSE is better. RMSE units are$100,000; it is not a percentage error. Alpha uses the ordinary fitted value with the prescribed confidence/tail weight. One eligible matrix means no layer-dispersion estimate.

| Cohort | Test RMSE mean±95%CI | Test RMSE sampleSD | Alpha mean±95%CI | Alpha sampleSD |
|---|---:|---:|---:|---:|
| original_adamw | 0.480780±0.002797 (n=5) | 0.002253 | 1.678083±0.023836 (n=5) | 0.019197 |
| original_modified_adamw | 0.483766±0.002055 (n=5) | 0.001655 | 1.612786±0.003481 (n=5) | 0.002803 |
| original_sgdm | 0.509690±0.005787 (n=5) | 0.004661 | 2.277816±0.009514 (n=5) | 0.007663 |
| sgdm_grid_selected | 0.492049±0.001202 (n=5) | 0.000968 | 1.690720±0.005065 (n=5) | 0.004079 |
