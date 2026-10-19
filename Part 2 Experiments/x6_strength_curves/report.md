# X6: erasure vs edit size

Relative edit size x = δ × median‖c_t‖/‖e_t‖ (per concept and method; `judge_rerun/Cinder/edit_size.csv`). QA_train accuracy interpolated at equal x (lower = more erased; pristine ≈ the δ→0 end). Figures: `english_<concept>.png`, `leak_<concept>.png` (when the G1 jobs are done).

| concept | method | picked_delta | qa@0.5 | qa@1 | qa@2 | qa@5 | qa@10 | qa@20 | qa@50 |
|---|---|---|---|---|---|---|---|---|---|
| Cannabis | EMBER | 200.0 | 0.752 | 0.721 | 0.682 | 0.7 | 0.7 | 0.698 | 0.672 |
| Cannabis | CINDER | 5.0 | 0.777 | 0.738 | 0.732 | 0.748 | 0.757 | 0.76 | 0.76 |
| Gun | EMBER | 5.0 | 0.755 | 0.683 | 0.604 | 0.64 | 0.64 | 0.64 | 0.64 |
| Gun | CINDER | 100.0 | 0.738 | 0.662 | 0.64 | 0.64 | 0.64 | 0.626 |  |
| Nazism | EMBER | 100.0 | 0.662 | 0.586 | 0.526 | 0.536 | 0.519 | 0.502 | 0.48 |
| Republic of Ireland | EMBER | 100.0 | 0.646 | 0.596 | 0.581 | 0.562 | 0.571 | 0.579 | 0.545 |
| Culture of Greece | EMBER | 200.0 | 0.669 | 0.537 | 0.48 | 0.433 | 0.385 | 0.359 | 0.34 |
| Culture of Greece | CINDER | 100.0 | 0.5 | 0.515 | 0.492 | 0.463 | 0.445 | 0.44 |  |
| Golf | EMBER | 10.0 | 0.726 | 0.655 | 0.612 | 0.613 | 0.63 | 0.648 | 0.66 |
| Golf | CINDER | 10.0 | 0.7 | 0.7 | 0.68 | 0.68 | 0.68 | 0.68 |  |

