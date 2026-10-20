# X7: QA_train accuracy on questions that contain an edited token (touched) vs not (untouched)

Columns: δ. Lower = more erased. A drop only in `touched` means the edit removes the words, not the knowledge.

| concept | method | group | n | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 | 50.0 | 100.0 | 200.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cannabis | CINDER | touched | 49 | 0.837 | 0.735 | 0.776 | 0.714 | 0.735 | 0.755 | 0.755 | 0.755 |
| Cannabis | CINDER | untouched | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Cannabis | EMBER | touched | 49 | 0.837 | 0.755 | 0.735 | 0.673 | 0.694 | 0.714 | 0.694 | 0.673 |
| Cannabis | EMBER | untouched | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| Culture of Greece | CINDER | touched | 50 | 0.72 | 0.58 | 0.5 | 0.52 | 0.48 | 0.44 | 0.44 | 0.44 |
| Culture of Greece | EMBER | touched | 50 | 0.74 | 0.62 | 0.48 | 0.48 | 0.4 | 0.34 | 0.34 | 0.32 |
| Golf | CINDER | touched | 44 | 0.818 | 0.75 | 0.705 | 0.705 | 0.682 | 0.682 | 0.682 | 0.682 |
| Golf | CINDER | untouched | 6 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| Golf | EMBER | touched | 42 | 0.833 | 0.81 | 0.69 | 0.595 | 0.571 | 0.619 | 0.643 | 0.643 |
| Golf | EMBER | untouched | 8 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| Gun | CINDER | touched | 41 | 0.902 | 0.78 | 0.732 | 0.61 | 0.61 | 0.61 | 0.585 | 0.61 |
| Gun | CINDER | untouched | 9 | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 |
| Gun | EMBER | touched | 37 | 0.865 | 0.757 | 0.676 | 0.514 | 0.568 | 0.568 | 0.568 | 0.568 |
| Gun | EMBER | untouched | 13 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 | 0.846 |
| Nazism | EMBER | touched | 40 | 0.8 | 0.7 | 0.6 | 0.5 | 0.525 | 0.475 | 0.45 | 0.45 |
| Nazism | EMBER | untouched | 10 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| Republic of Ireland | EMBER | touched | 16 | 0.75 | 0.75 | 0.5 | 0.438 | 0.438 | 0.438 | 0.375 | 0.438 |
| Republic of Ireland | EMBER | untouched | 34 | 0.618 | 0.618 | 0.647 | 0.647 | 0.618 | 0.647 | 0.618 | 0.618 |

