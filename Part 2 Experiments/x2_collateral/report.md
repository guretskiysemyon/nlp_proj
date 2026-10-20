# X2: collateral exposure of the edited tokens in ordinary English Wikipedia text

Background: 300,000 sentences, 8,365,171 tokens. `sent_share_%`: sentences with >= 1 edited token; `per_10k_tokens`: edited-token occurrences per 10,000 tokens.

| concept | method | n_edited | sent_share_% | per_10k_tokens |
|---|---|---|---|---|
| Cannabis | EMBER | 45 | 0.192 | 0.78 |
| Cannabis | CINDER | 34 | 0.114 | 0.47 |
| Gun | EMBER | 61 | 0.918 | 3.73 |
| Gun | CINDER | 67 | 1.3 | 5.3 |
| Nazism | EMBER | 41 | 0.486 | 2.03 |
| Nazism | CINDER | 0 | 0.0 | 0.0 |
| Republic of Ireland | EMBER | 31 | 0.806 | 3.46 |
| Republic of Ireland | CINDER | 0 | 0.0 | 0.0 |
| Culture of Greece | EMBER | 52 | 0.558 | 2.38 |
| Culture of Greece | CINDER | 106 | 2.781 | 11.91 |
| Golf | EMBER | 70 | 0.799 | 3.22 |
| Golf | CINDER | 44 | 0.951 | 3.85 |

## Most frequent edited tokens

- **Cannabis / EMBER**: `▁Mari` 91, `▁BCE` 78, `▁CB` 76, `▁fruits` 72, `CB` 47, `▁psychedelic` 44, `▁marijuana` 36, `Mari` 22, `CBN` 20, `▁hash` 19, `▁cannabis` 19, `▁Hash` 16, `mari` 15, `▁CBD` 12, `ativa` 9
- **Cannabis / CINDER**: `▁CB` 76, `▁tetra` 64, `CB` 47, `▁marijuana` 36, `Mari` 22, `CBN` 20, `▁cannabis` 19, `▁hash` 19, `▁Hash` 16, `mari` 15, `▁CBD` 12, `▁Cannabis` 7, `▁indica` 7, `▁hemp` 5, `▁legalization` 5
- **Gun / EMBER**: `▁gun` 342, `▁dead` 265, `▁killing` 206, `▁kill` 197, `▁hunting` 138, `▁Sho` 131, `▁sh` 123, `▁weapon` 117, `▁Gun` 109, `▁Confederation` 103, `▁shoot` 96, `▁rifle` 86, `▁firing` 85, `▁shock` 83, `▁sword` 78
- **Gun / CINDER**: `▁hit` 816, `▁gun` 342, `▁ball` 317, `▁attacked` 225, `▁killing` 206, `▁kill` 197, `▁struck` 176, `▁hunting` 138, `▁Sho` 131, `▁weapon` 117, `▁Gun` 109, `▁balls` 108, `▁shoot` 96, `▁rifle` 86, `▁firing` 85
- **Nazism / EMBER**: `▁Berlin` 347, `▁Nazi` 202, `▁Germans` 133, `German` 127, `▁Hitler` 93, `▁Reich` 72, `▁Holocaust` 69, `▁Prussian` 62, `▁Germanic` 55, `▁Stalin` 51, `▁Fas` 49, `▁Naz` 43, `▁Nazis` 41, `▁fascist` 40, `▁Nuremberg` 27
- **Republic of Ireland / EMBER**: `▁Scotland` 755, `▁Irish` 660, `▁Dublin` 210, `▁Celtic` 127, `▁Belfast` 109, `▁Gaelic` 104, `▁Ulster` 103, `▁Dá` 88, `▁Scots` 83, `▁Cork` 80, `▁€` 78, `▁Kerry` 52, `Ireland` 48, `Irish` 45, `▁Sinn` 40
- **Culture of Greece / EMBER**: `▁Greek` 606, `▁Greece` 182, `▁Athens` 179, `▁Byzantine` 115, `▁Homer` 58, `▁Greeks` 58, `▁Constantinople` 56, `Greek` 52, `▁Hellenic` 48, `▁Arist` 35, `▁Sparta` 35, `▁Crete` 33, `▁Mino` 29, `▁Hellen` 27, `▁Corinth` 25
- **Culture of Greece / CINDER**: `▁European` 1267, `▁Roman` 900, `▁Olympic` 725, `▁Italy` 709, `▁Greek` 606, `▁Latin` 557, `▁ancient` 556, `▁Rome` 468, `ath` 436, `▁Alexander` 376, `▁Ottoman` 192, `▁Greece` 182, `▁Athens` 179, `▁Romanian` 169, `▁Persian` 166
- **Golf / EMBER**: `▁courses` 229, `▁amateur` 197, `▁golf` 190, `▁penalty` 128, `▁gap` 111, `▁Tiger` 106, `▁polo` 99, `▁Golf` 98, `▁hole` 93, `▁bats` 83, `course` 82, `▁Woods` 77, `▁Amateur` 72, `hole` 70, `▁swing` 69
- **Golf / CINDER**: `▁Sports` 433, `▁sport` 321, `▁ball` 317, `▁baseball` 302, `▁golf` 190, `ball` 152, `▁gap` 111, `▁balls` 108, `▁Tiger` 106, `▁polo` 99, `▁Golf` 98, `▁Palmer` 94, `▁hole` 93, `▁Woods` 77, `hole` 70
