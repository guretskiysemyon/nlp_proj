# Background comparison: gemma

eps_rel: wiki=0.0001, full=0.0001; eta=0.05; gamma=1; target=page tokens

## 1. Top-50 eigenvector subspaces (principal-angle sines)

| pair | mean sine | max | min | #sines < 0.5 |
|---|---|---|---|---|
| wiki_vs_full | 0.408 | 0.999 | 0.030 | 34 |
| wiki_vs_latin | 0.297 | 0.993 | 0.022 | 41 |
| full_vs_latin | 0.249 | 0.988 | 0.019 | 44 |

## 2. Top-5 cPCA directions: top-15 background tokens by |projection| (* = token on the concept page)

### Harry Potter
**wiki** (mean in-page fraction 0.96)

- p1 (γ=14.2, in-page 1.00): *'▁the', *'▁that', *'▁this', *'▁in', *'▁with', *'▁to', *'▁as', *'▁for', *'▁they', *'▁and', *'▁it', *'▁from', *'▁which', *'▁a', *'▁these'
- p2 (γ=9.8, in-page 1.00): *'1', *'3', *'2', *'5', *'4', *'6', *'7', *'8', *'9', *'0', *'▁', *'.', *'-', *'$', *':'
- p3 (γ=8.6, in-page 1.00): *'▁as', *'5', *'▁the', *'▁with', *'▁and', *'4', *'3', *'▁for', *'6', *'7', *'▁to', *'▁their', *'1', *'2', *'8'
- p4 (γ=7.9, in-page 0.80): *',', *'.', *':', *'"', *'-', *';', *"'", *')', *'/', '(', *'?', *'",', '!', *'),', ',"'
- p5 (γ=7.7, in-page 1.00): *'.', *',', *':', *'"', *'▁could', *'▁would', *'▁was', *';', *'▁will', *'-', *'▁should', *'▁were', *'▁might', *'▁is', *'▁has'

**full** (mean in-page fraction 0.99)

- p1 (γ=20.1, in-page 1.00): *'▁the', *'▁that', *'▁this', *'▁which', *'▁with', *'▁they', *'▁in', *'▁to', *'▁from', *'▁for', *'▁where', *'▁there', *'▁who', *'▁it', *'▁when'
- p2 (γ=10.2, in-page 1.00): *'▁the', *'▁that', *'▁with', *'▁they', *'▁as', *'▁and', *'▁which', *'▁it', *'▁this', *'▁when', *'▁a', *'▁to', *'▁from', *'▁their', *'▁in'
- p3 (γ=9.2, in-page 1.00): *'1', *'2', *'3', *'5', *'4', *'8', *'6', *'7', *'9', *'▁A', *'0', *'▁The', *'▁D', *'▁I', *'A'
- p4 (γ=8.1, in-page 1.00): *'3', *'5', *'2', *'4', *'6', *'1', *'7', *'8', *'9', *'0', *'-', *'▁the', *"'", *'▁wrote', *'▁their'
- p5 (γ=7.4, in-page 0.93): *'5', *'7', *'6', *'8', *'4', *'9', *'3', *'1', *'2', *'0', *',', *'.', *"'", *'"', '７'

### Ancient Rome
**wiki** (mean in-page fraction 1.00)

- p1 (γ=10.9, in-page 1.00): *'▁the', *'▁that', *'▁with', *'▁in', *'▁as', *'▁from', *'▁for', *'▁this', *'▁to', *'▁and', *'▁at', *'▁which', *'▁when', *'▁a', *'▁they'
- p2 (γ=7.7, in-page 1.00): *'1', *'2', *'3', *'5', *'4', *'6', *'7', *'8', *'9', *'0', *'▁', *'.', *'a', *',', *'-'
- p3 (γ=7.1, in-page 1.00): *'3', *'5', *'4', *'2', *'1', *'7', *'6', *'8', *'9', *'0', *'▁with', *'▁as', *'▁', *'▁for', *'▁from'
- p4 (γ=6.7, in-page 1.00): *'6', *'5', *'7', *'3', *'9', *'4', *'8', *'2', *'1', *'▁became', *'0', *'▁had', *'▁began', *'▁appeared', *'▁started'
- p5 (γ=6.1, in-page 1.00): *',', *"'", *'.', *'"', *'-', *'5', *'7', *';', *':', *'6', *'8', *')', *'4', *'9', *"',"

**full** (mean in-page fraction 0.92)

- p1 (γ=16.2, in-page 1.00): *'▁the', *'▁that', *'▁with', *'▁which', *'▁in', *'▁this', *'▁from', *'▁they', *'▁only', *'▁both', *'▁to', *'▁also', *'▁where', *'▁other', *'▁one'
- p2 (γ=8.9, in-page 0.73): *'▁brought', *'▁formed', '▁offered', *'▁provided', *'▁obtained', *'▁introduced', *'▁entered', *'▁started', '▁performed', *'▁created', '▁produced', *'▁played', '▁carried', *'▁taught', *'▁represented'
- p3 (γ=8.7, in-page 0.87): *'▁a', *'as', *'a', *'in', *'▁the', *'an', 'at', *'and', *'▁as', *'i', *'▁and', *'▁in', *'▁to', 'it', *'on'
- p4 (γ=7.7, in-page 1.00): *'us', *'u', *'es', *'al', *'is', *'ed', *'y', *'o', *'c', *'ic', *'an', *'os', *'i', *'er', *'ian'
- p5 (γ=6.8, in-page 1.00): *'▁their', *'▁they', *'▁when', *'▁this', *'▁empires', *'▁the', *'The', *'▁that', *'Their', *'▁which', *'▁emperors', *'▁these', *'▁there', *'▁with', *'They'

### Valentine's Day
**wiki** (mean in-page fraction 0.99)

- p1 (γ=15.4, in-page 1.00): *'▁the', *'▁that', *'▁this', *'▁and', *'▁in', *'▁a', *'▁to', *'▁for', *'▁as', *'▁it', *'▁they', *'▁with', *'▁these', *'▁from', *'▁their'
- p2 (γ=11.2, in-page 1.00): *'1', *'3', *'5', *'2', *'4', *'6', *'7', *'8', *'9', *'0', *'▁', *'-', *'.', *'"', *'('
- p3 (γ=9.5, in-page 1.00): *',', *'"', *"'", *'.', *')', *':', *'",', *';', *'-', *'5', *'6', *',"', *'7', *'4', *'8'
- p4 (γ=8.5, in-page 0.93): *'▁the', *'▁this', *'▁these', *'▁their', *'▁year', *'▁it', *'▁as', *'▁its', *'▁when', *'▁for', *'▁day', *'▁they', *'the', *'▁with', '▁whilst'
- p5 (γ=8.4, in-page 1.00): *'▁A', *',', *'▁I', *'a', *'.', *'▁a', *'▁the', *'"', *'A', *'▁D', *'▁S', *'▁M', *'I', *'▁F', *'▁G'

**full** (mean in-page fraction 0.97)

- p1 (γ=20.1, in-page 1.00): *'▁the', *'▁that', *'▁this', *'▁they', *'▁to', *'▁in', *'▁with', *'▁a', *'▁which', *'▁it', *'▁for', *'▁there', *'▁from', *'▁as', *'▁their'
- p2 (γ=11.3, in-page 0.93): *'a', *'i', *'e', *'d', *'1', *'2', *'A', *'n', *'3', *'u', *'b', 'm', *'f', *'4', *'r'
- p3 (γ=10.1, in-page 0.93): *'▁when', *'▁Western', *'▁they', *'▁with', *'▁that', '▁whenever', *'▁European', *'▁country', *'▁British', *'▁Eastern', *'▁from', *'▁though', *'▁Roman', *'▁although', *'▁Catholic'
- p4 (γ=8.9, in-page 1.00): *'▁the', *'5', *'3', *'7', *'6', *'1', *'2', *'4', *'8', *'9', *'▁Italy', *'▁Spain', *'▁England', *'▁Japan', *'▁India'
- p5 (γ=8.8, in-page 1.00): *'5', *'7', *'6', *'4', *'3', *'8', *'9', *'2', *'1', *'0', *'▁a', *'▁an', *'▁the', *'▁D', *'▁in'

## 3. Concept-feature overlap (rho >= 2 features, best Jaccard match of token supports wiki -> full)

| concept | gamma | n_tau wiki | n_tau full | mean best J | #J>=0.5 | union J | AUC wiki / full | purity wiki / full |
|---|---|---|---|---|---|---|---|---|
| Harry Potter | 1 | 61 | 68 | 0.44 | 27 | 0.65 | 0.704 / 0.743 | 0.94 / 0.96 |
| Harry Potter | 0 | 26 | 27 | 0.48 | 13 | 0.52 | 0.588 / 0.611 | 0.94 / 0.94 |
| Ancient Rome | 1 | 52 | 61 | 0.43 | 21 | 0.57 | 0.676 / 0.725 | 0.88 / 0.96 |
| Ancient Rome | 0 | 18 | 31 | 0.56 | 12 | 0.45 | 0.565 / 0.625 | 0.75 / 0.90 |
| Valentine's Day | 1 | 67 | 72 | 0.43 | 27 | 0.58 | 0.669 / 0.723 | 0.90 / 0.78 |
| Valentine's Day | 0 | 25 | 29 | 0.43 | 10 | 0.44 | 0.573 / 0.611 | 0.82 / 0.76 |
