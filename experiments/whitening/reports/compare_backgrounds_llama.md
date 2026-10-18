# Background comparison: llama

eps_rel: wiki=0.0001, full=0.0001; eta=0.2; gamma=1; target=page tokens

## 1. Top-50 eigenvector subspaces (principal-angle sines)

| pair | mean sine | max | min | #sines < 0.5 |
|---|---|---|---|---|
| wiki_vs_full | 0.405 | 1.000 | 0.066 | 36 |
| wiki_vs_latin | 0.280 | 0.998 | 0.037 | 42 |
| full_vs_latin | 0.315 | 1.000 | 0.041 | 41 |

## 2. Top-5 cPCA directions: top-15 background tokens by |projection| (* = token on the concept page)

### Harry Potter
**wiki** (mean in-page fraction 0.92)

- p1 (γ=6.3, in-page 1.00): *'Ġthe', *'Ġin', *'Ġa', *'Ġand', *'Ġis', *'Ġto', *'Ġare', *'Ġwas', *'Ġthat', *'Ġit', *'Ġthey', *'Ġthis', *'Ġwere', *'Ġon', *'Ġan'
- p2 (γ=5.1, in-page 1.00): *'6', *'4', *'7', *'5', *'8', *'3', *'2', *'9', *'1', *'0', *'18', *'10', *'12', *'19', *'15'
- p3 (γ=4.5, in-page 0.87): *'ĠSeptember', *'ĠJanuary', *'ĠDecember', *'ĠOctober', *'ĠFebruary', *'ĠNovember', *'ĠApril', *'ĠJuly', *'ĠAugust', *'ĠJune', *'ĠMarch', *'ĠMay', *'September', 'April', 'October'
- p4 (γ=4.3, in-page 0.80): *'ĠJanuary', *'ĠSeptember', *'ĠNovember', *'ĠFebruary', *'ĠJuly', *'ĠDecember', *'ĠApril', *'ĠOctober', *'ĠJune', *'ĠAugust', *'ĠMarch', *'ĠMay', 'December', 'November', 'January'
- p5 (γ=4.2, in-page 0.93): *'17', *'14', *'16', *'19', *'18', *'15', *'12', *'24', *'Ġtwelve', *'Ġfourteen', *'21', *'11', *'10', *'Ġeight', '13'

**full** (mean in-page fraction 0.97)

- p1 (γ=9.1, in-page 1.00): *'Ġthe', *'Ġis', *'Ġare', *'Ġin', *'Ġwas', *'Ġa', *'Ġhas', *'Ġto', *'Ġand', *'Ġwere', *'Ġhave', *'Ġbe', *'Ġthey', *'Ġit', *'Ġcan'
- p2 (γ=6.6, in-page 1.00): *'ĠG', *'ĠH', *'ĠC', *'4', *'ĠB', *'ĠD', *'6', *'ĠM', *'ĠS', *'2', *'8', *'5', *'3', *'ĠK', *'7'
- p3 (γ=5.8, in-page 1.00): *'Ġnovels', *'Ġnovel', *'Ġstories', *'Ġfiction', *'Ġstory', *'Ġfictional', *'Ġliterary', *'Ġbooks', *'Ġpoetry', *'Ġbecame', *'Ġbook', *'Ġthriller', *'Ġappears', *'Ġadventure', *'Ġwriters'
- p4 (γ=5.3, in-page 1.00): *'ĠJanuary', *'ĠSeptember', *'ĠDecember', *'ĠFebruary', *'ĠOctober', *'ĠNovember', *'ĠApril', *'ĠJuly', *'ĠJune', *'ĠAugust', *'ĠMarch', *'Ġseven', *'Ġnine', *'Ġeight', *'Ġtwelve'
- p5 (γ=5.1, in-page 0.87): *'ĠSeptember', *'ĠJanuary', *'ĠOctober', *'ĠNovember', *'ĠFebruary', *'ĠDecember', *'ĠApril', *'ĠJuly', *'ĠJune', *'ĠAugust', *'ĠMarch', *'September', *'ĠMay', 'December', 'November'

### Ancient Rome
**wiki** (mean in-page fraction 1.00)

- p1 (γ=5.1, in-page 1.00): *'Ġthe', *'Ġin', *'Ġa', *'Ġto', *'Ġand', *'Ġof', *'Ġwas', *'Ġfrom', *'Ġfor', *'Ġthat', *'Ġas', *'Ġwere', *'Ġwith', *'Ġare', *'Ġat'
- p2 (γ=4.7, in-page 1.00): *'30', *'20', *'15', *'10', *'18', *'40', *'14', *'16', *'25', *'19', *'13', *'50', *'12', *'17', *'28'
- p3 (γ=4.2, in-page 1.00): *'30', *'35', *'40', *'50', *'28', *'80', *'20', *'38', *'60', *'25', *'17', *'27', *'48', *'15', *'36'
- p4 (γ=3.9, in-page 1.00): *'Ġemperor', *'Ġempire', *'Ġarmies', *'Ġkingdom', *'ĠEmperor', *'Ġarmy', *'Ġking', *'Ġkingdoms', *'ĠEmpire', *'Ġkings', *'Ġmonarch', *'Ġmonarchy', *'Ġqueen', *'Ġwars', *'Ġconquest'
- p5 (γ=3.8, in-page 1.00): *'Ġremained', *'Ġconsisted', *'Ġfour', *'Ġfew', *'Ġformed', *'Ġbecame', *'Ġthree', *'ĠJuly', *'Ġacted', *'Ġoccurred', *'Ġhundreds', *'Ġresulted', *'ĠDecember', *'Ġachieved', *'ĠApril'

**full** (mean in-page fraction 1.00)

- p1 (γ=7.6, in-page 1.00): *'Ġbecame', *'Ġmade', *'Ġused', *'Ġwas', *'Ġfirst', *'Ġwere', *'Ġthe', *'Ġhad', *'Ġin', *'Ġmany', *'Ġa', *'Ġboth', *'Ġknown', *'Ġcame', *'Ġoften'
- p2 (γ=6.2, in-page 1.00): *'30', *'15', *'20', *'40', *'18', *'10', *'28', *'25', *'50', *'14', *'16', *'19', *'35', *'13', *'27'
- p3 (γ=5.5, in-page 1.00): *'Ġempire', *'Ġemperor', *'Ġarmy', *'Ġkingdom', *'Ġarmies', *'Ġking', *'ĠEmperor', *'Ġkingdoms', *'Ġkings', *'ĠEmpire', *'Ġsoldiers', *'Ġmonarchy', *'Ġqueen', *'Ġmonarch', *'Ġwars'
- p4 (γ=5.3, in-page 1.00): *'38', *'40', *'80', *'35', *'30', *'60', *'50', *'70', *'28', *'36', *'37', *'48', *'54', *'68', *'31'
- p5 (γ=5.0, in-page 1.00): *'Ġlargely', *'Ġconquered', *'Ġlarge', *'Ġhundreds', *'Ġdozens', *'Ġfour', *'Ġoften', *'Ġnearly', *'Ġrelatively', *'Ġhighly', *'Ġmainly', *'Ġprimarily', *'Ġmostly', *'Ġfew', *'Ġseized'

### Valentine's Day
**wiki** (mean in-page fraction 0.91)

- p1 (γ=7.1, in-page 1.00): *'Ġthe', *'Ġa', *'Ġin', *'Ġto', *'Ġand', *'Ġis', *'ĠH', *'Ġan', *',', *'ĠB', *'ĠG', *'ĠM', *'Ġof', *'Ġare', *'ĠP'
- p2 (γ=5.7, in-page 1.00): *'Ġoften', *'ĠL', *'M', *'ĠG', *'m', *'ĠF', *'ĠP', *'B', *'ĠM', *'ĠK', *'ĠB', *'ĠS', *'ĠH', *'6', *'g'
- p3 (γ=5.3, in-page 0.87): *'197', *'196', *'195', *'198', *'194', *'199', *'193', *'200', '192', *'190', '191', *'201', *'184', *'185', *'186'
- p4 (γ=4.8, in-page 0.80): *'ĠJanuary', *'ĠFebruary', *'ĠDecember', *'ĠSeptember', *'ĠNovember', *'ĠAugust', *'ĠJune', *'ĠApril', *'ĠJuly', 'ĠOctober', *'ĠMarch', *'ĠMay', *'February', 'December', 'January'
- p5 (γ=4.5, in-page 0.87): *'12', *'15', *'18', *'14', *'16', *'13', *'20', *'19', *'22', *'11', *'21', *'24', '10', '17', *'23'

**full** (mean in-page fraction 0.88)

- p1 (γ=9.7, in-page 1.00): *'Ġthe', *'Ġa', *'Ġin', *'Ġis', *'Ġto', *'Ġand', *'Ġan', *'ĠB', *'Ġwas', *'Ġare', *'ĠH', *'ĠG', *'Ġof', *'ĠC', *'ĠI'
- p2 (γ=7.5, in-page 1.00): *'ĠB', *'ĠG', *'ĠL', *'ĠH', *'ĠP', *'ĠS', *'ĠF', *'B', *'ĠD', *'ĠK', *'ĠM', *'ĠN', *'ĠC', *'h', *'M'
- p3 (γ=6.3, in-page 0.73): *'195', *'196', *'197', *'194', *'198', *'193', *'199', '192', *'190', *'185', *'186', *'184', '191', '189', '188'
- p4 (γ=6.0, in-page 0.87): *'195', *'196', *'194', *'197', *'198', *'193', *'199', '192', *'184', *'190', '191', *'185', *'183', *'186', *'200'
- p5 (γ=5.5, in-page 0.80): *'ĠJanuary', *'ĠFebruary', *'ĠDecember', *'ĠNovember', *'ĠSeptember', *'ĠJune', *'ĠApril', *'ĠJuly', 'ĠOctober', *'ĠAugust', *'ĠMarch', *'ĠMay', *'February', 'January', 'November'

## 3. Concept-feature overlap (rho >= 2 features, best Jaccard match of token supports wiki -> full)

| concept | gamma | n_tau wiki | n_tau full | mean best J | #J>=0.5 | union J | AUC wiki / full | purity wiki / full |
|---|---|---|---|---|---|---|---|---|
| Harry Potter | 1 | 181 | 183 | 0.23 | 9 | 0.78 | 0.872 / 0.870 | 0.96 / 0.94 |
| Harry Potter | 0 | 45 | 54 | 0.27 | 9 | 0.49 | 0.619 / 0.637 | 0.71 / 0.59 |
| Ancient Rome | 1 | 181 | 176 | 0.22 | 11 | 0.74 | 0.852 / 0.854 | 0.94 / 0.86 |
| Ancient Rome | 0 | 45 | 60 | 0.23 | 4 | 0.43 | 0.606 / 0.645 | 0.56 / 0.62 |
| Valentine's Day | 1 | 185 | 184 | 0.21 | 13 | 0.73 | 0.879 / 0.857 | 0.96 / 0.96 |
| Valentine's Day | 0 | 44 | 55 | 0.25 | 9 | 0.45 | 0.611 / 0.615 | 0.37 / 0.31 |
