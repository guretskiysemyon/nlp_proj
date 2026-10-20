# X1: edited token sets and edit vectors, EMBER vs CINDER

Edit sets as embed_edit applies them (EMBER: Part 1 features; CINDER: L123, seed 42). `median_cos_shared`: median cosine between the two methods' edit vectors on the tokens both edit.

| concept | EMBER | CINDER | both | only_EMBER | only_CINDER | jaccard | median_cos_shared |
|---|---|---|---|---|---|---|---|
| Cannabis | 45 | 34 | 33 | 12 | 1 | 0.72 | 0.68 |
| Gun | 61 | 67 | 54 | 7 | 13 | 0.73 | 0.83 |
| Nazism | 41 | 0 | 0 | 41 | 0 | 0.0 |  |
| Republic of Ireland | 31 | 0 | 0 | 31 | 0 | 0.0 |  |
| Culture of Greece | 52 | 106 | 52 | 0 | 54 | 0.49 | 0.81 |
| Golf | 70 | 44 | 33 | 37 | 11 | 0.41 | 0.64 |

## Kind of edited token (share)

| concept | method | piece | short/other word | word |
|---|---|---|---|---|
| Cannabis | CINDER | 0.41 | 0.03 | 0.56 |
| Cannabis | EMBER | 0.4 | 0.02 | 0.58 |
| Culture of Greece | CINDER | 0.18 | 0.0 | 0.82 |
| Culture of Greece | EMBER | 0.12 | 0.0 | 0.88 |
| Golf | CINDER | 0.23 | 0.0 | 0.77 |
| Golf | EMBER | 0.19 | 0.0 | 0.81 |
| Gun | CINDER | 0.15 | 0.0 | 0.85 |
| Gun | EMBER | 0.16 | 0.02 | 0.82 |
| Nazism | EMBER | 0.27 | 0.0 | 0.73 |
| Republic of Ireland | EMBER | 0.1 | 0.06 | 0.84 |

## Relative edit size and alignment with the token (medians)

| concept | method | rel_edit | cos_with_token |
|---|---|---|---|
| Cannabis | CINDER | 0.264 | 0.342 |
| Cannabis | EMBER | 0.375 | 0.51 |
| Culture of Greece | CINDER | 0.248 | 0.487 |
| Culture of Greece | EMBER | 0.663 | 0.7 |
| Golf | CINDER | 0.203 | 0.213 |
| Golf | EMBER | 0.299 | 0.459 |
| Gun | CINDER | 0.245 | 0.311 |
| Gun | EMBER | 0.376 | 0.516 |
| Nazism | EMBER | 0.427 | 0.6 |
| Republic of Ireland | EMBER | 0.426 | 0.59 |

## Tokens only one method edits (largest edits first, up to 25)

**Cannabis**
- only EMBER (12): `Hash`, `▁Mari`, `ativa`, `▁bong`, `▁κάν`, `▁psychedelic`, `▁fruits`, `penes`, `▁dispens`, `Tetra`, `▁BCE`, `▁hydropon`
- only CINDER (1): `▁tetra`

**Gun**
- only EMBER (7): `▁slaughter`, `▁dead`, `▁burning`, `▁ordnance`, `▁sh`, `oters`, `▁Confederation`
- only CINDER (13): `▁hit`, `▁struck`, `▁strikes`, `▁cartridge`, `▁cartridges`, `▁ball`, `▁Remington`, `▁musk`, `▁blow`, `▁balls`, `▁striker`, `▁attacked`, `stri`

**Nazism**
- only EMBER (41): `▁Nazi`, `▁Hitler`, `Nazi`, `Hitler`, `▁Nazis`, `Naz`, `▁Naz`, `Adolf`, `German`, `▁Fas`, `▁Germans`, `▁Berlin`, `Fas`, `▁fascist`, `▁Reich`, `Reich`, `▁Weimar`, `▁Mussolini`, `▁Führer`, `▁fascism`, `▁Fascism`, `▁Reichstag`, `▁Reichs`, `▁Nuremberg`, `Volkswagen`
- only CINDER (0): 

**Republic of Ireland**
- only EMBER (31): `▁Dublin`, `Dublin`, `▁Irish`, `Irish`, `Ireland`, `▁Belfast`, `▁Galway`, `▁Cork`, `▁Limerick`, `▁Donegal`, `▁Ulster`, `▁Celtic`, `▁Kildare`, `▁Garda`, `▁Sinn`, `▁Gaelic`, `▁Scotland`, `▁Bally`, `▁Conor`, `▁€`, `▁Gael`, `▁Scots`, `▁Dá`, `▁IRA`, `▁Fian`
- only CINDER (0): 

**Culture of Greece**
- only EMBER (0): 
- only CINDER (54): `▁Cyprus`, `▁Mediterranean`, `▁Roman`, `▁Cypri`, `Roman`, `▁Ancient`, `▁Rome`, `Mediterranean`, `Ancient`, `▁Romans`, `▁Egyptians`, `▁Italy`, `opoulos`, `▁Romanian`, `▁ancient`, `▁Persian`, `▁Olympic`, `▁Odyssey`, `▁Constantine`, `▁Ottoman`, `retan`, `▁Latin`, `▁Persians`, `▁Alexand`, `Gre`

**Golf**
- only EMBER (37): `▁swing`, `▁swings`, `swing`, `▁swinging`, `▁amateur`, `▁Amateur`, `▁penalties`, `▁penalty`, `▁amateurs`, `course`, `icap`, `Amateur`, `▁courses`, `▁handic`, `Courses`, `Penalty`, `▁hazards`, `▁pendulum`, `▁hazard`, `▁bunkers`, `▁hamstring`, `▁traps`, `▁infractions`, `▁novice`, `▁cart`
- only CINDER (11): `▁baseball`, `▁ball`, `ball`, `▁balls`, `▁Sports`, `green`, `▁sport`, `▁balles`, `hol`, `▁archery`, `▁Palmer`

