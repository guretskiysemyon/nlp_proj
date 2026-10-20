# X12: token-choice control for extrapolation (EMBER, 6 concepts, real erasure, train questions)

Same δ (the fair run's EMBER pick) for all. X = EMBER+X; F_own / R_own = EMBER + the same number of added tokens with the same edit norms, chosen by sentence frequency / at random from the same concept-only pool, each edited along its own component on the selected features. ret_L = chance-corrected retention of QA_train (lower = more erased); sim_L = SimdomQA_train accuracy change vs pristine (collateral). EMBER, X and pristine from X11 (same items and scoring).

## Mean over concepts

| variant | concepts | n_tokens | ret_en | ret_de | ret_es | ret_fr | ret_ru | ret_foreign | leak | sim_en | sim_foreign | mmlu_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | 6 | 50.0 | 0.634 | 0.76 | 0.794 | 0.719 | 0.976 | 0.813 | 0.178 | -0.027 | -0.007 | 1.0 |
| X | 6 | 146.333 | 0.582 | 0.601 | 0.719 | 0.494 | 0.911 | 0.681 | 0.099 | -0.04 | -0.022 | 1.0 |
| F_own | 6 | 146.333 | 0.632 | 0.573 | 0.627 | 0.566 | 0.837 | 0.651 | 0.019 | -0.03 | -0.033 | 1.0 |
| R_own | 6 | 146.333 | 0.621 | 0.721 | 0.713 | 0.705 | 0.93 | 0.767 | 0.146 | -0.027 | -0.01 | 1.0 |

- F_own − X, ret_foreign: -0.030 (lower in 4, higher in 2 of 6)
- F_own − X, ret_en: +0.050 (lower in 0, higher in 4 of 6)
- F_own − X, sim_foreign: -0.012 (lower in 5, higher in 1 of 6)
- F_own − X, sim_en: +0.010 (lower in 0, higher in 2 of 6)
- R_own − X, ret_foreign: +0.086 (lower in 2, higher in 4 of 6)
- R_own − X, ret_en: +0.039 (lower in 0, higher in 4 of 6)
- R_own − X, sim_foreign: +0.012 (lower in 1, higher in 4 of 6)
- R_own − X, sim_en: +0.013 (lower in 0, higher in 3 of 6)

## Per concept

| concept | variant | n_tokens | ret_en | ret_de | ret_es | ret_fr | ret_ru | ret_foreign | leak | sim_en | sim_foreign | mmlu_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cannabis | EMBER | 45 | 0.738 | 0.511 | 0.673 | 0.424 | 0.92 | 0.632 | -0.106 | 0.0 | 0.0 | 1.0 |
| Cannabis | X | 120 | 0.639 | 0.467 | 0.673 | 0.39 | 0.76 | 0.573 | -0.066 | -0.02 | 0.0 | 1.0 |
| Cannabis | F_own | 120 | 0.738 | 0.422 | 0.633 | 0.322 | 0.84 | 0.554 | -0.184 | 0.0 | -0.045 | 1.0 |
| Cannabis | R_own | 120 | 0.705 | 0.378 | 0.429 | 0.39 | 0.76 | 0.489 | -0.216 | 0.0 | -0.01 | 1.0 |
| Gun | EMBER | 61 | 0.843 | 1.0 | 1.0 | 0.939 | 1.0 | 0.985 | 0.142 | -0.04 | 0.005 | 1.0 |
| Gun | X | 191 | 0.765 | 1.0 | 0.946 | 0.939 | 0.943 | 0.957 | 0.192 | -0.04 | 0.01 | 1.0 |
| Gun | F_own | 191 | 0.843 | 1.0 | 1.0 | 0.758 | 0.771 | 0.882 | 0.039 | -0.04 | 0.0 | 1.0 |
| Gun | R_own | 191 | 0.843 | 1.0 | 1.0 | 1.0 | 0.943 | 0.986 | 0.143 | -0.04 | 0.01 | 1.0 |
| Nazism | EMBER | 41 | 0.556 | 0.59 | 0.641 | 0.535 | 1.0 | 0.692 | 0.136 | -0.06 | -0.035 | 1.0 |
| Nazism | X | 122 | 0.467 | 0.333 | 0.436 | 0.349 | 1.0 | 0.529 | 0.062 | -0.06 | -0.085 | 1.0 |
| Nazism | F_own | 122 | 0.556 | 0.538 | 0.333 | 0.442 | 0.935 | 0.562 | 0.006 | -0.06 | -0.075 | 1.0 |
| Nazism | R_own | 122 | 0.511 | 0.487 | 0.59 | 0.488 | 1.0 | 0.641 | 0.13 | -0.06 | -0.045 | 1.0 |
| Republic of Ireland | EMBER | 31 | 0.721 | 0.818 | 1.0 | 0.818 | 1.0 | 0.909 | 0.188 | -0.02 | 0.0 | 1.0 |
| Republic of Ireland | X | 88 | 0.674 | 0.697 | 0.806 | 0.091 | 1.0 | 0.649 | -0.025 | -0.02 | -0.01 | 1.0 |
| Republic of Ireland | F_own | 88 | 0.674 | 0.394 | 0.806 | 0.818 | 0.826 | 0.711 | 0.037 | -0.02 | -0.015 | 1.0 |
| Republic of Ireland | R_own | 88 | 0.721 | 0.818 | 1.0 | 0.818 | 1.0 | 0.909 | 0.188 | -0.02 | -0.005 | 1.0 |
| Culture of Greece | EMBER | 52 | 0.388 | 0.806 | 0.852 | 1.0 | 1.0 | 0.914 | 0.526 | -0.04 | -0.01 | 1.0 |
| Culture of Greece | X | 151 | 0.388 | 0.355 | 0.852 | 0.548 | 0.826 | 0.645 | 0.257 | -0.08 | -0.025 | 1.0 |
| Culture of Greece | F_own | 151 | 0.388 | 0.29 | 0.481 | 0.548 | 0.652 | 0.493 | 0.105 | -0.04 | -0.035 | 1.0 |
| Culture of Greece | R_own | 151 | 0.388 | 0.806 | 0.704 | 0.935 | 1.0 | 0.861 | 0.473 | -0.04 | -0.01 | 1.0 |
| Golf | EMBER | 70 | 0.559 | 0.837 | 0.6 | 0.6 | 0.939 | 0.744 | 0.185 | 0.0 | 0.0 | 1.0 |
| Golf | X | 206 | 0.559 | 0.755 | 0.6 | 0.644 | 0.939 | 0.735 | 0.176 | -0.02 | -0.02 | 1.0 |
| Golf | F_own | 206 | 0.593 | 0.796 | 0.511 | 0.511 | 1.0 | 0.705 | 0.112 | -0.02 | -0.03 | 1.0 |
| Golf | R_own | 206 | 0.559 | 0.837 | 0.556 | 0.6 | 0.879 | 0.718 | 0.159 | 0.0 | 0.0 | 1.0 |

## Added tokens (first 30)

- Cannabis / X: ▁Canna cannabis ▁Hemp ▁weed nabis ▁vape ▁CBN ▁Endoc ▁endoc ▁edibles ▁кан ▁Kush CBC ▁Bong ioid ▁Vapor Vapor EEG ativas ▁BDS madol ▁Stoner ▁pot ▁Reggae ▁Gan ▁Pot ▁Epile Huile ▁cán ▁Coffe
- Cannabis / F_own: ▁consommation ▁употреб ▁кан ▁Kons ▁выращи ▁neuro ▁dro ▁flores ▁нарко ▁Canna ▁chan ▁consom ▁tabac ▁existe ▁planta ▁semillas ▁hojas ▁efectos ▁fumée ▁Anbau ▁neur ▁Wirkung ▁médical ▁cas ▁étude ▁vielen ▁cogni ▁inhal ▁psico ▁augmentation
- Cannabis / R_own: ▁слоя ▁Grow ▁impulso ▁diferencia ▁analogues ▁Bestimmungen ▁darf ▁bildet ▁aéri ▁usar ▁bild ▁muestran ▁curios ▁eau ▁versch ▁Felder ▁дели ▁Vereine ▁Form ▁correcta ▁Joint ▁Inland ▁Gesetz ▁diesen ▁marcos ▁dadurch ▁обу ▁gering ▁constaté ▁humains
- Gun / X: ▁Guns ▁pistola ▁rifles ▁pistolet ▁Weapons ▁gunner Pist ▁Weapon pist ▁disparos ▁disparo ▁fusil ▁оружие ▁Pist ▁Kill ▁оружия ▁стре ▁tiros ▁Schützen ▁ору Стре ▁tournage ▁pist ▁armes istolet ▁grenades ▁tiro ▁balas ▁balles ▁Ruger
- Gun / F_own: ▁Feuer ▁armes ▁fusil ▁fuego ▁feu ▁fus ▁стре ▁оружия ▁оружие ▁ство ▁огне ▁calibre ▁sem ▁кали ▁dispar ▁ame ▁munitions ▁ору ▁muni ▁disparo ▁asalto ▁Der ▁canon ▁автомати ▁pist ▁vitesse ▁chasse ▁воору ▁cer ▁infan
- Gun / R_own: ▁échanges ▁электри ▁chargés ▁Krause ▁kontinuier ▁Absch ▁medida ▁irres ▁sorgen ▁suficiente ▁schwerer ▁cin ▁permanece ▁Великобрита ▁grammes ▁legislación ▁давления ▁Iro ▁виду ▁informations ▁prolong ▁langsam ▁manuelle ▁armar ▁humanas ▁utilis ▁verfü ▁Moyen ▁кусо ▁универса
- Nazism / X: ▁nazis ▁nazi ▁Gesta ▁fas ▁german fas german ▁alemana ▁alemanes ▁Volkswagen naz Allemagne ▁Alemania ▁fasc ▁Nietzsche ▁Germania Alemania ▁Niem ▁Berlín ▁Vichy ▁reich ▁Герма ▁Германия ▁Allemagne Fü ▁Buchen ▁kaiser deutsche ▁Napoléon ▁Konrad
- Nazism / F_own: ▁nazis ▁nazi ▁Alemania ▁idé ▁Ide ▁fas ▁thé ▁нем ▁Juifs ▁judíos ▁sei ▁анти ▁утвер ▁régimen ▁alemanes ▁народ ▁peuple ▁ary ▁marx ▁gobierno ▁рей ▁Juden ▁jud ▁peu ▁Das ▁rass ▁народа ▁oppos ▁идеи ▁учё
- Nazism / R_own: ▁bev ▁Buchen ▁kommun ▁симво ▁Eug ▁completamente ▁sof ▁sodass ▁vino ▁Geb ▁utilisation ▁italianos ▁stehe ▁intellectuelle ▁distances ▁bürger ▁jug ▁благо ▁успе ▁versteht ▁обучения ▁motiv ▁asesin ▁bestehenden ▁iba ▁chambres ▁quer ▁Stellung ▁tenga ▁Noruega
- Republic of Ireland / X: Cork ▁Kilkenny ▁Sligo ▁Leinster Irlande ▁Wexford ▁Derry ▁Hiber ▁Munster ▁Waterford ▁GAA ▁Tipper ▁Dund ▁Yeats ▁Paddy ▁Wales ▁Niall Ire ▁Scot Tao ▁Bangor ▁celta ▁Brexit ▁Shetland ▁Glas Dá ▁Rory ▁lagos ▁Iris Erin
- Republic of Ireland / F_own: ▁iris ▁Der ▁страны ▁Dub ▁PIB ▁Einwohner ▁taux ▁Republik ▁betrug ▁cel ▁roi ▁Reino ▁économique ▁Unido ▁Insel ▁Königreich ▁Великобрита ▁Leinster ▁cond ▁Iris ▁economía ▁européenne ▁результате ▁стране ▁основном ▁годов ▁Motor ▁brit ▁inglés ▁allem
- Republic of Ireland / R_own: ▁делает ▁größeren ▁размер ▁evolucion ▁ecles ▁chefs ▁pagar ▁venus ▁Änderungen ▁pudiera ▁führte ▁Schott ▁оборудования ▁lords ▁acceptable ▁poseen ▁appartements ▁trom ▁doux ▁sede ▁permis ▁говоря ▁различ ▁карто ▁feuilles ▁hauptsächlich ▁REIT ▁предмет ▁Chinois ▁offizielle
- Culture of Greece / X: ▁greek ▁griech ▁griega ▁griechischen ▁griego ▁Grèce ▁Griechenland ▁Griechen ▁grecque ▁Atenas ▁grec ▁Grecs ▁greco ▁Aristote ▁Platon ▁Гре ▁Bacchus ▁Ptole ▁Athen Athen Гре ▁Etrus ▁byz ▁Thess ▁hellen ▁Pythag ▁Odys ▁Perseus ▁гре ▁Cleopatra
- Culture of Greece / F_own: ▁гре ▁Гре ▁griechischen ▁griech ▁Griechen ▁grecque ▁Chr ▁grec ▁griega ▁Grèce ▁griego ▁Griechenland ▁hell ▁gre ▁cultura ▁Grecs ▁hellen ▁biz ▁Alten ▁byz ▁Anti ▁языке ▁Einfluss ▁hel ▁Der ▁langue ▁arquitectura ▁kulture ▁поэ ▁archa
- Culture of Greece / R_own: ▁Wörter ▁дости ▁mano ▁dice ▁indican ▁представляет ▁nämlich ▁Human ▁inicios ▁Verlauf ▁интере ▁len ▁kaum ▁Este ▁Recht ▁Parlement ▁verwendet ▁fasc ▁goût ▁ritual ▁supérieures ▁homer ▁sandwich ▁dios ▁ausgesch ▁Spann ▁консер ▁zog ▁vieilles ▁mittel
- Golf / X: ▁Swing Swing ▁golfing ▁Handic Hole ▁Hole ▁Penalty ▁golfe ▁Tee ▁Gol ▁penal Gol gol ▁caddy ▁trous ▁agujero ▁agujeros ▁Hazard cours ▁Caddy ▁Bunker Kurs ▁suspend ▁Cart Loch ▁Loch ▁dizaines ▁Ryder ▁курса ▁пен
- Golf / F_own: ▁Gol ▁bola ▁parcours ▁balle ▁golpes ▁Der ▁площа ▁мини ▁golfe ▁jugadores ▁campos ▁клю ▁Ball ▁inglés ▁Spieler ▁joueur ▁поле ▁удар ▁Schlä ▁pal ▁existe ▁distancia ▁spielen ▁palo ▁Loch ▁спорта ▁Green ▁peu ▁resultado ▁coups
- Golf / R_own: ▁abhängig ▁Mick ▁características ▁Driving ▁процеду ▁creó ▁inver ▁Cuts ▁programa ▁всегда ▁supprim ▁появления ▁zer ▁говорит ▁qualités ▁Gast ▁couvre ▁adecuadamente ▁Rabatt ▁simult ▁сохранить ▁Hamburg ▁eaux ▁mantiene ▁faite ▁Ernie ▁suyo ▁норм ▁percep ▁Certificado
