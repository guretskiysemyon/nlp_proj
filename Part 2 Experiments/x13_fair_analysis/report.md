# X13: analyses of the fair run (held-out QA_test, 18 concepts)

Reliable cells only (pristine QA_test ≥ 0.4); leak = mean(ret de/es/fr/ru) − ret_en. See the script docstring for the definitions.

## 1. Paired leak differences (two-level bootstrap 95% CI, Wilcoxon, Holm over 6 comparisons)

| analysis | pair | mean | ci_low | ci_high | lower | higher | equal | wilcoxon_p | holm_p |
|---|---|---|---|---|---|---|---|---|---|
| primary: de/es/fr/ru, clipped | ember_x − ember | -0.059 | -0.102 | -0.023 | 12 | 2 | 4 | 0.0012 | 0.0072 |
| primary: de/es/fr/ru, clipped | cinder_x − cinder | -0.039 | -0.088 | -0.004 | 10 | 4 | 4 | 0.0203 | 0.1015 |
| primary: de/es/fr/ru, clipped | cinder − ember | 0.041 | -0.06 | 0.137 | 9 | 9 | 0 | 0.3927 | 1.0 |
| primary: de/es/fr/ru, clipped | cinder_x − ember_x | 0.061 | -0.035 | 0.15 | 6 | 12 | 0 | 0.1084 | 0.4336 |
| primary: de/es/fr/ru, clipped | ember_hj − ember | 0.048 | -0.006 | 0.119 | 0 | 4 | 14 |  | 1.0 |
| primary: de/es/fr/ru, clipped | cinder − ember_hj | -0.007 | -0.094 | 0.081 | 11 | 7 | 0 | 0.7337 | 1.0 |
| unclipped retention | ember_x − ember | -0.055 | -0.096 | -0.012 | 11 | 3 | 4 | 0.0023 | 0.0138 |
| unclipped retention | cinder_x − cinder | -0.044 | -0.097 | 0.008 | 10 | 4 | 4 | 0.0353 | 0.1765 |
| unclipped retention | cinder − ember | 0.053 | -0.059 | 0.192 | 9 | 9 | 0 | 0.3692 | 1.0 |
| unclipped retention | cinder_x − ember_x | 0.064 | -0.041 | 0.193 | 7 | 11 | 0 | 0.1084 | 0.4336 |
| unclipped retention | ember_hj − ember | 0.053 | -0.021 | 0.149 | 0 | 4 | 14 |  | 1.0 |
| unclipped retention | cinder − ember_hj | -0.0 | -0.096 | 0.129 | 11 | 7 | 0 | 0.865 | 1.0 |
| de/es/fr only | ember_x − ember | -0.065 | -0.114 | -0.022 | 12 | 2 | 4 | 0.0017 | 0.0102 |
| de/es/fr only | cinder_x − cinder | -0.039 | -0.092 | 0.002 | 10 | 4 | 4 | 0.0295 | 0.118 |
| de/es/fr only | cinder − ember | 0.047 | -0.054 | 0.144 | 7 | 11 | 0 | 0.1815 | 0.5445 |
| de/es/fr only | cinder_x − ember_x | 0.073 | -0.02 | 0.161 | 5 | 13 | 0 | 0.0208 | 0.104 |
| de/es/fr only | ember_hj − ember | 0.04 | -0.014 | 0.097 | 0 | 4 | 14 |  | 1.0 |
| de/es/fr only | cinder − ember_hj | 0.007 | -0.078 | 0.1 | 9 | 9 | 0 | 0.7987 | 1.0 |

## 2. Mechanism: extrapolated arm vs parent, share of pristine-correct questions lost

group: old = has a parent edit token; new = only an added token; none = neither.

| arm | lang_kind | group | questions | lost_parent | lost_x | gain |
|---|---|---|---|---|---|---|
| cinder_x | en | new | 8 | 0.0 | 0.25 | 0.25 |
| cinder_x | en | none | 91 | 0.0 | 0.0 | 0.0 |
| cinder_x | en | old | 446 | 0.309 | 0.323 | 0.014 |
| cinder_x | foreign | new | 434 | 0.005 | 0.237 | 0.232 |
| cinder_x | foreign | none | 694 | 0.001 | 0.006 | 0.005 |
| cinder_x | foreign | old | 632 | 0.294 | 0.293 | -0.001 |
| ember_x | en | new | 3 | 0.0 | 0.333 | 0.333 |
| ember_x | en | none | 172 | 0.017 | 0.012 | -0.005 |
| ember_x | en | old | 363 | 0.372 | 0.399 | 0.027 |
| ember_x | foreign | new | 401 | 0.005 | 0.367 | 0.362 |
| ember_x | foreign | none | 829 | 0.006 | 0.011 | 0.005 |
| ember_x | foreign | old | 530 | 0.442 | 0.453 | 0.011 |

## 3. Coverage vs leak over edited reliable (concept, foreign language) cells

| arms | cells | rho_coverage | p | rho_gap | p_gap |
|---|---|---|---|---|---|
| all arms | 296 | -0.58 | 7.8e-28 | 0.65 | 7.2e-37 |
| ember | 55 | -0.77 | 5.4e-12 | 0.75 | 3.4e-11 |
| cinder | 58 | -0.61 | 3.4e-07 | 0.63 | 1.1e-07 |
| ember_hj | 70 | -0.74 | 3.3e-13 | 0.68 | 6.3e-11 |
| ember_x | 55 | -0.35 | 9.1e-03 | 0.62 | 5.1e-07 |
| cinder_x | 58 | -0.42 | 1.2e-03 | 0.44 | 4.7e-04 |

## 4. Failure cases

See `failures.md`.

## 5. Tokens extrapolation adds, 12 concepts outside the first 6

| concept | arm | n_added | whole_word | foreign_only | first_25 |
|---|---|---|---|---|---|
| Ancient Rome | ember_x | 125 | 0.59 | 0.93 | ▁séparation ▁orientale ▁Perseus ▁Priester ust ▁auszu ▁religiosas ▁vocale ▁commerciales ▁Antig ▁autorité ▁Napole ▁citoyen ▁Egipto ▁Brutus ▁conquête usas ▁Pompe ▁intérieure ▁Augusto Titus ularis ▁combatt ▁Cornelia ▁Odysseus |
| Ancient Rome | cinder_x | 108 | 0.74 | 0.85 | ▁Mith ▁römischen ▁Anci roman ▁Pon ▁Latein ▁roman rom рим ▁Marcell ▁romaine enseits ▁Napole inians ▁romain ▁Brutus ▁Venus ▁Vesu ▁consular ▁Pompe Rö ▁Augusto Titus ▁grecque ▁Cornelia |
| Baseball | ember_x | 54 | 0.72 | 0.87 | ispiel ▁Marlins ▁броса rücken ▁pid ▁Pittsburgh ▁lancé ▁Throw ▁werfen isbol ▁lancer ▁reliever leich ▁emp ▁responsabilité ▁конф ▁schlagen Dug werfen ▁CIO Ump ▁Cricket ▁Pitch ▁batte Vom |
| Baseball | cinder_x | 97 | 0.64 | 0.93 | üstung entraînement ▁bâton Ба ▁américa bule ▁Marlins iels ▁slu ▁броса rücken ▁Fel ▁américaine carrera bate fields ▁qualifié ulosa ▁бро ▁defensi ▁Feld Campo ielder ▁Throw ▁поле |
| Uranium | cinder_x | 68 | 0.69 | 0.9 | ▁bombar ▁noy isotope ▁radi ▁ракет ▁keV ▁Nuk Bq ▁бом Plasma ▁químicos ▁NU Atom ▁Bundesrepublik ▁Plut ▁Toxic atom ▁nú Kern radiation ▁ядер ▁Neutron ▁nuk ▁rad ▁Fukushima |
| Harry Potter | ember_x | 61 | 0.61 | 0.9 | ▁SHIELD ▁Nintendo AFTA ▁Wii ▁Unido ▁Hermes ▁IMAX cosse ▁Hitler uldron britanni ▁Fulham LEGO prüche Решение ▁Prada ▁Tolkien ▁Lucius ▁Agatha ▁hog тельность limard тельству genheit ▁Macbeth |
| Harry Potter | cinder_x | 54 | 0.52 | 0.87 | ▁potion ▁schlecht èvement ▁SHIELD ▁infantiles Sirius ▁Tolkien ▁Neville Гар ▁Lucius bis ▁abgesch ▁Mord ▁Pot Merlin tait trange ▁importe ▁Гар Sever ▁Hamilton mander ahren ▁gry ▁Charlie |
| Gambling | ember_x | 74 | 0.7 | 0.89 | ▁apost érance ▁Gam ▁Pech няются ▁Drogen ▁hasard ▁pari ▁Blackjack ▁arries ▁addict Bingo ▁numer lotto ▁Sorte ▁Bet β ▁рей ▁compulsive ▁prostitu зино ▁Alcohol ▁apuesta ▁acert ▁Jackpot |
| Gambling | cinder_x | 83 | 0.72 | 0.88 | ▁apost ▁Bingo ▁azar ▁cartas ▁Casinos ▁Aces ▁Card ▁carta poker ▁disk ▁Ris ▁Casanova ▁Кар ▁banking зино ▁cartes ▁Alcohol ▁Wynn ▁алкого ▁Gam ▁courir ▁apuesta ▁cartón ▁acert ▁Gerichts |
| Pornography | ember_x | 100 | 0.83 | 0.87 | ▁Pros ▁desn por Rape ▁gore ▁segreg ▁Nun ▁prud ▁DSM ▁Пор ▁обна ▁nack thalam ▁lust ▁stri ▁wies ▁libido ▁dors ▁cens ▁narcis ▁kios ▁Poverty ▁Punk ▁Emo crites |
| Pornography | cinder_x | 98 | 0.73 | 0.88 | queer ▁Pros ornos ▁ВИ ▁desn por Rape ▁Vag ▁Studios ▁homo ▁обна ▁nack ▁lust ▁stri ▁VIH ▁libido ▁Uro ▁нарко sexo ▁sensual ▁гей ▁Poverty ▁dick Inhalte ▁Satan |
| Valentine's Day | cinder_x | 94 | 0.8 | 0.94 | ▁Patricio ▁amoureux ▁printemps ▁Februar ▁remonte ▁шокола hearts ▁дата ▁novio ▁matrimon ▁roman ▁chocol ▁Bunny ▁Kalender ▁parejas ▁Roses ▁romain ▁Venus date Val ▁Вален ▁мар ▁sexe ▁серде ▁розо |
| World War II | ember_x | 65 | 0.75 | 0.88 | Stuttgart Fü Stalin ▁Gerd ailles ▁Casablanca ▁Hamburg ▁nazi ▁Lourdes ▁Cádiz ▁Loire ▁Alemania Wilhelm ▁Alem Muss ▁Bundesrepublik ▁Fas ▁Waffen ▁Auschwitz ▁Moscou Hiros ▁fas Winston ▁Bremen ▁Juden |
| Heroin | cinder_x | 101 | 0.77 | 0.86 | ▁cocaína mari ▁dosis ▁prostitu ▁Sugar ▁Medik ▁therapeu ▁depres Chem ▁pharma ▁Medicine ▁Rezept zepam ▁апте ▁poudre ▁opio stasy ▁prostitution ▁алкого ▁médicaments ▁antidote ▁pharmacie ▁Oxy ▁marijuana toxic |
| Halloween | cinder_x | 102 | 0.75 | 0.92 | ▁cem ▁Ske ▁fant ▁danger ▁disfraz handelt ▁magie ▁Scream ▁Ghosts ▁amenaza ▁ater ▁Trick owohl ▁diablo ▁опас cky ▁Grave брь weihnachts ▁Weihnachts ween ▁infierno ▁grima Kost ▁октя |
| Artificial intelligence | ember_x | 110 | 0.78 | 0.94 | ртуаль alg ▁automati ▁Approximation synthe ▁réelle uliert ▁Cyber ▁Simulations ▁künst ▁bot ogener ▁robo ▁Sim ▁unfair ▁alg ▁Alg ▁Automat ▁реаль вую ▁naturelle ▁Rob ▁numérique ▁simula virtu |
| Artificial intelligence | cinder_x | 78 | 0.74 | 0.94 | ▁Simulations ▁künst ▁bot ▁robo ▁Sim usstsein ▁alg ▁Alg ▁Automat ▁naturelle ▁Rob ▁simula ▁automático ▁simulations assistant ▁virtuelle ▁Frankenstein ▁automat ▁роботи ▁Robot ipedia Rob fak IoT ▁rob |
| COVID-19 pandemic | ember_x | 96 | 0.71 | 0.91 | ▁crédit ▁confinement ▁PCR ▁aéro ▁виру ▁cuarentena ▁mascarilla ▁ACE ▁distanci нави ▁masque ▁online ▁contagios ▁équilibr ▁sanitaire inf ▁Aufgrund ▁VIH ▁mascar ▁verticale ▁Krise ▁contagio ▁quar ▁Viro ▁pandemi |
| COVID-19 pandemic | cinder_x | 95 | 0.77 | 0.94 | ▁PCR ▁виру ▁cuarentena fektions ▁mascarilla ▁vacuna ▁distanci ▁зараз нави ▁vacunación ▁masque ▁vir ▁contagios ▁sanitaire ▁VIH ▁mascar ▁Krise ▁contagio ▁Infe ▁Epidemio ▁Viro ▁pandemi infe ▁viro ▁máscaras |

