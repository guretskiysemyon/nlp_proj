# X10: does extrapolation need the features? (EMBER, 6 concepts, train questions, coverage only)

EMBER = original English edit; X = EMBER+X as built (x5b, tested in X9); X_foreign = only its added tokens that never occur in the English sentences of any of the 18 concepts; L = same number of tokens from the same concept-only pool, ranked by the number of concept sentences that contain them (no features); L_word = L with whole-word tokens only; L_foreign = L_word from the foreign-only pool, sized as X_foreign. qa_L / simdom_L = share of QA_train / SimdomQA_train questions in L with an edited token in the question or correct answer; bg_L = share of the other 17 concepts' native sentences in L with an edited token. Higher qa = more questions the edit can erase (X8); higher simdom / bg = more collateral. Coverage proxies only; no erasure is run.

## Mean over the 6 concepts

| variant | n_edit | qa_en | qa_foreign | simdom_en | simdom_foreign | bg_en | bg_foreign | qa_de | qa_es | qa_fr | qa_ru |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EMBER | 50.0 | 0.78 | 0.29 | 0.1 | 0.023 | 0.008 | 0.004 | 0.363 | 0.383 | 0.383 | 0.03 |
| X | 146.333 | 0.79 | 0.607 | 0.123 | 0.101 | 0.013 | 0.021 | 0.637 | 0.65 | 0.797 | 0.343 |
| X_foreign | 136.333 | 0.79 | 0.587 | 0.107 | 0.082 | 0.008 | 0.018 | 0.6 | 0.63 | 0.773 | 0.343 |
| L | 146.333 | 0.78 | 0.772 | 0.11 | 0.214 | 0.013 | 0.105 | 0.817 | 0.74 | 0.873 | 0.657 |
| L_word | 146.333 | 0.78 | 0.686 | 0.11 | 0.222 | 0.013 | 0.099 | 0.773 | 0.693 | 0.763 | 0.513 |
| L_foreign | 136.333 | 0.78 | 0.662 | 0.1 | 0.196 | 0.008 | 0.078 | 0.733 | 0.657 | 0.75 | 0.507 |

## Sizes and overlap of the added sets

`X_vs_L_overlap`: share of X's added tokens that the frequency rule also picks.

| concept | n_E0 | n_pool | n_pool_foreign | n_added_X | n_added_X_foreign | X_vs_L_overlap | X_vs_Lword_overlap | Xf_vs_Lf_overlap |
|---|---|---|---|---|---|---|---|---|
| Cannabis | 45 | 6640 | 6131 | 75 | 65 | 0.04 | 0.05 | 0.05 |
| Gun | 61 | 5973 | 5496 | 130 | 112 | 0.16 | 0.21 | 0.21 |
| Nazism | 41 | 6895 | 6399 | 81 | 74 | 0.11 | 0.09 | 0.09 |
| Republic of Ireland | 31 | 5877 | 5405 | 57 | 52 | 0.11 | 0.12 | 0.12 |
| Culture of Greece | 52 | 6725 | 6193 | 99 | 90 | 0.2 | 0.21 | 0.23 |
| Golf | 70 | 5517 | 5112 | 136 | 125 | 0.1 | 0.09 | 0.07 |

## Per concept

| concept | variant | n_edit | qa_en | qa_de | qa_es | qa_fr | qa_ru | simdom_en | simdom_foreign | bg_en | bg_foreign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cannabis | EMBER | 45 | 0.98 | 0.76 | 0.88 | 0.88 | 0.08 | 0.0 | 0.005 | 0.0024 | 0.0027 |
| Cannabis | X | 120 | 0.98 | 0.78 | 0.9 | 0.88 | 0.54 | 0.0 | 0.04 | 0.0076 | 0.0139 |
| Cannabis | X_foreign | 110 | 0.98 | 0.78 | 0.9 | 0.88 | 0.54 | 0.0 | 0.01 | 0.0025 | 0.0102 |
| Cannabis | L | 120 | 0.98 | 0.82 | 0.96 | 0.98 | 0.88 | 0.04 | 0.24 | 0.0082 | 0.0786 |
| Cannabis | L_word | 120 | 0.98 | 0.78 | 0.94 | 0.98 | 0.68 | 0.04 | 0.255 | 0.0055 | 0.0794 |
| Cannabis | L_foreign | 110 | 0.98 | 0.78 | 0.94 | 0.98 | 0.64 | 0.0 | 0.195 | 0.0027 | 0.06745 |
| Gun | EMBER | 61 | 0.74 | 0.12 | 0.1 | 0.1 | 0.04 | 0.28 | 0.015 | 0.0106 | 0.00115 |
| Gun | X | 191 | 0.78 | 0.28 | 0.66 | 0.6 | 0.5 | 0.34 | 0.19 | 0.0225 | 0.029925000000000004 |
| Gun | X_foreign | 173 | 0.78 | 0.28 | 0.66 | 0.46 | 0.5 | 0.28 | 0.15000000000000002 | 0.0108 | 0.023249999999999996 |
| Gun | L | 191 | 0.74 | 0.56 | 0.76 | 0.68 | 0.84 | 0.28 | 0.25 | 0.0133 | 0.11215 |
| Gun | L_word | 191 | 0.74 | 0.4 | 0.62 | 0.8 | 0.78 | 0.28 | 0.32499999999999996 | 0.0145 | 0.1118 |
| Gun | L_foreign | 173 | 0.74 | 0.36 | 0.54 | 0.72 | 0.78 | 0.28 | 0.315 | 0.0106 | 0.08234999999999999 |
| Nazism | EMBER | 41 | 0.8 | 0.64 | 0.42 | 0.42 | 0.02 | 0.16 | 0.09500000000000001 | 0.0094 | 0.01055 |
| Nazism | X | 122 | 0.82 | 0.7 | 0.7 | 0.72 | 0.08 | 0.16 | 0.20500000000000002 | 0.0108 | 0.02915 |
| Nazism | X_foreign | 115 | 0.82 | 0.7 | 0.7 | 0.72 | 0.08 | 0.16 | 0.185 | 0.0094 | 0.025525 |
| Nazism | L | 122 | 0.8 | 0.78 | 0.82 | 0.78 | 0.46 | 0.16 | 0.175 | 0.0102 | 0.09787499999999999 |
| Nazism | L_word | 122 | 0.8 | 0.72 | 0.74 | 0.76 | 0.14 | 0.16 | 0.195 | 0.0196 | 0.079925 |
| Nazism | L_foreign | 115 | 0.8 | 0.72 | 0.74 | 0.76 | 0.14 | 0.16 | 0.195 | 0.0094 | 0.06725 |
| Republic of Ireland | EMBER | 31 | 0.32 | 0.12 | 0.04 | 0.12 | 0.0 | 0.04 | 0.0 | 0.0098 | 0.001325 |
| Republic of Ireland | X | 88 | 0.32 | 0.3 | 0.12 | 0.82 | 0.0 | 0.06 | 0.04 | 0.0122 | 0.010775 |
| Republic of Ireland | X_foreign | 83 | 0.32 | 0.3 | 0.04 | 0.82 | 0.0 | 0.04 | 0.035 | 0.0098 | 0.010525 |
| Republic of Ireland | L | 88 | 0.32 | 0.82 | 0.22 | 0.9 | 0.18 | 0.04 | 0.1 | 0.0114 | 0.07852500000000001 |
| Republic of Ireland | L_word | 88 | 0.32 | 0.82 | 0.18 | 0.18 | 0.14 | 0.04 | 0.09 | 0.011 | 0.074825 |
| Republic of Ireland | L_foreign | 83 | 0.32 | 0.78 | 0.08 | 0.18 | 0.14 | 0.04 | 0.09 | 0.0098 | 0.060050000000000006 |
| Culture of Greece | EMBER | 52 | 1.0 | 0.12 | 0.08 | 0.06 | 0.0 | 0.08 | 0.015 | 0.0076 | 0.001 |
| Culture of Greece | X | 151 | 1.0 | 0.92 | 0.68 | 0.96 | 0.76 | 0.12 | 0.07 | 0.0108 | 0.01715 |
| Culture of Greece | X_foreign | 142 | 1.0 | 0.92 | 0.68 | 0.96 | 0.76 | 0.1 | 0.07 | 0.0076 | 0.01655 |
| Culture of Greece | L | 151 | 1.0 | 0.96 | 0.74 | 0.98 | 0.88 | 0.08 | 0.225 | 0.0196 | 0.11595 |
| Culture of Greece | L_word | 151 | 1.0 | 0.94 | 0.74 | 0.94 | 0.84 | 0.08 | 0.19 | 0.0124 | 0.11065 |
| Culture of Greece | L_foreign | 142 | 1.0 | 0.94 | 0.74 | 0.94 | 0.84 | 0.08 | 0.16 | 0.0076 | 0.077725 |
| Golf | EMBER | 70 | 0.84 | 0.42 | 0.78 | 0.72 | 0.04 | 0.04 | 0.01 | 0.0094 | 0.004425 |
| Golf | X | 206 | 0.84 | 0.84 | 0.84 | 0.8 | 0.18 | 0.06 | 0.06 | 0.0151 | 0.024524999999999998 |
| Golf | X_foreign | 195 | 0.84 | 0.62 | 0.8 | 0.8 | 0.18 | 0.06 | 0.045 | 0.0094 | 0.0229 |
| Golf | L | 206 | 0.84 | 0.96 | 0.94 | 0.92 | 0.7 | 0.06 | 0.29500000000000004 | 0.0163 | 0.1477 |
| Golf | L_word | 206 | 0.84 | 0.98 | 0.94 | 0.92 | 0.5 | 0.06 | 0.27499999999999997 | 0.0141 | 0.135325 |
| Golf | L_foreign | 195 | 0.84 | 0.82 | 0.9 | 0.92 | 0.5 | 0.04 | 0.21999999999999997 | 0.0094 | 0.11032499999999999 |

## Added tokens (most concept sentences first; * = foreign-only)

- Cannabis / X (75): ▁кан* ▁Canna* ▁neur ▁aceite* izophren* ▁Hemp* ▁endoc* ▁delta* ▁hierba* ▁Endoc* ▁pot ▁Delta nabis* ▁epile* ▁CBN* cannabis* ▁Pot* ▁mg ativas* ▁weed* ▁hallucin* ▁Ree* ▁neurop* ▁Narco mg ▁verde* MG* ▁covid* ▁Hip* ▁Gan*
- Cannabis / X_foreign (65): ▁кан* ▁Canna* ▁aceite* izophren* ▁Hemp* ▁endoc* ▁delta* ▁hierba* ▁Endoc* nabis* ▁epile* ▁CBN* cannabis* ▁Pot* ativas* ▁weed* ▁hallucin* ▁Ree* ▁neurop* ▁verde* MG* ▁covid* ▁Hip* ▁Gan* ▁Zel* ▁testo* ▁Omega* ▁vape* ▁cán* tetra*
- Cannabis / L (75): пли* ▁consommation* ▁употреб* ▁кан* isk ▁Kons* ▁выращи* amo* ▁neuro umenten* ▁dro* ▁flores* ▁нарко* ▁Canna* Г* vre* ▁chan* ▁consom* ▁tabac* ▁existe* ▁planta* ▁semillas* ак* вание* ▁hojas* ▁efectos* izop* ана* пле* ▁fumée*
- Cannabis / L_word (75): ▁consommation* ▁употреб* ▁кан* ▁Kons* ▁выращи* ▁neuro ▁dro* ▁flores* ▁нарко* ▁Canna* ▁chan* ▁consom* ▁tabac* ▁existe* ▁planta* ▁semillas* ▁hojas* ▁efectos* ▁fumée* ▁Anbau* ▁neur ▁Wirkung* ▁médical* ▁cas ▁étude* ▁vielen* ▁cogni* ▁inhal ▁psico* ▁augmentation
- Cannabis / L_foreign (65): ▁consommation* ▁употреб* ▁кан* ▁Kons* ▁выращи* ▁dro* ▁flores* ▁нарко* ▁Canna* ▁chan* ▁consom* ▁tabac* ▁existe* ▁planta* ▁semillas* ▁hojas* ▁efectos* ▁fumée* ▁Anbau* ▁Wirkung* ▁médical* ▁étude* ▁vielen* ▁cogni* ▁psico* ▁распростра* ▁Besitz* ▁associée* ▁Drogen* ▁signific*
- Gun / X (130): ▁armes* ▁fusil* ▁стре* ▁оружия* ▁оружие* ▁ство* ▁огне* ▁dispar* ▁munitions ▁ору* ▁muni* ▁disparo* ▁canon ▁pist* ▁chasse* ▁Schie* ▁pistola* ▁Waff* ▁pistolet* ▁охо* ▁proyec* ▁balle* ▁AK* ützen* ▁tiro* ▁Pist* ▁caza* ▁grenades* ▁disparos* ▁balles
- Gun / X_foreign (112): ▁armes* ▁fusil* ▁стре* ▁оружия* ▁оружие* ▁ство* ▁огне* ▁dispar* ▁ору* ▁muni* ▁disparo* ▁pist* ▁chasse* ▁Schie* ▁pistola* ▁Waff* ▁pistolet* ▁охо* ▁proyec* ▁balle* ▁AK* ützen* ▁tiro* ▁Pist* ▁caza* ▁grenades* ▁disparos* ▁balas* ▁muertes* ▁tirs*
- Gun / L (130): ▁Feuer* ▁armes* ▁fusil* ▁fuego* ▁feu* ▁fus* ▁стре* ▁пу* iles* ▁оружия* ▁оружие* ▁ство* ▁огне* ▁calibre* tral* tom ▁sem iau* ▁кали* ▁dispar* ▁ame* ▁ca* ▁munitions ▁ору* ▁muni* ▁disparo* ▁asalto* рез* ряд* ▁Der
- Gun / L_word (130): ▁Feuer* ▁armes* ▁fusil* ▁fuego* ▁feu* ▁fus* ▁стре* ▁оружия* ▁оружие* ▁ство* ▁огне* ▁calibre* ▁sem ▁кали* ▁dispar* ▁ame* ▁munitions ▁ору* ▁muni* ▁disparo* ▁asalto* ▁Der ▁canon ▁автомати* ▁pist* ▁vitesse* ▁chasse* ▁воору* ▁cer ▁infan*
- Gun / L_foreign (112): ▁Feuer* ▁armes* ▁fusil* ▁fuego* ▁feu* ▁fus* ▁стре* ▁оружия* ▁оружие* ▁ство* ▁огне* ▁calibre* ▁кали* ▁dispar* ▁ame* ▁ору* ▁muni* ▁disparo* ▁asalto* ▁автомати* ▁pist* ▁vitesse* ▁chasse* ▁воору* ▁infan* ▁automático* ▁Lauf* ▁SCAR* ▁guerre* ▁сна*
- Nazism / X (81): ▁nazis* ▁nazi* ▁Alemania* ▁fas* ▁alemanes* hrer* Allemagne* мец* ▁alemana* ▁Allemagne* ▁german* ▁Alem* ▁фа* deutsche* Fü* ▁Fritz german* ▁Герма* ▁Nietzsche* ▁Kaiser ▁Wilhelm* ▁Wolfgang* фю* ▁Германия* fas* ▁Herr* ▁DDR* Mein* ▁Versailles ▁Bernhard*
- Nazism / X_foreign (74): ▁nazis* ▁nazi* ▁Alemania* ▁fas* ▁alemanes* hrer* Allemagne* мец* ▁alemana* ▁Allemagne* ▁german* ▁Alem* ▁фа* deutsche* Fü* german* ▁Герма* ▁Nietzsche* ▁Wilhelm* ▁Wolfgang* фю* ▁Германия* fas* ▁Herr* ▁DDR* Mein* ▁Bernhard* Deutsche* ▁Erich* ▁Deutschlands*
- Nazism / L (81): ismus* ▁nazis* ▁nazi* ismo* “.* ▁NS* зен* ▁Alemania* ▁idé* ▁Ide* ▁fas* ▁thé* ▁нем* циали* ▁Juifs* ▁judíos* ▁sei* ▁анти* ▁утвер* ▁régimen* лера* ДА* жда* iker* ологи* ▁ми* ciale* ▁alemanes* ▁“* ▁их*
- Nazism / L_word (81): ▁nazis* ▁nazi* ▁Alemania* ▁idé* ▁Ide* ▁fas* ▁thé* ▁нем* ▁Juifs* ▁judíos* ▁sei* ▁анти* ▁утвер* ▁régimen* ▁alemanes* ▁народ* ▁peuple* ▁ary* ▁marx* ▁gobierno* ▁рей* ▁Juden* ▁jud* ▁peu* ▁Das ▁rass* ▁народа* ▁oppos* ▁идеи* ▁учё*
- Nazism / L_foreign (74): ▁nazis* ▁nazi* ▁Alemania* ▁idé* ▁Ide* ▁fas* ▁thé* ▁нем* ▁Juifs* ▁judíos* ▁sei* ▁анти* ▁утвер* ▁régimen* ▁alemanes* ▁народ* ▁peuple* ▁ary* ▁marx* ▁gobierno* ▁рей* ▁Juden* ▁jud* ▁peu* ▁rass* ▁народа* ▁oppos* ▁идеи* ▁учё* ▁Kershaw*
- Republic of Ireland / X (57): Irlande* ▁iris* ▁Dub ▁Leinster* ▁Iris* ▁brit* nacht* ▁Waterford* ▁Munster* Bretagne* ▁Clare ▁британ* irse* ▁Erin* ▁Derry* ▁Yeats* ▁GAA* ▁Niall* ▁Brexit* ▁Arma* ▁Seam* cosse* ▁Oma* ▁lagos* ▁celta* ▁Tipper* ▁Wexford* ▁Kilkenny* ▁Hiber* ▁Sligo*
- Republic of Ireland / X_foreign (52): Irlande* ▁iris* ▁Leinster* ▁Iris* ▁brit* nacht* ▁Waterford* ▁Munster* Bretagne* ▁британ* irse* ▁Erin* ▁Derry* ▁Yeats* ▁GAA* ▁Niall* ▁Brexit* ▁Arma* ▁Seam* cosse* ▁Oma* ▁lagos* ▁celta* ▁Tipper* ▁Wexford* ▁Kilkenny* ▁Hiber* ▁Sligo* Dá* ▁dub*
- Republic of Ireland / L (57): Irlande* ▁iris* esa* landais* lande* ▁Der ▁страны* ▁Dub ken ▁PIB* вые* ▁Ду* ▁Einwohner* дию* ▁taux* ▁Republik* landaise* ▁betrug* ▁cel ▁ga* ▁roi* ▁Reino* ▁économique* ▁Unido* ▁Insel* ▁Königreich* ▁Великобрита* ▁Leinster* рен* ▁cond
- Republic of Ireland / L_word (57): ▁iris* ▁Der ▁страны* ▁Dub ▁PIB* ▁Einwohner* ▁taux* ▁Republik* ▁betrug* ▁cel ▁roi* ▁Reino* ▁économique* ▁Unido* ▁Insel* ▁Königreich* ▁Великобрита* ▁Leinster* ▁cond ▁Iris* ▁economía* ▁européenne* ▁результате* ▁стране* ▁основном* ▁годов* ▁Motor ▁brit* ▁inglés* ▁allem*
- Republic of Ireland / L_foreign (52): ▁iris* ▁страны* ▁PIB* ▁Einwohner* ▁taux* ▁Republik* ▁betrug* ▁roi* ▁Reino* ▁économique* ▁Unido* ▁Insel* ▁Königreich* ▁Великобрита* ▁Leinster* ▁Iris* ▁economía* ▁européenne* ▁результате* ▁стране* ▁основном* ▁годов* ▁brit* ▁inglés* ▁allem* ▁экономи* ▁Prozent* ▁wirtschaft* ▁ang* ▁gobierno*
- Culture of Greece / X (99): ▁гре* ▁Гре* ▁griechischen* ▁griech* ▁Griechen* ▁grecque* ▁grec* ▁griega* ▁Grèce* ▁griego* ▁Griechenland* ▁gre* ▁Grecs* ▁hellen* ▁byz* ▁Atenas* ▁Athen* ▁Macedonia* ▁Aristote* ▁Маке* ▁römischen* grie* λλην* ▁Odys* ▁Platon* ▁Romains* ▁Medit* ▁Olymp* ▁Egipto* ▁Acha*
- Culture of Greece / X_foreign (90): ▁гре* ▁Гре* ▁griechischen* ▁griech* ▁Griechen* ▁grecque* ▁grec* ▁griega* ▁Grèce* ▁griego* ▁Griechenland* ▁gre* ▁Grecs* ▁hellen* ▁byz* ▁Atenas* ▁Athen* ▁Macedonia* ▁Aristote* ▁Маке* ▁römischen* grie* λλην* ▁Odys* ▁Platon* ▁Romains* ▁Medit* ▁Olymp* ▁Egipto* ▁Acha*
- Culture of Greece / L (99): ▁гре* ▁Гре* ▁griechischen* ▁griech* ▁Griechen* ▁grecque* ▁Chr* ▁grec* ▁griega* ▁Grèce* ken ▁griego* ▁Griechenland* ▁hell ▁gre* ▁ми* ismus* фи* ▁cultura* ènes* нисти* ▁Grecs* ческом* ▁hellen* gri* ▁biz* antin* ▁Alten* .- ▁ци*
- Culture of Greece / L_word (99): ▁гре* ▁Гре* ▁griechischen* ▁griech* ▁Griechen* ▁grecque* ▁Chr* ▁grec* ▁griega* ▁Grèce* ▁griego* ▁Griechenland* ▁hell ▁gre* ▁cultura* ▁Grecs* ▁hellen* ▁biz* ▁Alten* ▁byz* ▁Anti ▁языке* ▁Einfluss* ▁hel* ▁Der ▁langue* ▁arquitectura* ▁kulture* ▁поэ* ▁archa
- Culture of Greece / L_foreign (90): ▁гре* ▁Гре* ▁griechischen* ▁griech* ▁Griechen* ▁grecque* ▁Chr* ▁grec* ▁griega* ▁Grèce* ▁griego* ▁Griechenland* ▁gre* ▁cultura* ▁Grecs* ▁hellen* ▁biz* ▁Alten* ▁byz* ▁языке* ▁Einfluss* ▁hel* ▁langue* ▁arquitectura* ▁kulture* ▁поэ* ▁Atenas* ▁allem* ▁histori* ▁Athen*
- Golf / X (136): льф* ▁Gol ▁parcours* ▁golpes* голь* ▁golfe* ▁Schlä* ▁Loch* ▁trous* clubs* ▁Resort ▁Swing* ▁Ryder Loch* ▁sacs* ▁obstáculos* ▁penal ▁Pine* Gol* ▁штра* ▁чу* ▁Cad mud* ▁Loft* ▁chaînes* chränkungen* ▁Bir* ▁obst* ▁Tee ▁Hole*
- Golf / X_foreign (125): льф* ▁parcours* ▁golpes* голь* ▁golfe* ▁Schlä* ▁Loch* ▁trous* clubs* ▁Swing* Loch* ▁sacs* ▁obstáculos* ▁Pine* Gol* ▁штра* ▁чу* mud* ▁Loft* ▁chaînes* chränkungen* ▁Bir* ▁obst* ▁Hole* cours* ▁polos* ▁Penalty* siert* elson* ▁Haskell*
- Golf / L (136): льф* ▁Gol ▁bola* ▁parcours* ▁balle* ▁golpes* голь* ▁Der ▁лу* ▁площа* ▁мини* ▁golfe* ▁jugadores* ▁campos* ▁клю* nois* ger фи* ▁Ball ▁inglés* ▁Spieler* луб* ▁joueur* ▁поле* chlag* fsch* ▁удар* клу* ▁Schlä* ▁pal
- Golf / L_word (136): ▁Gol ▁bola* ▁parcours* ▁balle* ▁golpes* ▁Der ▁площа* ▁мини* ▁golfe* ▁jugadores* ▁campos* ▁клю* ▁Ball ▁inglés* ▁Spieler* ▁joueur* ▁поле* ▁удар* ▁Schlä* ▁pal ▁existe* ▁distancia* ▁spielen* ▁palo* ▁Loch* ▁спорта* ▁Green ▁peu* ▁resultado* ▁coups*
- Golf / L_foreign (125): ▁bola* ▁parcours* ▁balle* ▁golpes* ▁площа* ▁мини* ▁golfe* ▁jugadores* ▁campos* ▁клю* ▁inglés* ▁Spieler* ▁joueur* ▁поле* ▁удар* ▁Schlä* ▁existe* ▁distancia* ▁spielen* ▁palo* ▁Loch* ▁спорта* ▁peu* ▁resultado* ▁coups* ▁dificultad* ▁trous* ▁salida* ▁heute* ▁международ*
