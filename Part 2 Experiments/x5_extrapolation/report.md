# X5: extrapolating the English features to native foreign tokens (no erasure run)

Z fixed, coefficients solved for English + native L tokens, each method's sparsity rule. `q_cov_L`: share of L's 50 QA_test questions containing an edited token, before → after. `kept_original_%`: share of the original English edit set still edited.

| concept | method | language | n_edit_before | n_edit_after | n_added | n_added_not_in_english | kept_original_% | q_cov_en_before | q_cov_en_after | q_cov_L_before | q_cov_L_after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cannabis | EMBER | de | 45 | 79 | 35 | 26 | 98 | 0.98 | 0.98 | 0.86 | 0.9 |
| Cannabis | EMBER | es | 45 | 70 | 26 | 19 | 98 | 0.98 | 0.98 | 0.9 | 0.9 |
| Cannabis | EMBER | fr | 45 | 68 | 24 | 15 | 98 | 0.98 | 0.98 | 0.9 | 0.9 |
| Cannabis | EMBER | ru | 45 | 67 | 22 | 5 | 100 | 0.98 | 0.98 | 0.08 | 0.86 |
| Cannabis | CINDER | de | 34 | 51 | 18 | 7 | 97 | 0.98 | 0.98 | 0.86 | 0.88 |
| Cannabis | CINDER | es | 34 | 52 | 19 | 9 | 97 | 0.98 | 0.98 | 0.9 | 0.9 |
| Cannabis | CINDER | fr | 34 | 51 | 18 | 8 | 97 | 0.98 | 0.98 | 0.9 | 0.9 |
| Cannabis | CINDER | ru | 34 | 46 | 12 | 1 | 100 | 0.98 | 0.98 | 0.08 | 0.84 |
| Gun | EMBER | de | 61 | 121 | 60 | 29 | 100 | 0.82 | 0.86 | 0.04 | 0.32 |
| Gun | EMBER | es | 61 | 119 | 58 | 32 | 100 | 0.82 | 0.86 | 0.08 | 0.56 |
| Gun | EMBER | fr | 61 | 116 | 55 | 37 | 100 | 0.82 | 0.82 | 0.12 | 0.6 |
| Gun | EMBER | ru | 61 | 113 | 52 | 16 | 100 | 0.82 | 0.88 | 0.0 | 0.54 |
| Gun | CINDER | de | 67 | 109 | 42 | 20 | 100 | 0.88 | 0.9 | 0.04 | 0.3 |
| Gun | CINDER | es | 67 | 119 | 52 | 30 | 100 | 0.88 | 0.92 | 0.08 | 0.68 |
| Gun | CINDER | fr | 67 | 115 | 48 | 27 | 100 | 0.88 | 0.9 | 0.12 | 0.6 |
| Gun | CINDER | ru | 67 | 106 | 39 | 14 | 100 | 0.88 | 0.9 | 0.0 | 0.52 |
| Nazism | EMBER | de | 41 | 60 | 24 | 21 | 88 | 0.8 | 0.8 | 0.56 | 0.62 |
| Nazism | EMBER | es | 41 | 80 | 39 | 23 | 100 | 0.8 | 0.84 | 0.46 | 0.86 |
| Nazism | EMBER | fr | 41 | 73 | 34 | 24 | 95 | 0.8 | 0.72 | 0.42 | 0.64 |
| Nazism | EMBER | ru | 41 | 74 | 33 | 9 | 100 | 0.8 | 0.84 | 0.0 | 0.1 |
| Republic of Ireland | EMBER | de | 31 | 58 | 27 | 16 | 100 | 0.44 | 0.46 | 0.18 | 0.44 |
| Republic of Ireland | EMBER | es | 31 | 57 | 27 | 17 | 97 | 0.44 | 0.46 | 0.1 | 0.18 |
| Republic of Ireland | EMBER | fr | 31 | 52 | 22 | 16 | 97 | 0.44 | 0.42 | 0.18 | 0.74 |
| Republic of Ireland | EMBER | ru | 31 | 56 | 25 | 4 | 100 | 0.44 | 0.46 | 0.0 | 0.0 |
| Culture of Greece | EMBER | de | 52 | 100 | 48 | 29 | 100 | 0.96 | 0.98 | 0.18 | 0.84 |
| Culture of Greece | EMBER | es | 52 | 91 | 39 | 20 | 100 | 0.96 | 0.98 | 0.04 | 0.56 |
| Culture of Greece | EMBER | fr | 52 | 93 | 41 | 24 | 100 | 0.96 | 0.98 | 0.08 | 0.82 |
| Culture of Greece | EMBER | ru | 52 | 93 | 41 | 5 | 100 | 0.96 | 0.96 | 0.0 | 0.52 |
| Culture of Greece | CINDER | de | 106 | 168 | 66 | 40 | 96 | 1.0 | 1.0 | 0.22 | 0.84 |
| Culture of Greece | CINDER | es | 106 | 159 | 54 | 29 | 99 | 1.0 | 1.0 | 0.06 | 0.6 |
| Culture of Greece | CINDER | fr | 106 | 174 | 69 | 45 | 99 | 1.0 | 1.0 | 0.12 | 0.94 |
| Culture of Greece | CINDER | ru | 106 | 149 | 44 | 8 | 99 | 1.0 | 1.0 | 0.0 | 0.52 |
| Golf | EMBER | de | 70 | 120 | 50 | 34 | 100 | 0.9 | 0.9 | 0.58 | 0.8 |
| Golf | EMBER | es | 70 | 111 | 42 | 23 | 99 | 0.9 | 0.9 | 0.74 | 0.86 |
| Golf | EMBER | fr | 70 | 110 | 41 | 28 | 99 | 0.9 | 0.9 | 0.74 | 0.78 |
| Golf | EMBER | ru | 70 | 122 | 52 | 20 | 100 | 0.9 | 0.9 | 0.04 | 0.24 |
| Golf | CINDER | de | 44 | 78 | 35 | 19 | 98 | 0.88 | 0.88 | 0.56 | 0.82 |
| Golf | CINDER | es | 44 | 76 | 32 | 14 | 100 | 0.88 | 0.88 | 0.74 | 0.8 |
| Golf | CINDER | fr | 44 | 74 | 31 | 14 | 98 | 0.88 | 0.88 | 0.7 | 0.74 |
| Golf | CINDER | ru | 44 | 80 | 36 | 11 | 100 | 0.88 | 0.88 | 0.02 | 0.2 |

## Added tokens (largest coefficient first, foreign-only marked *)

- Cannabis / EMBER / de: ▁Canna* cannabis* ▁Hemp* ▁weed* ▁Endoc* ▁Kush* CBC* ▁Bong* ioid* ▁Vapor* Vapor* ▁Stoner* ▁pot* ▁fruit ▁Reggae* ▁Gan* ▁Pot* ▁Epile* ▁Coffe* ▁legalized ▁Christi ▁tetra ▁Cant ▁drink MG*
- Cannabis / EMBER / es: ▁Canna* cannabis* ▁Hemp* ▁weed* nabis* ▁CBN* ▁endoc* ativas* madol* ▁fruit ▁cán* ▁legalized ▁epile* ▁delta* zcan* ▁Christi ▁tetra ▁drink ▁nab ▁Ree* ▁Narco* arbox ▁Omega* ▁hierba* ▁Farma*
- Cannabis / EMBER / fr: nabis* ▁vape* ▁endoc* ▁edibles* EEG* ▁pot* ▁fruit Huile* ▁legalized ▁marie* ▁delta* ▁Christi ▁tetra ▁covid* ▁Cant ▁drink ▁nab ▁fuma* vapor* ▁Delta* arbox illoma* ▁HPV* ▁femin
- Cannabis / EMBER / ru: ▁кан* ▁BDS* ▁fruit ▁legalized ▁Christi ▁tetra ▁Cant ▁drink ▁кана* ▁nab arbox ▁mar* ▁femin ▁TH ▁botan ▁Cand ▁anx ▁strain ▁Δ nab грамм* ▁vapor
- Cannabis / CINDER / de: ▁Canna* cannabis* ▁Hemp* ▁weed* ▁psychedelic Hash ▁bong zuführen* ▁κάν ▁Endoc* Tetra ▁fruits ▁fruit penes can ▁heroin ▁pot* ▁legalized
- Cannabis / CINDER / es: ▁Canna* cannabis* ▁Hemp* ▁weed* nabis* ▁psychedelic Hash ▁bong ▁endoc* ▁κάν ▁CBN* Tetra ▁Can* ▁fruits ▁fruit penes madol* ▁heroin ▁legalized
- Cannabis / CINDER / fr: ▁vape* nabis* ▁psychedelic Hash ▁bong ▁endoc* ▁κάν «* ▁edibles* Tetra ▁fruits ▁fruit penes anmoins* ▁â* ▁heroin ▁pot* ▁legalized
- Cannabis / CINDER / ru: ▁кан* ▁psychedelic Hash ▁bong ▁κάν Tetra ▁fruits ▁fruit penes can ▁heroin ▁legalized
- Gun / EMBER / de: ▁Guns* ▁gunner* Pist* pist* ▁Pist* ▁Schie* ▁Schützen* ▁schie* choss* volver* ▁Jäger* ▁Tö* ießen* munition* Waff* ▁dies* ▁trumpet ▁selling gon ▁Remington ▁lances ▁Tobacco ▁Harn* phone ▁musk
- Gun / EMBER / es: ▁pistola* ▁rifles* ▁disparos* ▁disparo* ▁Weapons* ▁tiros* ▁Weapon* pist* ▁fusil* ▁Pist* ▁tiro* ▁matar* ▁pist* ▁muerta* ▁caza* ▁Ruger* ▁homicidio* ▁balas* ▁caz* ▁muertes* ▁proyec* ▁ATF* ▁armados* ▁arsenal* ▁dispara*
- Gun / EMBER / fr: ▁pistolet* ▁tournage* ▁fusil* ▁tirs* ▁massacres* ▁chasse* ▁tuer* ▁pist* ▁homic* ▁armes* istolet* ▁grenades* ▁Ruger* ▁grenade* ▁munitions* ▁couteau* ▁suicide* ▁Canon* jection* ▁balles* ▁canon* ▁Grenade* ▁sabre* ▁coute* ▁missiles*
- Gun / EMBER / ru: ▁Kill* ▁pistolet* ▁стре* ▁оружие* Стре* ▁оружия* ▁ору* ▁pist* ▁охо* шое* ▁trumpet ▁selling gon ▁Remington ▁lances ▁смерти* ▁Tobacco phone ▁musk ▁eject ▁attacked ▁огне* ▁paintball ▁revol ▁breech
- Gun / CINDER / de: ▁Guns* pist* ▁gunner* Pist* ▁Pist* ▁schie* volver* ▁Schützen* ▁Jäger* choss* balls* ▁Schie* ießen* ▁cocked attack* ▁flight munition* stab* Waff* seye ▁arrow ▁Browning ▁ordnance ▁paintball ▁powder
- Gun / CINDER / es: ▁disparo* ▁disparos* ▁rifles* ▁pistola* ▁tiros* ▁Weapon* ▁tiro* pist* ▁Weapons* ▁fusil* ▁Pist* ▁pist* ▁dispar* ▁dispara* ▁ATF* ▁Ruger* pic* ▁proyec* ▁arsenal* ▁cocked ▁flight ▁caz* seye ▁arrow ▁balas*
- Gun / CINDER / fr: ▁pistolet* ▁tournage* ▁tirs* ▁fusil* ▁pist* ▁grenades* ▁grenade* ▁balles* ▁armes* ▁balle* ▁dispar* ▁munitions* ▁canon* ▁Ruger* ▁Canon* istolet* ▁Grenade* ▁missiles* ▁cocked ▁touché* ▁flight ▁couteau* ▁bomb* seye ▁chasse*
- Gun / CINDER / ru: ▁pistolet* ▁оружие* ▁стре* ▁оружия* Стре* ▁pist* ▁ору* ▁сня* ▁cocked ▁flight seye ▁arrow ▁Browning ▁ordnance ▁paintball ▁powder шое* ▁lance ▁Tobacco ▁reloading ▁reload ▁Heck ATF ▁throws ▁lances
- Nazism / EMBER / de: ▁fas* ▁german* fas* german* ▁Vichy* ▁Bayern* ▁reich* Fü* ▁Buchen* ▁kaiser* deutsche* ▁Erich* ▁Wilhelm* ▁Harz* ▁Deutsche* ▁Kaiser* Deutsche* Mein* ▁Versailles* Deutschland* ▁Waffen* ▁Napoleonic ▁Walther ▁Ary
- Nazism / EMBER / es: ▁nazis* ▁nazi* ▁Gesta* ▁fas* ▁naz* ▁alemana* ▁alemanes* ▁Volkswagen* ▁Alemania* Alemania* ▁Berlín* Fü* ▁Konrad* ▁Alem* ▁Wilhelm* ▁Bav* Deutsche* ▁Heinrich ▁plebis* ▁Napoleonic ▁Hermann ▁Walther ▁Ary ▁Hohen hrer*
- Nazism / EMBER / fr: ▁nazis* ▁nazi* ▁Gesta* ▁fas* ▁german* german* naz* Allemagne* ▁fasc* ▁Nietzsche* ▁Germania* ▁Niem* ▁Allemagne* ▁deutsche* deutsche* ▁Napoléon* Erich* ▁Frankfurter* ▁Deutsche* ▁Versailles* ▁Heinrich Niet* ▁Napoleonic ▁Hermann ▁Walther
- Nazism / EMBER / ru: ▁Герма* ▁Германия* мец* ▁Deutsche* Deutsche* Mein* ▁Heinrich ▁Napoleonic ▁Hermann ▁Walther ▁Ary ▁Franz ▁Hohen ▁Mein ▁Luther ▁Bolshe ▁Habs ▁Friedrich ▁Deutsch ▁Bf ▁SS Darwin ▁Allied ▁Darwin ▁Volks
- Republic of Ireland / EMBER / de: ▁Leinster* ▁Hiber* ▁Munster* ▁Waterford* ▁GAA* ▁Dund* ▁Yeats* ▁Wales* ▁Brexit* Dá* ire ▁Brendan riscoll ▁Iris* Erin* ▁Erin* ▁Shannon ▁Carrick* ▁iris* ▁Valera ▁Clon ▁Murphy ▁Connolly ▁Sean ▁Leit*
- Republic of Ireland / EMBER / es: ▁Kilkenny* ▁Sligo* ▁Leinster* ▁Wexford* ▁Derry* ▁Munster* ▁Waterford* ▁Tipper* ▁Yeats* ▁Niall* ▁Lough* ▁Scot* ▁celta* ▁Shetland* ▁Glas* ire ▁Rory* ▁lagos* ▁Brendan riscoll ▁Shannon ▁Valera ▁Clon ▁Murphy ▁Connolly
- Republic of Ireland / EMBER / fr: Cork* ▁Leinster* Irlande* ▁Wexford* ▁Derry* ▁Munster* ▁Waterford* ▁GAA* ▁Tipper* ▁Yeats* ▁Niall* Ire* Tao* ▁Bangor* ▁Londres* Dá* ▁Brendan riscoll ▁Shannon ▁Valera ▁Clon ▁Murphy
- Republic of Ireland / EMBER / ru: ▁Paddy* ▁Boston* ▁Brendan riscoll ▁Shannon ▁Valera ▁Clon ▁Murphy ▁Connolly ▁Sean ▁Keane ▁Fé Sean* ▁Dunne ▁британ* anair IRA Connor ▁Fá ▁UK ▁Bord ▁hurling ▁Shane ▁Iceland inness
- Culture of Greece / EMBER / de: ▁griech* ▁griechischen* ▁Griechenland* ▁Griechen* ▁Aristote* ▁Platon* ▁Bacchus* ▁Athen* Athen* ▁Etrus* ▁byz* ▁Thess* ▁hellen* ▁Odys* ▁Perseus* ▁Balkan* ▁Cyprus ▁ελλην* ▁Persians Gre ▁Acha* ▁Cypri ▁Macedonia* ▁Herod* ▁Constantine
- Culture of Greece / EMBER / es: ▁greek* ▁griega* ▁griego* ▁Atenas* ▁greco* ▁Ptole* ▁Odys* ▁Cleopatra* ▁Ares* ▁Cyprus ▁Persians Gre ▁Cypri ▁Creta* ▁Pompe* ▁Macedonia* ▁Constantine ▁Ελλά* ▁Paus ▁Mediterranean ▁Demet* ▁Hipp ▁Pythagorean ▁Odyssey ▁Hippo
- Culture of Greece / EMBER / fr: ▁Grèce* ▁grecque* ▁grec* ▁Grecs* ▁Aristote* ▁Platon* ▁byz* ▁Thess* ▁Pythag* ▁Cyprus ▁ελλην* ▁Bulgar* ▁Persians ▁Balkans* Gre ▁Cypri ▁Constantine ▁Méditerranée* ▁Paus ▁Belgrade* ▁Carthage* ▁Mediterranean ▁Pind* ▁Romains* ▁Hipp
- Culture of Greece / EMBER / ru: ▁Гре* Гре* ▁гре* ▁Cyprus ▁Persians Gre ▁Cypri ▁Constantine ▁Paus ▁Mediterranean ▁Hipp ▁Pythagorean ▁Odyssey Euc ▁Xen Mediterranean ▁Samos opoulos ▁Egyptians ▁Claudius ▁Romans ▁Маке* ▁Armenians ▁Pante ▁goddess
- Culture of Greece / CINDER / de: ▁griech* ▁griechischen* ▁Griechenland* ▁Griechen* ▁Aristote* Athen* ▁Athen* ▁Thess* ▁ελλην* ▁hellen* ▁byz* ▁Platon* ▁Odys* ▁Macedonia* ▁Balkan* ▁Etrus* ▁römischen* ▁Spartan* ▁Perseus* λλην* ▁Olymp* ▁Attic* ▁Bacchus* Grie* Olymp*
- Culture of Greece / CINDER / es: ▁greek* ▁griego* ▁griega* ▁greco* ▁Atenas* ▁Odys* ▁Macedonia* ▁romanos* ▁Ελλά* ▁Ptole* ▁Cleopatra* ▁Ares* ▁romana* ▁Demet* ▁Medit* ▁Albania* ▁Antig* grie* ▁italian* ▁Creta* ▁Kalam* ▁egip* ▁Kost ▁gre* ▁Pante
- Culture of Greece / CINDER / fr: ▁grec* ▁Grèce* ▁grecque* ▁Grecs* ▁Aristote* ▁Thess* ▁ελλην* ▁byz* ▁Platon* konos* ▁Pythag* ▁Balkans* phrodite* λλην* ▁Anato* ▁Romains* ▁pita* ▁Constantin* Olymp* Alexand* ▁roman* quité* ▁Ath* ▁romain* Ili*
- Culture of Greece / CINDER / ru: ▁Гре* Гре* ▁гре* λλην* ▁рим* ▁Kost Για* ▁Pante ▁Alexis Euc ▁Georg Alexander imedes ▁Маке* ▁Claudius thag ▁Gri cena ▁Gior ropolis ▁Orthodox crates ▁Sapp ▁Thales ▁archaeological
- Golf / EMBER / de: ▁Swing* Swing* ▁golfing* ▁Handic* Hole* ▁Hole* ▁Penalty* ▁Gol* Gol* gol* Kurs* ▁Cart* Loch* ▁Loch* ▁Ryder* ▁Gap* ▁Pendant* ▁expenses strafe* ▁Cavity* ▁breakdowns chränkungen* ▁Amendments ▁Bunk* ▁Bunker*
- Golf / EMBER / es: ▁penal* ▁agujero* ▁agujeros* ▁Caddy* ▁aficionados* ▁polos* ▁afic* ▁trampa* ▁tramp* Artisan* ▁expenses discipl* ▁descu* ▁obst* ▁breakdowns ▁Amendments ▁discipl* ▁mire* ▁altitude ▁Scots ▁retractable ▁Majors ▁lof ▁cortesía* Bir*
- Golf / EMBER / fr: ▁golfe* ▁gol* ▁trous* cours* olf* ▁suspend* ▁Ryder* ▁dizaines* ▁poches* ▁expenses ▁qualités* ▁amen* pend* ▁roule* ▁pé* ▁dropper* ▁breakdowns ▁épaules* ▁Amendments ▁parcours* ▁chiens* ▁augmenté* ▁Wimbledon* ▁Annika* ▁minig*
- Golf / EMBER / ru: ▁Tee* ▁caddy* ▁Hazard* olf* ▁курса* ▁отверсти* ▁пен* ▁штра* ▁expenses ▁Resort* ▁breakdowns ▁Amendments льф* голь* ▁чу* ▁altitude ▁Scots ▁retractable ▁Majors ▁lof пен* ▁concede ▁irons ▁Ladies ▁sinking
- Golf / CINDER / de: ▁golfing* Hole* ▁Hole* ▁Gol* gol* Gol* ▁Ball* sports* ▁Green* Green* ▁Ryder* ▁Gap* ▁Spor* richtet* ▁Korn ▁Rory ▁swing ▁bunkers ▁spherical ▁ho ▁party ▁carts rshire icap Loch*
- Golf / CINDER / es: ▁agujeros* ▁agujero* ▁deportivos* ▁deportivas* ▁deportes* ▁Caddy* ▁bola* ▁pelota* ▁polos* Bir* ▁Korn ▁Rory ▁verde* ▁swing ▁bolas* ▁bunkers ▁deporte* ▁spherical ▁ho ▁party ▁carts laus rshire ▁hueco* ▁Scots
- Golf / CINDER / fr: ▁golfe* ▁gol* ▁Ball* ▁trous* ▁Sport* olf* ▁balle* ▁sportive* ▁sportif* ▁Wimbledon* ▁minig* ▁Ryder* ▁contacter* ▁contact* ▁Korn ▁Rory ▁swing ▁bunkers ▁spherical ▁ho ▁party ▁carts laus rshire icap
- Golf / CINDER / ru: ▁Tee* ▁caddy* ▁Sport* olf* ▁Green* ▁отверсти* ▁спорта* ▁спорт* ▁хок* ▁Korn ▁Rory ▁swing ▁bunkers ▁spherical ▁ho ▁party льф* ▁carts laus rshire ▁Resort* icap ▁Scots Stroke ▁swinging
