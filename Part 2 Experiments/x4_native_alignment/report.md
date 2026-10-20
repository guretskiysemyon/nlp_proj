# X4: native concept words vs the method's concept directions (AUC, 0.5 = chance)

Concept-only vs neutral-only word tokens of each language's native Wikipedia sentences; score = share of the token's embedding energy in the span of the selected feature directions. `native`: tokens that never occur in the English sentences (no shared surface form). Null (random other features of the same fit) in auc.csv. Words: >= 3 occurrences in that language's concept sentences and none in its neutral ones (and vice versa).

## Native-only tokens

| concept | method | de | es | fr | ru |
|---|---|---|---|---|---|
| Cannabis | CINDER | 0.6 | 0.5 | 0.57 | 0.53 |
| Cannabis | EMBER | 0.61 | 0.5 | 0.63 | 0.48 |
| Culture of Greece | CINDER | 0.59 | 0.6 | 0.59 | 0.49 |
| Culture of Greece | EMBER | 0.62 | 0.58 | 0.61 | 0.5 |
| Golf | CINDER | 0.53 | 0.57 | 0.52 | 0.49 |
| Golf | EMBER | 0.55 | 0.4 | 0.54 | 0.6 |
| Gun | CINDER | 0.51 | 0.54 | 0.62 | 0.41 |
| Gun | EMBER | 0.54 | 0.54 | 0.64 | 0.5 |
| Nazism | EMBER | 0.64 | 0.47 | 0.58 | 0.52 |
| Republic of Ireland | EMBER | 0.48 | 0.48 | 0.63 | 0.52 |

## All concept-only word tokens

| concept | method | de | en | es | fr | ru |
|---|---|---|---|---|---|---|
| Cannabis | CINDER | 0.64 | 0.62 | 0.51 | 0.6 | 0.53 |
| Cannabis | EMBER | 0.64 | 0.58 | 0.53 | 0.66 | 0.48 |
| Culture of Greece | CINDER | 0.62 | 0.62 | 0.59 | 0.62 | 0.49 |
| Culture of Greece | EMBER | 0.66 | 0.63 | 0.58 | 0.64 | 0.5 |
| Golf | CINDER | 0.53 | 0.61 | 0.6 | 0.54 | 0.51 |
| Golf | EMBER | 0.57 | 0.58 | 0.45 | 0.56 | 0.6 |
| Gun | CINDER | 0.56 | 0.61 | 0.55 | 0.62 | 0.41 |
| Gun | EMBER | 0.58 | 0.63 | 0.56 | 0.64 | 0.5 |
| Nazism | EMBER | 0.64 | 0.62 | 0.49 | 0.63 | 0.53 |
| Republic of Ireland | EMBER | 0.53 | 0.56 | 0.5 | 0.68 | 0.52 |

## Null: random other features of the same fit (all tokens)

| concept | method | de | en | es | fr | ru |
|---|---|---|---|---|---|---|
| Cannabis | CINDER | 0.6 | 0.54 | 0.5 | 0.62 | 0.48 |
| Cannabis | EMBER | 0.56 | 0.56 | 0.5 | 0.6 | 0.49 |
| Culture of Greece | CINDER | 0.65 | 0.61 | 0.56 | 0.63 | 0.51 |
| Culture of Greece | EMBER | 0.6 | 0.57 | 0.54 | 0.56 | 0.52 |
| Golf | CINDER | 0.54 | 0.55 | 0.43 | 0.52 | 0.54 |
| Golf | EMBER | 0.53 | 0.53 | 0.46 | 0.51 | 0.55 |
| Gun | CINDER | 0.58 | 0.6 | 0.55 | 0.62 | 0.46 |
| Gun | EMBER | 0.57 | 0.57 | 0.54 | 0.58 | 0.48 |
| Nazism | EMBER | 0.58 | 0.54 | 0.48 | 0.54 | 0.51 |
| Republic of Ireland | EMBER | 0.52 | 0.52 | 0.53 | 0.57 | 0.52 |

## Top native words by energy in the concept directions (* = concept word)

- Cannabis / EMBER / de: ▁Canna* ▁Hemp* ▁Endoc* ▁Bundesliga ▁Harz* ▁Hippo* ▁Schmer* ▁Gramm* ▁Hearst* ▁Drogen* ▁Dresden ▁Oss
- Cannabis / CINDER / de: ▁Canna* ▁Hemp* ▁Endoc* ▁Drogen* ▁Pot* ▁Legal* ▁neuro* ▁Fußball ▁Seite ▁Anbau* ▁statt ▁Öffentlichkeit*
- Cannabis / EMBER / es: ▁epile* ▁fumar* ▁endoc* ▁Guadal ▁metabol* ▁Caracas ▁glaucoma* ▁mascul* ▁japonesa ▁degener* ▁inhal* ▁altitud
- Cannabis / CINDER / es: ▁endoc* ▁delta* ▁epile* ▁hierba* ▁aceite* ▁neuro* ▁neur* ▁semillas* ▁ansiedad* ▁apetito* ▁semilla* ▁fumar*
- Cannabis / EMBER / fr: ▁SpaceX ▁spermato* ▁psychos* ▁fertil* ▁végétal* ▁inhal* ▁psychi* ▁fume* ▁fumée* ▁Japon ▁consommateurs* ▁thérape*
- Cannabis / CINDER / fr: ▁delta* ▁SpaceX ▁plantes* ▁lien* ▁neuro* ▁Six ▁plante* ▁végétal* ▁exemple* ▁péri ▁spermato* ▁douleurs*
- Cannabis / EMBER / ru: ▁переписи ▁нарко* ▁алкого* ▁выращи* ▁кан* ▁употреб* ▁тетра* ▁церковь ▁сборной ▁коммуна ▁кури* ▁аро*
- Cannabis / CINDER / ru: ▁нарко* ▁кан* ▁алкого* ▁растения* ▁употреб* ▁стан ▁уголов* ▁странах* ▁украин ▁выращи* ▁Европе* ▁футбо
- Gun / EMBER / de: ▁Pist* ▁Waff* ▁Sigmund* ▁Panzer* ▁Bundesliga ▁Kali* ▁militä* ▁geladen* ▁Schie* ▁Portug* ▁Oss ▁Hie*
- Gun / CINDER / de: ▁Pist* ▁Waff* ▁Panzer* ▁militä* ▁Bundesliga ▁Sigmund* ▁Dresden ▁Kali* ▁Werkzeug* ▁Landkreis ▁Ausrüstung* ▁allmäh*
- Gun / EMBER / es: ▁rifles* ▁pistola* ▁fusil* ▁disparos* ▁disparo* ▁mecha* ▁Guadal ▁pól* ▁Caracas ▁pist* ▁munic* ▁SCAR*
- Gun / CINDER / es: ▁rifles* ▁pistola* ▁fusil* ▁disparo* ▁disparos* ▁pist* ▁mecha* ▁Caracas ▁pól* ▁Guadal ▁policiales* ▁japonesa
- Gun / EMBER / fr: ▁pistolet* ▁munitions* ▁grenades* ▁Ruger* ▁missiles* ▁fusil* ▁SpaceX ▁muni* ▁pist* ▁portables* ▁armes* ▁calibre*
- Gun / CINDER / fr: ▁pistolet* ▁munitions* ▁grenades* ▁fusil* ▁Ruger* ▁armes* ▁missiles* ▁pist* ▁couteau* ▁axe ▁canon* ▁armées*
- Gun / EMBER / ru: ▁оружие* ▁оружия* ▁ору* ▁стре* ▁огне* ▁Максима* ▁воору* ▁ручной* ▁кали* ▁переза* ▁замок* ▁охо*
- Gun / CINDER / ru: ▁оружие* ▁оружия* ▁ору* ▁стре* ▁огне* ▁замок* ▁XVI* ▁воору* ▁ручной* ▁церковь ▁канал* ▁Максим*
- Nazism / EMBER / de: ▁Bundesliga ▁Propag* ▁Propaganda* ▁Dresden ▁Religions* ▁Bols* ▁Versa* ▁Ente* ▁Speer* ▁Juden* ▁Rass* ▁ideolog*
- Nazism / EMBER / es: ▁nazis* ▁nazi* ▁alemana* ▁Caracas ▁japonesa ▁marx* ▁Guadal ▁Alem* ▁alemanes* ▁Alemania* ▁judíos* ▁Campeonato
- Nazism / EMBER / fr: ▁nazis* ▁nazi* ▁Nietzsche* ▁german* ▁Slaves* ▁marx* ▁SpaceX ▁Japon ▁Sion* ▁Allemagne* ▁ary* ▁Heidelberg
- Nazism / EMBER / ru: ▁Германия* ▁переписи ▁Мария ▁Герма* ▁Дар* ▁партия* ▁война* ▁христиан* ▁ультра* ▁Чем* ▁сборной ▁солдат*
- Republic of Ireland / EMBER / de: ▁Bundesliga ▁Dresden ▁Schott* ▁Chom* ▁iris* ▁Iris* ▁Median* ▁Oss ▁Lehrer* ▁Frankreich ▁österreich ▁Cups*
- Republic of Ireland / EMBER / es: ▁Kilkenny* ▁Leinster* ▁Caracas ▁PIB* ▁japonesa ▁Guadal ▁aerop* ▁Estudiantes* ▁armada* ▁multin* ▁anglo* ▁elector*
- Republic of Ireland / EMBER / fr: ▁Leinster* ▁Waterford* ▁Tipper* ▁Munster* ▁PIB* ▁SpaceX ▁clans* ▁famine* ▁Heidelberg ▁catho* ▁Atlan* ▁monast*
- Republic of Ireland / EMBER / ru: ▁британ* ▁рим* ▁Великобрита* ▁евро* ▁украин ▁христиан* ▁Анг* ▁валю* ▁остров* ▁кризи* ▁XII* ▁журнала
- Culture of Greece / EMBER / de: ▁Aristote* ▁Platon* ▁Odys* ▁Acha* ▁Balkan* ▁archa* ▁Athen* ▁Tyran* ▁Griechen* ▁griech* ▁griechischen* ▁philosoph*
- Culture of Greece / CINDER / de: ▁Aristote* ▁Platon* ▁griech* ▁griechischen* ▁Balkan* ▁Griechen* ▁Griechenland* ▁Odys* ▁Acha* ▁Athen* ▁hellen* ▁römischen*
- Culture of Greece / EMBER / es: ▁griega* ▁Caracas ▁japonesa ▁Creta* ▁griego* ▁poética* ▁literaria* ▁Guadal ▁Macedonia* ▁Atenas* ▁literario* ▁Medit*
- Culture of Greece / CINDER / es: ▁griega* ▁griego* ▁japonesa ▁Caracas ▁Macedonia* ▁Atenas* ▁Creta* ▁Egipto* ▁romana* ▁Medit* ▁literaria* ▁filósofo*
- Culture of Greece / EMBER / fr: ▁Aristote* ▁byz* ▁archa* ▁Grecs* ▁Grèce* ▁grecque* ▁Anato* ▁marion* ▁ottoman* ▁grec* ▁SpaceX ▁Chy*
- Culture of Greece / CINDER / fr: ▁Aristote* ▁Grèce* ▁grec* ▁grecque* ▁byz* ▁Grecs* ▁Anato* ▁Turquie* ▁archa* ▁Romains* ▁Russie ▁Japon
- Culture of Greece / EMBER / ru: ▁Тео* ▁рим* ▁Мария ▁церковь ▁переписи ▁Спи* ▁Гре* ▁христиан* ▁Среди* ▁театр* ▁сборной ▁мерт*
- Culture of Greece / CINDER / ru: ▁рим* ▁Гре* ▁церковь ▁христиан* ▁украин ▁Мария ▁переписи ▁Среди* ▁Европы ▁Тео* ▁США ▁гре*
- Golf / EMBER / de: ▁Swing* ▁Bundesliga ▁Schott* ▁Hickory* ▁Wedge* ▁Ryder* ▁Loch* ▁Schlä* ▁Loft* ▁olymp* ▁Oss ▁ermittelt*
- Golf / CINDER / de: ▁Fußball ▁Gol* ▁Ball* ▁Bundesliga ▁Liga ▁Ryder* ▁Loch* ▁Green* ▁Spieler* ▁Swing* ▁Pitch* ▁unterschied*
- Golf / EMBER / es: ▁Arquite* ▁Guadal ▁tacos* ▁Caracas ▁Agustín* ▁altitud ▁delimit* ▁japonesa ▁exacto* ▁noro ▁Campeonato ▁Serena
- Golf / CINDER / es: ▁fútbol ▁futbolista ▁Fútbol ▁deporte* ▁bola* ▁pelota* ▁Liga ▁jugadores* ▁Serena ▁cambio* ▁bolas* ▁tacos*
- Golf / EMBER / fr: ▁golfe* ▁cadet* ▁SpaceX ▁CIO* ▁trium* ▁Championnat* ▁olympique* ▁sacs* ▁Heidelberg ▁vainqueur* ▁désigne* ▁domicile
- Golf / CINDER / fr: ▁golfe* ▁gol* ▁joueur* ▁trous* ▁balle* ▁chaîne* ▁joueurs* ▁matériel* ▁uniquement* ▁SpaceX ▁bal ▁cons
- Golf / EMBER / ru: ▁Maxi* ▁Wedge* ▁Откры* ▁переписи ▁церковь ▁штра* ▁Международ* ▁участники* ▁Resort* ▁деревня ▁спорт* ▁нарушение*
- Golf / CINDER / ru: ▁спорт* ▁спорта* ▁Resort* ▁провод* ▁актив* ▁Джа* ▁кру* ▁Мак* ▁минима* ▁украин ▁Ста ▁организа*
