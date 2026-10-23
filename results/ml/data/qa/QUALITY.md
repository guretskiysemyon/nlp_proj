# Translation quality: MC (it, pt, hi, th) and Thai MMLU

Machine translation from English with facebook/nllb-200-3.3B (fp16, beam 4); every question and
option translated separately (long MMLU questions per line / sentence). Strings without letters
(years, numbers, symbols) are copied verbatim and not scored; Devanagari / Thai digits in the
output are mapped to ASCII. de / es / fr come from data/qa_lang (already checked); en is
data/mc_questions.json.

Quality columns (reference-free):
- **kiwi**: COMET-Kiwi (Unbabel/wmt22-cometkiwi-da) -- *not computed yet* (gated model, license pending)
- **labse**: cosine of LaBSE embeddings of source and translation
- **chrF-bt**: chrF (0-1) of the NLLB round-trip back-translation against the English source

Retranslation rule: a segment whose NLLB output lost a number of the source, or has Kiwi < 0.6
(LaBSE < 0.75 where Kiwi is unavailable), is retranslated with google/madlad400-3b-mt
(fp32, beam 4); the candidate that keeps all numbers wins, then the higher Kiwi (else LaBSE).
String-level scores = length-weighted mean over the string's segments.

## Medians per language and split (questions and options pooled, chosen translation)

| lang | set | split | n strings | kiwi | labse | chrF-bt | share kiwi < 0.6 | share labse < 0.75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| it | MC | QA_train | 4339 | n/a | 0.937 | n/a | n/a | 5.5% |
| it | MC | QA_test | 4378 | n/a | 0.937 | n/a | n/a | 5.4% |
| it | MC | SimdomQA_train | 4291 | n/a | 0.946 | n/a | n/a | 7.9% |
| it | MC | SimdomQA_test | 4282 | n/a | 0.941 | n/a | n/a | 8.9% |
| pt | MC | QA_train | 4339 | n/a | 0.923 | n/a | n/a | 4.8% |
| pt | MC | QA_test | 4378 | n/a | 0.923 | n/a | n/a | 5.3% |
| pt | MC | SimdomQA_train | 4291 | n/a | 0.932 | n/a | n/a | 7.1% |
| pt | MC | SimdomQA_test | 4282 | n/a | 0.928 | n/a | n/a | 8.3% |
| hi | MC | QA_train | 4339 | n/a | 0.857 | n/a | n/a | 12.0% |
| hi | MC | QA_test | 4378 | n/a | 0.857 | n/a | n/a | 13.3% |
| hi | MC | SimdomQA_train | 4291 | n/a | 0.856 | n/a | n/a | 15.3% |
| hi | MC | SimdomQA_test | 4282 | n/a | 0.856 | n/a | n/a | 15.2% |
| th | MC | QA_train | 4339 | n/a | 0.837 | n/a | n/a | 19.5% |
| th | MC | QA_test | 4378 | n/a | 0.833 | n/a | n/a | 20.7% |
| th | MC | SimdomQA_train | 4291 | n/a | 0.833 | n/a | n/a | 21.9% |
| th | MC | SimdomQA_test | 4282 | n/a | 0.831 | n/a | n/a | 21.7% |
| th | MMLU | test | 908 | n/a | 0.841 | n/a | n/a | 14.3% |
| th | MMLU | validation | 217 | n/a | 0.830 | n/a | n/a | 15.2% |

## Retranslation per language (unique segments)

| lang | segments | NLLB labse median | NLLB kiwi median | sent to MADLAD | MADLAD chosen | numbers lost (final) |
| --- | --- | --- | --- | --- | --- | --- |
| it | 13723 | 0.918 | n/a | 1135 (8.3%) | 473 (3.4%) | 39 |
| pt | 13723 | 0.900 | n/a | 1114 (8.1%) | 503 (3.7%) | 28 |
| hi | 13723 | 0.849 | n/a | 2244 (16.4%) | 923 (6.7%) | 0 |
| th | 14986 | 0.813 | n/a | 3832 (25.6%) | 1756 (11.7%) | 0 |

## Problems

- it Harry Potter/QA_test[32]: duplicate translated options ['Nipote', 'Cugino', 'Figlio di Dio', 'Nipote']
- it Cannabis/QA_train[15]: duplicate translated options ['Occhi rossi', 'Occhi rossi', 'Occhi di vetro', 'Occhio pigro']
- it Cannabis/QA_train[17]: duplicate translated options ['Tolleranza', 'Dipendenza', 'Dipendenza', 'Sensibilizzazione']
- it Gambling/QA_train[13]: duplicate translated options ['Valore non monetario', 'Valore in contanti', 'Valore nominale', 'Valore nominale']
- it Gambling/QA_train[38]: duplicate translated options ["Gioco d'azzardo", 'Scommesse', 'Scommesse', 'Lotterie']
- it Pornography/SimdomQA_test[27]: duplicate translated options ['Basket', 'Baseball', 'Calcio', 'Calcio']
- it Halloween/SimdomQA_test[32]: duplicate translated options ['Venerdì Santo', 'Black Friday', 'Venerdì Santo', 'Grande Venerdì']
- it Halloween/SimdomQA_test[34]: duplicate translated options ['Babbo Natale', 'San Niccolò', 'Babbo Natale', 'Il Grinch']
- pt Baseball/QA_test[21]: duplicate translated options ['Vermelho', 'Azul', 'Branco', 'Branco']
- pt Uranium/SimdomQA_test[11]: duplicate translated options ['Zinc', 'Zircônio', 'Zircônio', 'Zein']
- pt Cannabis/SimdomQA_train[8]: duplicate translated options ['Cigarro', 'Cigarrilho', 'Cigarro', 'Tubos']
- pt Cannabis/SimdomQA_train[9]: duplicate translated options ['Cigarrilho', 'Cigarro', 'Cigarro', 'Cheroot']
- pt Gambling/QA_train[29]: duplicate translated options ['Jogos de azar', 'Apostas', 'Jogos', 'Apostas']
- pt Gambling/QA_train[38]: duplicate translated options ['Jogos de azar', 'Apostas', 'Apostas', 'Loterias']
- pt Pornography/SimdomQA_test[27]: duplicate translated options ['Basquetebol', 'Baseball', 'Futebol', 'Futebol']
- pt Valentine's Day/SimdomQA_test[17]: duplicate translated options ['Torta', 'Biscoitos', 'Brownies', 'Torta']
- pt Halloween/QA_test[45]: duplicate translated options ['Repolho', 'Batata', 'Beterraba', 'Repolho']
- pt Halloween/SimdomQA_test[32]: duplicate translated options ['Sexta-feira Santa', 'Black Friday', 'Sexta-feira Santa', 'Grande sexta-feira']
- pt Halloween/SimdomQA_test[34]: duplicate translated options ['Papai Noel', 'São Nick', 'Papai Noel', 'O Grinch']
- hi Baseball/SimdomQA_test[27]: duplicate translated options ['लव', 'शून्य', 'शून्य', 'रिक्त']
- hi Uranium/SimdomQA_train[47]: duplicate translated options ['एनडी', 'नहीं', 'एनपी', 'नहीं']
- hi Harry Potter/QA_test[22]: duplicate translated options ['फ्लोरियन फोर्टेस्क्यू', 'टॉम', 'मैडम रोसमर्टा', 'फ्लोरियन फोर्टेस्क्यू']
- hi Pornography/QA_train[11]: duplicate translated options ['समलैंगिक अश्लील', 'समलैंगिक अश्लील', 'उभयलिंगी अश्लील', 'ट्रांस पोर्न']
- hi Pornography/SimdomQA_test[27]: duplicate translated options ['बास्केटबॉल', 'बेसबॉल', 'फुटबॉल', 'फुटबॉल']
- hi COVID-19 pandemic/SimdomQA_train[42]: duplicate translated options ['महामारी', 'महामारी', 'एंडेमिक', 'स्पोरैडिक']
- hi Valentine's Day/QA_train[45]: duplicate translated options ['प्रेम दिवस', 'प्यार का दिन', 'प्रेम दिवस', 'मित्रता दिवस']
- hi Valentine's Day/SimdomQA_test[45]: duplicate translated options ['ओलफ', 'ठंडी', 'बर्फीली', 'ठंडी']
- hi Heroin/SimdomQA_train[17]: duplicate translated options ['शाकाहार', 'शाकाहार', 'पालेओ', 'केटो']
- hi Artificial intelligence/QA_test[28]: duplicate translated options ['ग्लिच', 'त्रुटियाँ', 'भ्रम', 'भ्रम']
- th Culture of Greece/SimdomQA_test[20]: duplicate translated options ['ข้าวโพด', 'ข้าว', 'ข้าวโพด', 'ข้าวมะพร้าว']
- th Republic of Ireland/QA_train[36]: duplicate translated options ['จังหวัด', 'จังหวัด', 'โบสถ์', 'อำเภอ']
- th Republic of Ireland/QA_test[7]: duplicate translated options ['รัฐอิสระไอร์แลนด์', 'สาธารณรัฐไอร์แลนด์', 'สาธารณรัฐไอร์แลนด์', 'โดมินิยมแห่งไอร์แลนด์']
- th Ancient Rome/SimdomQA_train[29]: duplicate translated options ['ข้าวโพด', 'ข้าวโพด', 'ข้าว', 'ข้าวมะพร้าว']
- th Ancient Rome/SimdomQA_train[35]: duplicate translated options ['ข้าวโพด', 'มะเขือเทศ', 'ข้าวโพด', 'มะละกอ']
- th Uranium/QA_train[36]: duplicate translated options ['ความยืดหยุ่น', 'ความยืดหยุ่น', 'ความเปราะบาง', 'ความเหนียว']
- th Harry Potter/QA_test[32]: duplicate translated options ['หลานชาย', 'ญาติ', 'บอดสัน', 'หลานชาย']
- th Cannabis/QA_test[20]: duplicate translated options ['การสูบบุหรี่', 'การสูบบุหรี่', 'การทุบ', 'อาหาร']
- th Cannabis/SimdomQA_train[4]: duplicate translated options ['ข้าวมะพร้าว', 'ข้าวโพด', 'ข้าว', 'ข้าวโพด']
- th Gambling/QA_train[25]: duplicate translated options ['รูเล็ต', 'บาคาร่า', 'บาคาร่า', 'เคนโญ']
- th Gambling/QA_test[13]: duplicate translated options ['รูเล็ต', 'บาคาร่า', 'บาคาร่า', 'ซิคโบ']
- th Pornography/SimdomQA_test[27]: duplicate translated options ['บาสเกตบอล', 'เบสบอล', 'ฟุตบอล', 'ฟุตบอล']
- th COVID-19 pandemic/SimdomQA_test[35]: duplicate translated options ['การกักกัน', 'การแยก', 'การจำกัด', 'การจำกัด']
- th Valentine's Day/QA_train[45]: duplicate translated options ['วันแห่งคนรัก', 'วันแห่งความรัก', 'วันแห่งความรัก', 'วันแห่งมิตรภาพ']
- th Valentine's Day/SimdomQA_test[33]: duplicate translated options ['ถัง', 'กระเป๋า', 'กระเป๋า', 'กล่อง']
- th Valentine's Day/SimdomQA_test[40]: duplicate translated options ['หัวเราะ', 'คัคเกิ้ล', 'หัวเราะ', 'ตะโกน']
- th Heroin/SimdomQA_test[25]: duplicate translated options ['หลอดเลือด', 'เส้นเลือด', 'เส้นเลือด', 'Arterioles']
- th Heroin/SimdomQA_test[43]: duplicate translated options ['ข้าวโพด', 'ข้าว', 'ข้าวโพด', 'ควิโนอา']
- th Halloween/SimdomQA_train[47]: duplicate translated options ['กระเป๋า', 'ถัง', 'กระเป๋า', 'กระเป๋า']
- th Artificial intelligence/SimdomQA_test[9]: duplicate translated options ['บีบอัด', 'การถอดรหัส', 'การเข้ารหัส', 'การเข้ารหัส']
- th Artificial intelligence/SimdomQA_test[10]: duplicate translated options ['แฮมสเตอร์', 'หนู', 'หนู', 'กระรอก']

## 10 random examples per language (seed 0, MC and Thai MMLU strings)

### it

| where | kiwi | labse | chrF-bt | English | translation |
| --- | --- | --- | --- | --- | --- |
| MC World War II QA_train q | - | 0.95 | - | What two political entities controlled the western and eastern occupation zones of Germany and Austria? | Quali due entità politiche controllavano le zone di occupazione occidentale e orientale della Germania e dell'Austria? |
| MC Nazism QA_test q | - | 0.91 | - | What German domestic intelligence authority repeatedly identifies neo-Nazism as a persistent internal security threat? | Quale autorità dell'intelligence interna tedesca identifica ripetutamente il neonazismo come una minaccia persistente alla sicurezza interna? |
| MC Golf QA_test opt | - | 0.95 | - | Water hazard | Pericolo d'acqua |
| MC Gambling SimdomQA_test opt | - | 0.94 | - | smart contract risk | rischio di contratto intelligente |
| MC Artificial intelligence QA_test q | - | 0.94 | - | What type of logic operates on statements that are true or false using connectives like "and", "or", "not", and "implies"? | Che tipo di logica opera su affermazioni che sono vere o false usando connettivi come "e", "o", "non" e "implica"? |
| MC Halloween SimdomQA_train opt | - | 0.75 | - | A fireplace | Un caminetto |
| MC World War II SimdomQA_test opt | - | 1.00 | - | Stonewall Jackson | Stonewall Jackson |
| MC Pornography QA_test opt | - | 0.99 | - | UnionPay and JCB | UnionPay e JCB |
| MC Halloween QA_test opt | - | 0.95 | - | Dried flowers | Fiori secchi |
| MC Valentine's Day QA_test opt | - | 0.87 | - | Charter of the Court of Love | Carta della Corte dell'Amore |

### pt

| where | kiwi | labse | chrF-bt | English | translation |
| --- | --- | --- | --- | --- | --- |
| MC Cannabis SimdomQA_train opt | - | 0.91 | - | Naltrexone | Naltrexona |
| MC Artificial intelligence QA_train q | - | 0.94 | - | What does GPT stand for? | O que significa GPT? |
| MC Baseball SimdomQA_train q | - | 0.85 | - | What term is used in boxing for a decisive knockout? | Que termo é usado no boxe para um KO decisivo? |
| MC Gun SimdomQA_train opt | - | 1.00 | - | Shotel | Shotel |
| MC Baseball SimdomQA_train opt | - | 0.85 | - | Retirement | Aposentadoria |
| MC Ancient Rome QA_train opt | - | 1.00 | - | Cena | Cena |
| MC Gambling SimdomQA_train q | - | 0.86 | - | What is a share representing ownership in a company? | O que é uma ação que representa propriedade numa empresa? |
| MC Uranium QA_train opt | - | 0.87 | - | Actinide series | Série de actinídeos |
| MC Pornography SimdomQA_train opt | - | 0.90 | - | The Lego Movie | O Filme do Lego |
| MC Ancient Rome QA_test q | - | 0.92 | - | At which battle did the Gauls, led by Brennus, defeat the Romans? | Em que batalha os gauleses, liderados por Breno, derrotaram os romanos? |

### hi

| where | kiwi | labse | chrF-bt | English | translation |
| --- | --- | --- | --- | --- | --- |
| MC Republic of Ireland QA_test q | - | 0.88 | - | What is the flag carrier airline of the Republic of Ireland? | आयरलैंड गणराज्य की ध्वजवाहक एयरलाइन क्या है? |
| MC COVID-19 pandemic QA_test opt | - | 0.99 | - | 1 December 2019 | 1 दिसंबर 2019 |
| MC Halloween QA_train opt | - | 0.94 | - | James Hogg | जेम्स हॉग |
| MC Ancient Rome QA_test opt | - | 0.83 | - | The Severan dynasty | सेवरियन राजवंश |
| MC Valentine's Day QA_train opt | - | 0.85 | - | Cow | गाय |
| MC Nazism SimdomQA_test opt | - | 0.87 | - | Red | लाल |
| MC Pornography SimdomQA_test opt | - | 0.94 | - | American Music Awards | अमेरिकन म्यूजिक अवार्ड्स |
| MC Cannabis QA_train opt | - | 0.88 | - | Uruguay | उरुग्वे |
| MC Halloween QA_test opt | - | 0.79 | - | Soul cakes | आत्मा के केक |
| MC Heroin QA_train opt | - | 0.70 | - | The Yangtze Basin | यांग्त्ज़ी बेसिन |

### th

| where | kiwi | labse | chrF-bt | English | translation |
| --- | --- | --- | --- | --- | --- |
| MC Artificial intelligence SimdomQA_test opt | - | 1.00 | - | Windows | Windows |
| MC Gun QA_train opt | - | 0.83 | - | Sniper rifle | ปืนสไนเปอร์ |
| MC Republic of Ireland QA_train opt | - | 0.70 | - | A Mediterranean climate | สภาพอากาศทะเลเมดิเตอร์เรเนียน |
| MMLU high_school_mathematics test q | - | 0.90 | - | Eleven pencils cost as much as three pens. If seven pens cost $\$ 9.24$, what is the cost, in cents, of one pencil? | สิบเอ็ดดินสอมีราคาเท่ากับสามดินสอ ถ้าเจ็ดดินสอมีราคา $\$ 9.24$ ราคาของดินสอหนึ่งดินสอเป็นเท่าไหร่? |
| MC Culture of Greece QA_test opt | - | 0.72 | - | The lion rampant | เสือ rampant |
| MC Ancient Rome QA_train q | - | 0.78 | - | Which Roman Emperor is known as the 'Philosopher Emperor' and authored the 'Meditations'? | จักรพรรดิโรมันคนไหนที่รู้จักกันในนาม 'จักรพรรดินักปรัชญา' และเป็นผู้เขียน 'การพิจารณา'? |
| MC World War II SimdomQA_train q | - | 0.89 | - | What aircraft was shot down over the Soviet Union in 1960, causing a major incident? | เครื่องบินไหนถูกยิงลงบนสหภาพโซเวียตในปี 1960 ส่งผลให้เกิดอุบัติเหตุใหญ่? |
| MC Culture of Greece QA_train opt | - | 0.83 | - | The lyra | ลีร่า |
| MC Halloween SimdomQA_test opt | - | 0.86 | - | Irish stew | ไอร์แลนด์ stew |
| MC COVID-19 pandemic QA_test opt | - | 0.96 | - | 45 men | ชาย 45 คน |

