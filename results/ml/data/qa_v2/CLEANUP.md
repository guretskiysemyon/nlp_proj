# QA v2 cleanup: it, pt, hi, th MC sets

`results/ml/data/qa_v2/mc_questions_<lang>.json`: same schema, concept keys, splits, item order and
option order as v1 (`results/ml/data/qa/`) and as en/de/es/fr, so item *i* and its correct letter
stay aligned across languages; `correct_answer` is the translated option at the English correct
position. en/de/es/fr are unchanged and not copied here.

## What was done

v1 = NLLB-200 3.3B per string + option lists translated in context ("a; b; c; d") + MADLAD-400
for LaBSE < 0.75, chosen by LaBSE. Remaining v1 problems: items with two identical translated
options, Hindi/Thai strings left in English (a copy of the source gets LaBSE ~1, so v1 preferred
it), and segments with LaBSE < 0.75. v2 targets every segment that is low (LaBSE < 0.75), lost
a number, or (hi, th) has no character of the target script, plus every option of a duplicate
item. New candidates for them: 4-best of NLLB (beam 8), 4-best of MADLAD (beam 8), and the item's
option list translated in one line by MADLAD. All candidates (v1 ones included) are ranked by
numbers kept > target script present (hi, th) > LaBSE. Duplicates still left: next-best distinct
candidate, else the English source text of that option (listed below).

LaBSE caveat: cosine of LaBSE embeddings is a weak judge for single words and names (correct
transliterations such as it 'Isocrate' score ~0.7), so 'LaBSE < 0.75' over-counts real errors,
and it cannot see register or nuance. No Kiwi (gated model) and no human check.

## Summary

| lang | items | items changed | strings changed | target segments | segments changed | dup items v1 -> v2 | English fallback options | no-script strings v1 -> v2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| it | 3600 | 705 | 1095 | 868 | 673 | 8 -> 0 | 2 | n/a |
| pt | 3600 | 691 | 1077 | 816 | 657 | 11 -> 0 | 1 | n/a |
| hi | 3600 | 1576 | 2807 | 2088 | 1735 | 10 -> 0 | 2 | 248 -> 0 |
| th | 3600 | 2318 | 4942 | 4242 | 3457 | 21 -> 0 | 2 | 1189 -> 1 |

'items changed' = items with at least one string (question or option) different from v1.

## LaBSE per unique segment, v1 -> v2

**it** (changed target segments by engine: {'nllb_nbest': 312, 'madlad_nbest': 318, 'madlad_ctx': 43})

- all segments, v1: median 0.936, p10 0.805, mean 0.915, < 0.75: 5.9%, < 0.5: 1.0%
- all segments, v2: median 0.938, p10 0.826, mean 0.925, < 0.75: 2.7%, < 0.5: 0.3%
- target segments, v1: median 0.669, p10 0.450, mean 0.638, < 0.75: 92.9%, < 0.5: 16.1%
- target segments, v2: median 0.774, p10 0.582, mean 0.779, < 0.75: 43.1%, < 0.5: 4.6%

**pt** (changed target segments by engine: {'nllb_nbest': 291, 'madlad_ctx': 64, 'madlad_nbest': 302})

- all segments, v1: median 0.921, p10 0.801, mean 0.907, < 0.75: 5.5%, < 0.5: 0.9%
- all segments, v2: median 0.923, p10 0.820, mean 0.917, < 0.75: 2.6%, < 0.5: 0.2%
- target segments, v1: median 0.666, p10 0.456, mean 0.640, < 0.75: 93.1%, < 0.5: 14.6%
- target segments, v2: median 0.774, p10 0.591, mean 0.783, < 0.75: 43.6%, < 0.5: 4.2%

**hi** (changed target segments by engine: {'madlad_ctx': 94, 'nllb_nbest': 1065, 'madlad_nbest': 536, 'nllb': 36, 'ctx': 4})

- all segments, v1: median 0.856, p10 0.727, mean 0.839, < 0.75: 13.2%, < 0.5: 0.4%
- all segments, v2: median 0.861, p10 0.763, mean 0.850, < 0.75: 6.8%, < 0.5: 0.1%
- target segments, v1: median 0.704, p10 0.589, mean 0.713, < 0.75: 86.7%, < 0.5: 2.7%
- target segments, v2: median 0.764, p10 0.656, mean 0.776, < 0.75: 44.9%, < 0.5: 0.7%

**th** (changed target segments by engine: {'nllb_nbest': 2104, 'madlad_nbest': 1102, 'madlad_ctx': 117, 'nllb': 113, 'ctx': 21})

- all segments, v1: median 0.834, p10 0.680, mean 0.821, < 0.75: 20.2%, < 0.5: 1.3%
- all segments, v2: median 0.839, p10 0.733, mean 0.829, < 0.75: 12.3%, < 0.5: 0.5%
- target segments, v1: median 0.713, p10 0.566, mean 0.752, < 0.75: 71.4%, < 0.5: 4.4%
- target segments, v2: median 0.774, p10 0.638, mean 0.781, < 0.75: 42.1%, < 0.5: 1.6%

## Items not fully fixed

- it Republic of Ireland/QA_train[36]: option 1 kept in English ('Provinces'; translations ['Province', 'Province', 'Parrocchie', 'Borough'])
- it Artificial intelligence/SimdomQA_test[9]: option 3 kept in English ('Encoding'; translations ['Compressione', 'Decodifica', 'Codifica', 'Codifica'])
- pt Halloween/SimdomQA_train[47]: option 3 kept in English ('Pouch'; translations ['Carrinho', 'Cubo', 'Bolsa', 'Bolsa'])
- hi Cannabis/SimdomQA_train[8]: option 2 kept in English ('Cigarette'; translations ['सिगरेट', 'सिगारिलो', 'सिगरेट', 'पाइप'])
- hi Cannabis/SimdomQA_train[9]: option 2 kept in English ('Cigarette'; translations ['सिगारिलो', 'सिगरेट', 'सिगरेट', 'चेरोट'])
- th COVID-19 pandemic/SimdomQA_test[48]: option 1 kept in English ('Cat'; translations ['แมว', 'แมว', 'สุนัข', 'นก'])
- th World War II/SimdomQA_test[14]: option 1 kept in English ('Restoration'; translations ['การฟื้นฟู', 'การฟื้นฟู', 'ความตื่นตัวอันยิ่งใหญ่', 'ยุคฟื้นฟูศิลปะ (Renaissance)'])

Build wall time: 7 s (CPU). GPU wall time: see the ml_eval_v2 job logs.
