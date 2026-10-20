"""CINDER vs EMBER, English fit only (docs/Part 2/plan.md, "Current plan").

Pipeline, one concept at a time (see docs/Part 2/cinder_code_review.md):
    build_features  Ori's fit (whitened matrix, method rn) + two-way judge -> EMBER's file format
    edit_size       CPU: edit size per token, CINDER vs EMBER -> the delta grid
    grid            GPU: edit at each delta, fixed MC scoring, Alpaca completions for the top 3
    pick            GPU: Qwen judge on the top 3, best delta (Part 1's selection code)
    final_test      GPU: QA_test / SimdomQA_test in 5 languages + MMLU test at the best delta
"""
