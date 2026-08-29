"""The loop: propose -> build -> smoke -> screen -> full -> gate -> record.

The LLM has broad reasoning freedom inside a narrow typed contract. It cannot
modify the evaluator, read hidden-test data, approve its own promotion, or
delete a failed experiment from the ledger (Section 7.2).
"""
