# Evaluation results

Run: 2026-10-21T13:50:26.553282+00:00 · Model: `openai/gpt-oss-20b`

- Expected answer/abstention behavior: 10/10.
- Mean expected-source recall@4: 100.0% across 8 completed answerable cases.
- Errors: 0.

This is a small synthetic smoke benchmark, not an accuracy estimate. Term matching checks coverage only; inspect answers and quotes for semantic correctness. Retrieval and citation checks use independently specified source IDs. The runtime model-based verifier is not an independent evaluator.

## q1: How long do I have to return an unused item?

Reference: Unused items can be returned within thirty days of purchase.

Status: answered · Time: 1.26s

Customers may return unused items within thirty days of purchase.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: Customers may return unused items within thirty days of purchase.

## q2: How long does standard delivery take?

Reference: Standard delivery takes five business days.

Status: answered · Time: 1.37s

Standard delivery takes five business days.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/shipping: Standard delivery takes five business days

## q3: What does the electronics warranty cover and how long does it last?

Reference: Manufacturing defects are covered for twelve months; accidental and water damage are excluded.

Status: answered · Time: 1.50s

The electronics warranty lasts twelve months and covers manufacturing defects.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/warranty: Electronics have a twelve-month warranty covering manufacturing defects.

## q4: When is customer support available?

Reference: Monday-Friday, 09:00-17:00 UTC; closed weekends.

Status: answered · Time: 1.17s

Customer support is available Monday through Friday from 09:00 to 17:00 UTC.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/support: Customer support is available Monday through Friday from 09:00 to 17:00 UTC.

## q5: If I return an unused item, what is the return deadline and will my original delivery charge be refunded?

Reference: Thirty days; original delivery charge is not refunded for unwanted-item returns.

Status: answered · Time: 1.03s

Unused items can be returned within thirty days of purchase. The original delivery charge is not refunded when a customer returns an unwanted item.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: Customers may return unused items within thirty days of purchase.
- https://example.test/shipping: The original delivery charge is not refunded when a customer returns an unwanted item.

## q6: What proof is needed for a return and for an electronics warranty claim?

Reference: A receipt for returns and proof of purchase for warranty claims.

Status: answered · Time: 2.94s

A receipt is required to return an unused item. Proof of purchase is required to file an electronics warranty claim.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: A receipt is required.
- https://example.test/warranty: Warranty claims require proof of purchase.

## q7: Do I need a receipt for that?

Reference: Yes, a receipt is required for a return.

Status: answered · Time: 3.99s

A receipt is required to return an unused item.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: A receipt is required.

## q8: And how long does the faster option take?

Reference: Express delivery takes two business days.

Status: answered · Time: 1.68s

Express delivery takes two business days.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/shipping: Express delivery takes two business days

## q9: What is the founder's date of birth?

Reference: Abstain: no founder birth date is provided.

Status: insufficient_evidence · Time: 1.82s

I couldn’t find enough evidence in the retrieved passages to answer this question.

Checks: behavior=True, source recall=None, expected citations=None, reference terms=None


## q10: How much revenue did the company earn last year?

Reference: Abstain: no revenue is provided.

Status: insufficient_evidence · Time: 7.04s

I couldn’t find enough evidence in the retrieved passages to answer this question.

Checks: behavior=True, source recall=None, expected citations=None, reference terms=None

