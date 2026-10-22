# Evaluation results

Run: 2026-10-21T13:57:02.963274+00:00 · Model: `openai/gpt-oss-20b`

- Expected answer/abstention behavior: 16/16.
- Mean expected-source recall@4: 100.0% across 13 completed answerable cases.
- Errors: 0.

This is a small synthetic smoke benchmark, not an accuracy estimate. Term matching checks coverage only; inspect answers and quotes for semantic correctness. Retrieval and citation checks use independently specified source IDs. The runtime model-based verifier is not an independent evaluator.

## q1: How long do I have to return an unused item?

Reference: Unused items can be returned within thirty days of purchase.

Status: answered · Time: 1.84s

Customers may return unused items within thirty days of purchase.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: Customers may return unused items within thirty days of purchase.

## q2: How long does standard delivery take?

Reference: Standard delivery takes five business days.

Status: answered · Time: 1.18s

Standard delivery takes five business days.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/shipping: Standard delivery takes five business days and costs five dollars.

## q3: What does the electronics warranty cover and how long does it last?

Reference: Manufacturing defects are covered for twelve months; accidental and water damage are excluded.

Status: answered · Time: 0.98s

The electronics warranty lasts twelve months and covers manufacturing defects.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/warranty: Electronics have a twelve-month warranty covering manufacturing defects.

## q4: When is customer support available?

Reference: Monday-Friday, 09:00-17:00 UTC; closed weekends.

Status: answered · Time: 1.25s

Customer support is available Monday through Friday from 09:00 to 17:00 UTC.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/support: Customer support is available Monday through Friday from 09:00 to 17:00 UTC.

## q5: If I return an unused item, what is the return deadline and will my original delivery charge be refunded?

Reference: Thirty days; original delivery charge is not refunded for unwanted-item returns.

Status: answered · Time: 1.09s

Unused items can be returned within thirty days of purchase. The original delivery charge is not refunded when a customer returns an unwanted item.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: Customers may return unused items within thirty days of purchase.
- https://example.test/shipping: The original delivery charge is not refunded when a customer returns an unwanted item.

## q6: What proof is needed for a return and for an electronics warranty claim?

Reference: A receipt for returns and proof of purchase for warranty claims.

Status: answered · Time: 1.19s

A receipt is required to return an item. Proof of purchase is required for an electronics warranty claim.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: A receipt is required.
- https://example.test/warranty: Warranty claims require proof of purchase.

## q7: Do I need a receipt for that?

Reference: Yes, a receipt is required for a return.

Status: answered · Time: 6.02s

A receipt is required to return an unused item.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: A receipt is required.

## q8: And how long does the faster option take?

Reference: Express delivery takes two business days.

Status: answered · Time: 14.10s

Express delivery takes two business days.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/shipping: Express delivery takes two business days and costs twelve dollars.

## q9: What is the founder's date of birth?

Reference: Abstain: no founder birth date is provided.

Status: insufficient_evidence · Time: 6.60s

I couldn’t find enough evidence in the retrieved passages to answer this question.

Checks: behavior=True, source recall=None, expected citations=None, reference terms=None


## q10: How much revenue did the company earn last year?

Reference: Abstain: no revenue is provided.

Status: insufficient_evidence · Time: 6.93s

I couldn’t find enough evidence in the retrieved passages to answer this question.

Checks: behavior=True, source recall=None, expected citations=None, reference terms=None


## h1: Are accidental damage and water damage covered by the electronics warranty?

Reference: No. Accidental damage and water damage are excluded.

Status: answered · Time: 6.51s

Accidental damage and water damage are excluded from the electronics warranty.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/warranty: Accidental damage and water damage are excluded.

## h2: Will my original delivery charge be refunded if I return a damaged product?

Reference: Abstain. The shipping policy only specifies unwanted-item returns, not damaged-product returns.

Status: insufficient_evidence · Time: 11.43s

I couldn’t find enough evidence in the retrieved passages to answer this question.

Checks: behavior=True, source recall=None, expected citations=None, reference terms=None


## h3: How many points do members earn per dollar, and do those points expire?

Reference: One point per dollar spent on eligible items. Points expire after twelve months of inactivity.

Status: answered · Time: 13.37s

Members earn one reward point per dollar spent on eligible items. Reward points expire after twelve months of inactivity.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/loyalty: Members earn one reward point per dollar spent on eligible items.
- https://example.test/loyalty: Points expire after twelve months of inactivity.

## h4: Can I exchange reward points for cash?

Reference: No, reward points cannot be exchanged for cash.

Status: answered · Time: 11.15s

Reward points cannot be exchanged for cash.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/loyalty: Reward points cannot be exchanged for cash.

## h5: How long do approved refunds take, and which payment method receives the refund?

Reference: Within seven business days after inspection, to the original payment method.

Status: answered · Time: 7.08s

Approved refunds are issued to the original payment method within seven business days after inspection.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: Approved refunds are issued to the original payment method within seven business days after inspection.

## h6: If I have a receipt, can I return a final-sale product?

Reference: No. Final-sale products cannot be returned even if the customer has a receipt.

Status: answered · Time: 5.85s

Final‑sale products cannot be returned, even if a receipt is presented.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=False

- https://example.test/returns: Final-sale products cannot be returned.
