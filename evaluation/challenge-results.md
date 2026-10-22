# Evaluation results

Run: 2026-10-21T13:51:30.567909+00:00 · Model: `openai/gpt-oss-20b`

- Expected answer/abstention behavior: 6/6.
- Mean expected-source recall@4: 100.0% across 5 completed answerable cases.
- Errors: 0.

This is a small synthetic smoke benchmark, not an accuracy estimate. Term matching checks coverage only; inspect answers and quotes for semantic correctness. Retrieval and citation checks use independently specified source IDs. The runtime model-based verifier is not an independent evaluator.

## h1: Are accidental damage and water damage covered by the electronics warranty?

Reference: No. Accidental damage and water damage are excluded.

Status: answered · Time: 1.60s

Accidental damage and water damage are excluded from the electronics warranty.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/warranty: Accidental damage and water damage are excluded.

## h2: Will my original delivery charge be refunded if I return a damaged product?

Reference: Abstain. The shipping policy only specifies unwanted-item returns, not damaged-product returns.

Status: insufficient_evidence · Time: 0.49s

I couldn’t find enough evidence in the retrieved passages to answer this question.

Checks: behavior=True, source recall=None, expected citations=None, reference terms=None


## h3: How many points do members earn per dollar, and do those points expire?

Reference: One point per dollar spent on eligible items. Points expire after twelve months of inactivity.

Status: answered · Time: 1.43s

Members earn one reward point per dollar spent on eligible items. Reward points expire after twelve months of inactivity.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/loyalty: Members earn one reward point per dollar spent on eligible items.
- https://example.test/loyalty: Points expire after twelve months of inactivity.

## h4: Can I exchange reward points for cash?

Reference: No, reward points cannot be exchanged for cash.

Status: answered · Time: 1.46s

Reward points cannot be exchanged for cash.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/loyalty: Reward points cannot be exchanged for cash.

## h5: How long do approved refunds take, and which payment method receives the refund?

Reference: Within seven business days after inspection, to the original payment method.

Status: answered · Time: 2.48s

Approved refunds are issued to the original payment method within seven business days after inspection.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=True

- https://example.test/returns: Approved refunds are issued to the original payment method within seven business days after inspection.

## h6: If I have a receipt, can I return a final-sale product?

Reference: No. Final-sale products cannot be returned even if the customer has a receipt.

Status: answered · Time: 10.03s

Final‑sale products cannot be returned, even if a receipt is provided.

Checks: behavior=True, source recall=1.0, expected citations=True, reference terms=False

- https://example.test/returns: Final-sale products cannot be returned.
