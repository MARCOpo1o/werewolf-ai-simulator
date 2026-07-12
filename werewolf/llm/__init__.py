"""Provider-neutral LLM invocation and cost-accounting layer.

Modules:
    records   -- normalized usage/cost record types (schema_version 1)
    provider  -- Provider protocol and typed request/result objects
    fake_provider -- deterministic scripted provider for tests
    ledger    -- in-memory usage ledger with JSONL sink + aggregation
    registry  -- model alias registry and API-key env-var resolution

No module in this package may log, store, or expose API key material.
"""
