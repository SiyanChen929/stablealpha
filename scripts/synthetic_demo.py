"""Offline formula demonstration; synthetic data, no investment meaning."""
from cogalpha_plus.synthetic import make_synthetic_market
from cogalpha_plus.dsl import evaluate_expression, validate_expression, ValidationSpec

panel = make_synthetic_market(days=100, symbols=20, seed=7)
expr = {"op": "return", "args": [{"op": "field", "name": "close"}], "periods": 5}
spec = ValidationSpec(allowed_fields=frozenset({"close"}))
report = validate_expression(expr, spec)
values = evaluate_expression(expr, panel, spec)
print("SYNTHETIC SOFTWARE DEMO — NOT EMPIRICAL PERFORMANCE")
print(report)
print(values.dropna().head().to_string())
