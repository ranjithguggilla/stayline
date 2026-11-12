import numpy as np

from stayline import economics as eco
from stayline.config import Costs


def test_replacement_cost():
    assert eco.replacement_cost(np.array([5000.0]), Costs(replacement_share_of_annual_pay=0.5))[0] == 30000.0


def test_net_value_arithmetic():
    costs = Costs(save_rate=0.5, cost_per_conversation=100)
    leaves = np.array([1.0, 0.0, 1.0])
    repl = np.array([1000.0, 1000.0, 2000.0])
    assert eco.net_value(leaves, repl, np.array([True, True, False]), costs) == 1000 * 0.5 - 200


def test_top_fraction_counts():
    s = np.arange(10, dtype=float)
    m = eco.top_fraction(s, 0.3)
    assert m.sum() == 3 and m[-3:].all()
    assert eco.top_fraction(s, 0).sum() == 0


def test_oracle_beats_everything(parts):
    train, _, _, _ = parts
    leaves = train.Attrition.to_numpy().astype(float)
    repl = eco.replacement_cost(train.MonthlyIncome.to_numpy())
    p = np.random.default_rng(0).random(len(train))
    rule = eco.rule_of_thumb(train, float(train.MonthlyIncome.median()))
    tab = eco.policy_table(p, repl, rule, leaves, 0.2, n_boot=20)
    assert tab.iloc[0].policy.startswith("oracle")
    assert set(tab.columns) == {"policy", "net_per_1000", "lo", "hi"}


def test_break_even_is_where_value_is_zero():
    leaves = np.array([1.0, 0.0, 1.0, 0.0])
    repl = np.full(4, 10000.0)
    score = np.array([0.9, 0.1, 0.8, 0.2])
    costs = Costs(cost_per_conversation=500)
    be = eco.break_even_save_rate(score, repl, leaves, 0.5, costs)
    sel = eco.top_fraction(score, 0.5)
    assert abs(eco.net_value(leaves, repl, sel, costs, save_rate=be)) < 1e-6


def test_budget_curve_endpoints():
    n = 20
    c = eco.budget_curve(np.arange(n, dtype=float), np.full(n, 1000.0), np.zeros(n), [0.5, 1.0])
    assert c.net_per_1000.iloc[1] < c.net_per_1000.iloc[0] < 0


def test_rule_of_thumb_range(parts):
    train = parts[0]
    r = eco.rule_of_thumb(train, 5000)
    assert r.min() >= 0 and r.max() <= 5
