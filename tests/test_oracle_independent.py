"""Unabhängige Orakel für den gewichteten Blossom: (1) Aufzählung ALLER Paarungen über Kantenmengen (anderer Rechenweg als die Bitmasken-Dynamik in
`wb_oracle`), (2) ganzzahliges Programm (`scipy.optimize.milp`) auf mittleren Graphen, (3) Soundness des Zertifikats: es darf keine zu teure
Paarung "beweisen" (Regression: ohne die Bedingung "freie Ecken tragen den kleinsten Dualwert" ging das durch)."""

import random

import numpy as np
import pytest

import wb_blossom as B
from wb_scenario import Scenario


def _scenario(n, edges):
    cost = np.zeros((n, n), dtype=np.int64)
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        cost[i, j] = cost[j, i] = w
        adj[i].append(j)
        adj[j].append(i)
    return Scenario(tuple((i, 0) for i in range(n)), 0, cost, tuple(tuple(sorted(a)) for a in adj))


def _enumerate(edges):
    """Alle Paarungen durch Rekursion über die Kantenliste: {Größe: kleinste Kosten}."""
    best = {}

    def rec(k, used, size, cost):
        if k == len(edges):
            if size not in best or cost < best[size]:
                best[size] = cost
            return
        rec(k + 1, used, size, cost)
        i, j, w = edges[k]
        if not (used >> i & 1) and not (used >> j & 1):
            rec(k + 1, used | (1 << i) | (1 << j), size + 1, cost + w)

    rec(0, 0, 0, 0)
    return best


@pytest.mark.parametrize("seed", range(150))
def test_matches_full_enumeration(seed):
    rng = random.Random(500 + seed)
    n = rng.randint(2, 9)
    wmax = rng.choice([1, 2, 3, 10, 50])
    edges = [(i, j, rng.randint(0, wmax)) for i in range(n) for j in range(i + 1, n) if rng.random() < rng.choice([0.4, 0.8])][:18]
    sc = _scenario(n, edges)
    best = _enumerate(edges)
    top = max(best)
    res = B.run(sc, maxcardinality=True, record=False)
    assert (len(res.pairs), res.cost) == (top, best[top])
    assert B.run(sc, maxcardinality=False, record=False).cost == min(best.values())


def test_matches_milp_on_medium_graphs():
    milp = pytest.importorskip("scipy.optimize").milp
    from scipy.optimize import Bounds, LinearConstraint

    rng = random.Random(77)
    for _ in range(40):
        n = rng.randint(8, 22)
        edges = [(i, j, rng.randint(1, rng.choice([3, 20, 100]))) for i in range(n) for j in range(i + 1, n) if rng.random() < rng.choice([0.15, 0.3, 0.6])]
        if not edges:
            continue
        a = np.zeros((n, len(edges)))
        for e, (i, j, _) in enumerate(edges):
            a[i, e] = a[j, e] = 1
        big = sum(w for _, _, w in edges) + 1
        kw = dict(constraints=LinearConstraint(a, -np.inf, 1), integrality=np.ones(len(edges)), bounds=Bounds(0, 1))
        x = np.round(milp(np.array([w - big for _, _, w in edges], float), **kw).x).astype(int)
        size, cost = int(x.sum()), sum(w for e, (_, _, w) in enumerate(edges) if x[e])
        sc = _scenario(n, edges)
        res = B.run(sc, maxcardinality=True, record=False)
        assert (len(res.pairs), res.cost) == (size, cost)
        assert B.run(sc, maxcardinality=False, record=False).cost == int(round(milp(np.array([w for _, _, w in edges], float), **kw).fun))


def test_certificate_rejects_a_pricier_matching():
    """Zwei getrennte Kanten (Kosten 5 und 1), gewählt ist die teure. y = (1, 1, 5, 5) ist zulässig, auf der gewählten Kante straff, die Identität geht auf -
    nur die Bedingung an die freien Ecken (y = 5 ist nicht der kleinste Dualwert) entlarvt die Paarung."""
    sc = _scenario(4, [(0, 1, 5), (2, 3, 1)])
    cert = B.certificate(sc, ((0, 1),), (1, 1, 5, 5), (), {}, maxcardinality=True)
    assert cert["feasible"] and cert["matched_tight"] and cert["identity_holds"] and not cert["free_ok"] and not cert["all_ok"]
    res = B.run(sc, maxcardinality=True, record=False)
    assert res.cost == 6 and res.count == 2
    good = B.certificate(sc, res.pairs, res.y, res.z, {}, maxcardinality=True)
    assert good["all_ok"]


@pytest.mark.parametrize("seed", range(60))
def test_certificate_passes_on_real_runs(seed):
    rng = random.Random(900 + seed)
    n = rng.randint(2, 26)
    edges = [(i, j, rng.randint(0, rng.choice([1, 5, 40]))) for i in range(n) for j in range(i + 1, n) if rng.random() < rng.choice([0.1, 0.3, 0.7])]
    sc = _scenario(n, edges)
    for mc in (True, False):
        res = B.run(sc, maxcardinality=mc, record=False)
        by_id = {b.id: b.members for b in res.blossoms}
        assert B.certificate(sc, res.pairs, res.y, res.z, by_id, maxcardinality=mc)["all_ok"]
