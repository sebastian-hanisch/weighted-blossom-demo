"""Unabhängige Gegenprobe der Vergleichsgröße "Kosten ignoriert" (`wb_evaluation.blind_result`): Sie ist eine beliebige größte Paarung, also
gleich groß wie das Optimum und zwischen der billigsten und der teuersten größten Paarung - aber NICHT die teuerste. Orakel: Brute Force über
alle Paarungen (eigene Kantenrekursion, anderer Rechenweg als das Bitmasken-DP in `wb_oracle`) und networkx auf negierten Kosten."""

import numpy as np
import pytest

import wb_constants as C
import wb_evaluation as ev
import wb_scenario as ws
from wb_blossom import run

nx = pytest.importorskip("networkx")


def _all_matchings(edges):
    out = []

    def rec(k, used, cnt, cost):
        if k == len(edges):
            out.append((cnt, cost))
            return
        rec(k + 1, used, cnt, cost)
        i, j, w = edges[k]
        if not (used >> i) & 1 and not (used >> j) & 1:
            rec(k + 1, used | (1 << i) | (1 << j), cnt + 1, cost + w)

    rec(0, 0, 0, 0)
    return out


@pytest.mark.parametrize("seed", range(60))
def test_blind_is_a_maximum_matching_between_cheapest_and_dearest(seed):
    sc = ws.generate(10, 30, 0, 300 + seed)
    edges = [(i, j, int(sc.cost[i, j])) for i in range(sc.n) for j in sc.adj[i] if i < j]
    if not edges:
        pytest.skip("keine Kante")
    allm = _all_matchings(edges)
    k = max(c for c, _ in allm)
    costs = [c for cnt, c in allm if cnt == k]
    res = run(sc, maxcardinality=True, record=False)
    blind = ev.blind_result(sc)
    assert (res.count, res.cost) == (k, min(costs))
    assert blind.count == k
    assert min(costs) <= ev.real_cost(sc, blind.pairs) <= max(costs)


def test_dearest_maximum_matching_is_well_above_blind_on_the_fixed_maps():
    """README: die teuerste größte Paarung kostet auf den 100 festen Karten im Mittel 205,4 Minuten - mehr als die Vergleichsgröße (179,72)."""
    dear, blind = [], []
    for sd in C.DIST_SEEDS:
        sc = ws.generate(30, 20, 0, sd)
        g = nx.Graph()
        for i in range(sc.n):
            for j in sc.adj[i]:
                if i < j:
                    g.add_edge(i, j, weight=-int(sc.cost[i, j]))
        m = nx.min_weight_matching(g)
        dear.append(sum(int(sc.cost[i, j]) for i, j in m))
        blind.append(ev.real_cost(sc, ev.blind_result(sc).pairs))
    assert abs(np.mean(dear) - 205.4) <= 0.05
    assert np.mean(dear) > np.mean(blind) + 20
