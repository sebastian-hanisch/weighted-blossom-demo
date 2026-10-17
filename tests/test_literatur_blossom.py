"""Literatur-Referenzwerte für den gewichteten Blossom (ORACLE-PLAYBOOK §5.1).

Quellen (alle am 2026-10-17 eingesehen; Werte stammen aus der Quelle, nicht aus dem Code der Demo):

1. Referenzsoftware NetworkX, Testsuite `networkx/algorithms/tests/test_matching.py`, Klasse `TestMaxWeightMatching`
   (installierte Version networkx 3.7 im Test-venv, Datei gelesen; dieselben Zahlen über
   https://raw.githubusercontent.com/networkx/networkx/main/networkx/algorithms/tests/test_matching.py abgerufen).
   Die Fälle `test_s_blossom`, `test_s_t_blossom`, `test_nested_s_blossom`, `test_nested_s_blossom_relabel`,
   `test_nested_s_blossom_expand`, `test_s_blossom_relabel_expand`, `test_nested_s_blossom_relabel_expand`, `test_nasty_blossom1`,
   `test_nasty_blossom2`, `test_nasty_blossom_least_slack`, `test_nasty_blossom_augmenting`, `test_nasty_blossom_expand_recursively`
   (Kantenlisten mit Gewichten und die dort erwartete Paarung `answer`) sind Blüten-Sonderfälle (S-/T-Blüte, verschachtelt,
   Aufklappen); `test_negative_weights` und `test_min_weight_matching_max_cardinality` (gh-8062) liefern die Konventionen
   "größtes Gewicht", "größtes Gewicht bei größter Paarzahl" und "kleinstes Gewicht bei größter Paarzahl".
   Übersetzung für die Demo (Test-seitig, keine Quelle): networkx maximiert Gewichte w, die Demo minimiert Kosten. Mit
   Kosten = -w minimiert `run(maxcardinality=False)` genau -w(M), also maximiert w(M) wie `max_weight_matching`; mit
   `maxcardinality=True` entspricht es `max_weight_matching(maxcardinality=True)`. `min_weight_matching` ist laut networkx-Docstring das
   kleinste Gewicht unter allen Paarungen größter Paarzahl, also die Demo mit Kosten = w und `maxcardinality=True`.
   Ecken der Quelle sind ab 1 nummeriert, die Demo ab 0 (Test zieht 1 ab).
2. SciPy-Dokumentation `scipy.optimize.linear_sum_assignment`, Abschnitt Examples,
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html: Kostenmatrix
   [[4, 1, 3], [2, 0, 5], [3, 2, 2]], `col_ind = [1, 0, 2]`, Gesamtkosten 5. Die Matrix wird als zweiseitiger Graph
   (Zeilen 0..2, Spalten 3..5) eingelesen; das ist der Fall "Bipartit = Ungarisch" des gewichteten Blossom.

Nicht eingesehen (kein Test): Galil, "Efficient Algorithms for Finding Maximum Matching in Graphs", ACM Computing Surveys 18 (1986),
und Edmonds, "Maximum matching and a polyhedron with 0,1-vertices", J. Res. NBS 69B (1965): beide nur als Zitat bekannt, kein
Beispiel mit Lösung eingesehen.
"""

import numpy as np
import pytest

import wb_blossom as B
from wb_scenario import Scenario


def make(n, edges):
    cost = np.zeros((n, n), dtype=np.int64)
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        cost[i, j] = cost[j, i] = w
        adj[i].append(j)
        adj[j].append(i)
    return Scenario(tuple((i, 0) for i in range(n)), 0, cost, tuple(tuple(sorted(a)) for a in adj))


def solve(weighted_edges, maxcardinality, negate):
    """Kanten (u, v, w) mit Ecken ab 1 -> Demo-Karte; negate=True: Kosten = -w, sonst Kosten = w. Rückgabe: Paarung als Menge von Paaren (ab 1)."""
    n = max(max(u, v) for u, v, _ in weighted_edges)
    sign = -1 if negate else 1
    sc = make(n, [(u - 1, v - 1, sign * w) for u, v, w in weighted_edges])
    res = B.run(sc, maxcardinality=maxcardinality)
    return {tuple(sorted((i + 1, j + 1))) for i, j in res.pairs}


def weight(edges, pairs):
    w = {tuple(sorted((u, v))): x for u, v, x in edges}
    return sum(w[p] for p in pairs)


# Kantenlisten und erwartete Paarungen aus networkx TestMaxWeightMatching (Quelle 1)
S_BLOSSOM_A = ([(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)], {(1, 2), (3, 4)})
S_BLOSSOM_B = ([(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7), (1, 6, 5), (4, 5, 6)], {(1, 6), (2, 3), (4, 5)})
S_T_A = ([(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 4), (1, 6, 3)], {(1, 6), (2, 3), (4, 5)})
S_T_B = ([(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 3), (1, 6, 4)], {(1, 6), (2, 3), (4, 5)})
S_T_C = ([(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 3), (3, 6, 4)], {(1, 2), (3, 6), (4, 5)})
NESTED = ([(1, 2, 9), (1, 3, 9), (2, 3, 10), (2, 4, 8), (3, 5, 8), (4, 5, 10), (5, 6, 6)], {(1, 3), (2, 4), (5, 6)})
NESTED_RELABEL = ([(1, 2, 10), (1, 7, 10), (2, 3, 12), (3, 4, 20), (3, 5, 20), (4, 5, 25), (5, 6, 10), (6, 7, 10), (7, 8, 8)],
                  {(1, 2), (3, 4), (5, 6), (7, 8)})
NESTED_EXPAND = ([(1, 2, 8), (1, 3, 8), (2, 3, 10), (2, 4, 12), (3, 5, 12), (4, 5, 14), (4, 6, 12), (5, 7, 12), (6, 7, 14), (7, 8, 12)],
                 {(1, 2), (3, 5), (4, 6), (7, 8)})
S_RELABEL_EXPAND = ([(1, 2, 23), (1, 5, 22), (1, 6, 15), (2, 3, 25), (3, 4, 22), (4, 5, 25), (4, 8, 14), (5, 7, 13)],
                    {(1, 6), (2, 3), (4, 8), (5, 7)})
NESTED_RELABEL_EXPAND = ([(1, 2, 19), (1, 3, 20), (1, 8, 8), (2, 3, 25), (2, 4, 18), (3, 5, 18), (4, 5, 13), (4, 7, 7), (5, 6, 7)],
                         {(1, 8), (2, 3), (4, 7), (5, 6)})
NASTY1 = ([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 35), (5, 7, 26), (9, 10, 5)],
          {(1, 6), (2, 3), (4, 8), (5, 7), (9, 10)})
NASTY2 = ([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 26), (5, 7, 40), (9, 10, 5)],
          {(1, 6), (2, 3), (4, 8), (5, 7), (9, 10)})
NASTY_LEAST_SLACK = ([(1, 2, 45), (1, 5, 45), (2, 3, 50), (3, 4, 45), (4, 5, 50), (1, 6, 30), (3, 9, 35), (4, 8, 28), (5, 7, 26), (9, 10, 5)],
                     {(1, 6), (2, 3), (4, 8), (5, 7), (9, 10)})
NASTY_AUGMENTING = ([(1, 2, 45), (1, 7, 45), (2, 3, 50), (3, 4, 45), (4, 5, 95), (4, 6, 94), (5, 6, 94), (6, 7, 50), (1, 8, 30), (3, 11, 35),
                     (5, 9, 36), (7, 10, 26), (11, 12, 5)],
                    {(1, 8), (2, 3), (4, 6), (5, 9), (7, 10), (11, 12)})
NASTY_RECURSIVE = ([(1, 2, 40), (1, 3, 40), (2, 3, 60), (2, 4, 55), (3, 5, 55), (4, 5, 50), (1, 8, 15), (5, 7, 30), (7, 6, 10), (8, 10, 10), (4, 9, 30)],
                   {(1, 2), (3, 5), (4, 9), (6, 7), (8, 10)})

ALL_MAX_WEIGHT = {
    "s_blossom_a": S_BLOSSOM_A, "s_blossom_b": S_BLOSSOM_B, "s_t_a": S_T_A, "s_t_b": S_T_B, "s_t_c": S_T_C, "nested": NESTED,
    "nested_relabel": NESTED_RELABEL, "nested_expand": NESTED_EXPAND, "s_relabel_expand": S_RELABEL_EXPAND,
    "nested_relabel_expand": NESTED_RELABEL_EXPAND, "nasty1": NASTY1, "nasty2": NASTY2, "nasty_least_slack": NASTY_LEAST_SLACK,
    "nasty_augmenting": NASTY_AUGMENTING, "nasty_recursive": NASTY_RECURSIVE,
}


@pytest.mark.parametrize("name", sorted(ALL_MAX_WEIGHT))
@pytest.mark.parametrize("maxcardinality", [False, True])
def test_networkx_max_weight_cases(name, maxcardinality):
    """Kosten = -w: die Demo liefert die in networkx erwartete Paarung größten Gewichts (alle Antworten sind perfekt, also gilt es mit
    und ohne Kardinalitätszwang)."""
    edges, answer = ALL_MAX_WEIGHT[name]
    got = solve(edges, maxcardinality=maxcardinality, negate=True)
    assert got == answer
    assert weight(edges, got) == weight(edges, answer)


@pytest.mark.parametrize("name", sorted(ALL_MAX_WEIGHT))
def test_networkx_cases_with_cardinality_first_use_min_weight_convention(name):
    """`min_weight_matching` hat in der networkx-Testsuite für dieselben Fälle dieselbe erwartete Antwort (Quelle 1); die Demo mit Kosten = w
    und Kardinalitätszwang ist deren Konvention (kleinstes Gewicht unter den Paarungen größter Paarzahl). Die Kantenliste hat in
    diesen Fällen genau eine perfekte Paarung bzw. dieselbe Antwort unter beiden Konventionen."""
    edges, answer = ALL_MAX_WEIGHT[name]
    got = solve(edges, maxcardinality=True, negate=False)
    assert got == answer


def test_networkx_negative_weights():
    """networkx `test_negative_weights`: Kanten (1,2,2) (1,3,-2) (2,3,1) (2,4,-1) (3,4,-6);
    max_weight_matching = {(1,2)}; mit maxcardinality=True {(1,3),(2,4)}; min_weight_matching = {(1,2),(3,4)}."""
    edges = [(1, 2, 2), (1, 3, -2), (2, 3, 1), (2, 4, -1), (3, 4, -6)]
    assert solve(edges, maxcardinality=False, negate=True) == {(1, 2)}
    assert solve(edges, maxcardinality=True, negate=True) == {(1, 3), (2, 4)}
    assert solve(edges, maxcardinality=True, negate=False) == {(1, 2), (3, 4)}


def test_networkx_min_weight_matching_max_cardinality_gh8062():
    """networkx `test_min_weight_matching_max_cardinality` (gh-8062): Kanten (1,2,1000) (2,3,2) (3,4,3000); das kleinste Gewicht unter ALLEN
    maximalen Paarungen wäre {(2,3)}, die Antwort der Quelle (kleinstes Gewicht unter den Paarungen größter Paarzahl) ist {(1,2),(3,4)}."""
    edges = [(1, 2, 1000), (2, 3, 2), (3, 4, 3000)]
    assert solve(edges, maxcardinality=True, negate=False) == {(1, 2), (3, 4)}
    # ohne Kardinalitätszwang ist {(2,3)} (Kosten 2) billiger als jede Paarung größter Paarzahl, die leere Paarung (0) ist es noch mehr
    assert B.run(make(4, [(0, 1, 1000), (1, 2, 2), (2, 3, 3000)]), maxcardinality=False).cost == 0


def test_scipy_linear_sum_assignment_docs_example():
    """SciPy-Doku: cost = [[4,1,3],[2,0,5],[3,2,2]] -> col_ind [1,0,2], Gesamtkosten 5; als zweiseitiger Graph (Zeilen 0..2, Spalten 3..5)."""
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    edges = [(r, 3 + c, cost[r][c]) for r in range(3) for c in range(3)]
    res = B.run(make(6, edges), maxcardinality=True)
    assert res.count == 3 and res.cost == 5
    assert sorted(res.pairs) == [(0, 4), (1, 3), (2, 5)]       # col_ind = [1, 0, 2]
