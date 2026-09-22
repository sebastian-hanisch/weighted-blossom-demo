"""Blossom (gewichtet) gegen unabhängige Prüfer: networkx, scipy (zweiseitig), Brute Force, feste Fallstrick-Reproduzenten,
Zertifikat (inkl. Negativkontrolle) und die Ereignis-/Schnappschuss-Architektur. Siehe `wb_blossom.py`s Moduldoku für das
Id-Schema und die zwei dokumentierten Abweichungen von der ursprünglichen Vorgabe (Ersatzgewicht bei maxcardinality=False,
Vorzeichen/Summationsbereich im Zertifikat)."""

import random

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import wb_blossom as B
import wb_oracle as O
import wb_scenario as ws
from wb_scenario import Scenario

DIST_SEEDS = range(100000, 100100)


# --- Hilfsfunktionen ----------------------------------------------------------------------------------------------------

def graph_scenario(n, adj, cost):
    c = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in adj[i]:
            c[i, j] = cost[i][j]
    return Scenario(tuple((i, 0) for i in range(n)), 0, c, tuple(tuple(sorted(a)) for a in adj))


def gnp(n, p, wmax, rng):
    """Zufallsgraph G(n, p), ganzzahlige Gewichte 1..wmax (Python-`random`, nicht `SplitMix64` - hier reicht Reproduzierbarkeit
    über einen expliziten `random.Random`-Seed je Testlauf, wie bei `blossom-demo`s eigenem `gnp`)."""
    adj = [[] for _ in range(n)]
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.randint(1, wmax)
                adj[i].append(j)
                adj[j].append(i)
                cost[i][j] = cost[j][i] = w
    return graph_scenario(n, adj, cost)


def nx_graph(sc):
    g = nx.Graph()
    g.add_nodes_from(range(sc.n))
    for i in range(sc.n):
        for j in sc.adj[i]:
            if i < j:
                g.add_edge(i, j, weight=int(sc.cost[i, j]))
    return g


def nx_min_cost(sc, maxcardinality):
    """Ground Truth über networkx, in derselben Konvention wie `wb_blossom.run`: maxcardinality=True -> `min_weight_matching`
    (immer maximale Kardinalität, dann kleinste Kosten); maxcardinality=False -> `max_weight_matching` auf negierten Gewichten
    (reine Kostenminimierung ohne Nebenbedingung an die Anzahl)."""
    g = nx_graph(sc)
    if g.number_of_edges() == 0:
        return 0, 0
    if maxcardinality:
        m = nx.min_weight_matching(g)
    else:
        g2 = nx.Graph()
        g2.add_nodes_from(range(sc.n))
        for i, j, d in g.edges(data=True):
            g2.add_edge(i, j, weight=-d["weight"])
        m = nx.max_weight_matching(g2, maxcardinality=False, weight="weight")
    return len(m), sum(int(sc.cost[i, j]) for i, j in m)


def is_matching(sc, pairs):
    used = set()
    for i, j in pairs:
        if j not in sc.adj[i] or i in used or j in used:
            return False
        used |= {i, j}
    return True


def cert_of(sc, res):
    by_id = {b.id: b.members for b in res.blossoms}
    return B.certificate(sc, res.pairs, res.y, res.z, by_id, maxcardinality=res.maxcardinality)


def wgraph_scenario(n, edges):
    cost = np.zeros((n, n), dtype=np.int64)
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        cost[i, j] = w
        cost[j, i] = w
        adj[i].append(j)
        adj[j].append(i)
    return Scenario(tuple((i, 0) for i in range(n)), 0, cost, tuple(tuple(sorted(a)) for a in adj))


# --- networkx-Gegenprobe: 300+ Zufallsgraphen -----------------------------------------------------------------------------

RANDOM_CASES = []
for _n in list(range(2, 41)):
    for _p in (0.15, 0.3, 0.5, 0.75, 1.0):
        for _wmax in (1, 5, 50):
            RANDOM_CASES.append((_n, _p, _wmax))
# ein paar explizit leere und vollständig isolierte Graphen dazu
RANDOM_CASES += [(5, 0.0, 1), (12, 0.0, 5), (1, 0.0, 1), (0, 0.0, 1)]
assert len(RANDOM_CASES) >= 300, len(RANDOM_CASES)


@pytest.mark.parametrize("n,p,wmax", RANDOM_CASES)
@pytest.mark.parametrize("maxcardinality", [True, False])
def test_random_graphs_match_networkx(n, p, wmax, maxcardinality):
    rng = random.Random((n, round(p * 100), wmax, maxcardinality).__hash__() & 0xFFFFFFFF)
    sc = gnp(n, p, wmax, rng)
    res = B.run(sc, maxcardinality=maxcardinality, record=False)
    assert is_matching(sc, res.pairs)
    nx_size, nx_cost = nx_min_cost(sc, maxcardinality)
    assert len(res.pairs) == nx_size
    assert res.cost == nx_cost
    cert = cert_of(sc, res)
    assert cert["all_ok"], cert


# --- Brute Force (Orakel) für kleine n --------------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(60))
def test_small_graphs_match_brute_force(trial):
    rng = random.Random(9000 + trial)
    n = rng.randint(2, 12)
    p = rng.uniform(0.2, 1.0)
    sc = gnp(n, p, rng.choice((1, 3, 20)), rng)
    adjm = tuple(sc.adj)
    res_true = B.run(sc, maxcardinality=True, record=False)
    bsize, bcost = O.max_weight_matching_brute(sc.cost, adjm)
    assert len(res_true.pairs) == bsize
    assert res_true.cost == bcost
    res_false = B.run(sc, maxcardinality=False, record=False)
    bcost_nc = O.min_weight_matching_brute(sc.cost, adjm)
    assert res_false.cost == bcost_nc


# --- Fahrgemeinschaften-Geometrie: DIST_SEEDS + ein paar (n, reach) -----------------------------------------------------

@pytest.mark.parametrize("seed", DIST_SEEDS)
def test_geometric_dist_seeds(seed):
    sc = ws.generate(30, 20, 0, seed)
    res_true = B.run(sc, maxcardinality=True, record=False)
    nx_size, nx_cost = nx_min_cost(sc, True)
    assert len(res_true.pairs) == nx_size and res_true.cost == nx_cost
    assert cert_of(sc, res_true)["all_ok"]

    res_false = B.run(sc, maxcardinality=False, record=False)
    nx_size_f, nx_cost_f = nx_min_cost(sc, False)
    assert res_false.cost == nx_cost_f and len(res_false.pairs) == nx_size_f
    # entartet: leer ist immer mindestens so gut wie jede andere Paarung (alle Kosten >= 0); bei einer zufälligen
    # Nullkosten-Kante (zwei deckungsgleiche Punkte) ist ein Paar mit Kosten 0 gleichwertig - Kardinalität dann nicht
    # zwingend 0, Kosten aber immer 0.
    assert res_false.cost == 0
    assert cert_of(sc, res_false)["all_ok"]


@pytest.mark.parametrize("n,reach", [(10, 15), (20, 12), (40, 10), (15, 60)])
def test_geometric_other_sizes(n, reach):
    for seed in (1, 2, 3):
        sc = ws.generate(n, reach, 0, seed)
        for maxcard in (True, False):
            res = B.run(sc, maxcardinality=maxcard, record=False)
            nx_size, nx_cost = nx_min_cost(sc, maxcard)
            assert len(res.pairs) == nx_size and res.cost == nx_cost
            assert cert_of(sc, res)["all_ok"]


def test_maxcardinality_degeneracy_seed_100000():
    """Die im Auftrag vorausberechnete Zahl: `generate(30, 20, 0, 100000)` - maxcardinality=True 14 Paare/171 Minuten,
    maxcardinality=False 0 Paare/0 Minuten (keine Nullkosten-Kante auf dieser Karte, s. Bericht)."""
    sc = ws.generate(30, 20, 0, 100000)
    res_true = B.run(sc, maxcardinality=True, record=False)
    assert (res_true.count, res_true.cost) == (14, 171)
    res_false = B.run(sc, maxcardinality=False, record=False)
    assert (res_false.count, res_false.cost) == (0, 0)


# --- zweiseitig gegen scipy ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k,m,reach", [(5, 5, 30), (7, 9, 35), (9, 7, 40), (4, 10, 45), (9, 9, 40)])
@pytest.mark.parametrize("seed", range(20))
def test_bipartite_matches_scipy(k, m, reach, seed):
    sc = ws.bipartite(k, m, reach, 0, seed)
    for i in range(sc.n):
        for j in sc.adj[i]:
            assert (i < k) != (j < k), "bipartite: keine Kante innerhalb einer Gruppe"
    res = B.run(sc, maxcardinality=True, record=False)
    BIG = 10 ** 6
    cm = np.full((k, m), BIG, dtype=np.int64)
    for i in range(k):
        for j in sc.adj[i]:
            cm[i, j - k] = sc.cost[i, j]
    row, col = linear_sum_assignment(cm)
    chosen = [(i, j) for i, j in zip(row, col) if cm[i, j] < BIG]
    assert len(chosen) == res.count
    assert sum(int(cm[i, j]) for i, j in chosen) == res.cost
    assert cert_of(sc, res)["all_ok"]


def test_bipartite_default_net():
    sc = ws.build("bipartit", 0, 0, 0, 0)
    for i in range(sc.n):
        for j in sc.adj[i]:
            assert (i < 9) != (j < 9)
    res = B.run(sc, maxcardinality=True, record=False)
    assert res.count > 0
    assert cert_of(sc, res)["all_ok"]


# --- Fallstrick-Reproduzenten (aus dem Auftrag) --------------------------------------------------------------------------

def test_pitfall_mybestedges_k4():
    """#1: mybestedges-Konsolidierung beim Kontrahieren. K4, Kosten wie im Auftrag angegeben. ABWEICHUNG: die im Auftrag
    genannte Zielzahl ("muss 37 ergeben") und die als Fehlersymptom beschriebene Zahl (33, 'zu hohe Kosten') sind vertauscht -
    von den drei perfekten Paarungen in K4 ({0,1}+{2,3}=33, {0,2}+{1,3}=37, {0,3}+{1,2}=33) ist 33 nachweislich das Minimum
    (von `networkx.min_weight_matching` unabhängig bestätigt), 37 wäre die zu teure, falsche Antwort einer fehlenden
    mybestedges-Konsolidierung. Getestet wird deshalb: Ergebnis == 33 (NICHT 37)."""
    edges = [(0, 1, 3), (0, 2, 13), (0, 3, 11), (1, 2, 22), (1, 3, 24), (2, 3, 30)]
    sc = wgraph_scenario(4, edges)
    res = B.run(sc, maxcardinality=True, record=False)
    assert res.cost == 33
    assert res.cost != 37
    nxm = nx.min_weight_matching(nx.Graph([(i, j, {"weight": w}) for i, j, w in edges]))
    assert sum(w for i, j, w in edges if (i, j) in nxm or (j, i) in nxm) == 33
    assert cert_of(sc, res)["all_ok"]


def test_pitfall_inphase_expansion_n8():
    """#3: Aufklappen mitten in einer Stufe (delta4, endstage=False) muss wirklich passieren, nicht nur am Stufenende.
    n=8-Graph aus dem Auftrag. ABWEICHUNG: unter der Minimierungs-Konvention dieses Moduls (Ersatzgewicht (maxcost+1)-cost)
    löst genau DIESE Kostenbelegung auf DIESEM Graphen keine Aufklappung mitten in der Stufe aus (0 `expand`-Ereignisse,
    unabhängig mit einer instrumentierten Kopie von `networkx.max_weight_matching` auf denselben transformierten Gewichten
    nachgeprüft: auch dort feuert `expandBlossom(b, False)` dort kein einziges Mal). Direkt (`nx.max_weight_matching` MAXIMIERT
    die angegebenen Zahlen statt sie zu minimieren) feuert es dagegen genau einmal (ebenfalls mit derselben instrumentierten
    Kopie nachgeprüft) - reproduziert hier über die kostenkomplementären Gewichte `(maxcost+1) - w`, die `wb_blossom.run`s
    Minimierung auf denselben Dualverlauf wie die direkte Maximierung zwingen."""
    edges = [(0, 1, 16), (1, 2, 6), (1, 4, 20), (1, 5, 18), (1, 6, 11), (2, 3, 1), (2, 6, 1), (3, 7, 13), (4, 5, 19), (4, 6, 7)]
    sc = wgraph_scenario(8, edges)
    res = B.run(sc, maxcardinality=True, record=True)
    nx_size, nx_cost = nx_min_cost(sc, True)
    assert len(res.pairs) == nx_size and res.cost == nx_cost
    assert cert_of(sc, res)["all_ok"]

    maxc = max(w for _, _, w in edges)
    edges_t = [(i, j, (maxc + 1) - w) for i, j, w in edges]
    sc_t = wgraph_scenario(8, edges_t)
    res_t = B.run(sc_t, maxcardinality=True, record=True)
    n_expand = sum(1 for e in res_t.events if e.kind == B.EV_EXPAND)
    assert n_expand >= 1, "erwartete mindestens eine In-Stufe-Aufklappung auf dem kostenkomplementären n=8-Graphen"
    assert cert_of(sc_t, res_t)["all_ok"]


def test_inphase_expansion_frequency_sample():
    """Wie oft feuert delta4 (In-Stufe-Aufklappung) über eine Zufallsstichprobe? Nur eine Zählung/ein Bericht, kein
    Bestehen/Scheitern-Kriterium (0 ist ein gültiges, wenn auch seltenes Ergebnis, s. `test_pitfall_inphase_expansion_n8`)."""
    rng = random.Random(777)
    fired = 0
    total = 300
    for _ in range(total):
        n = rng.randint(4, 40)
        p = rng.uniform(0.2, 1.0)
        sc = gnp(n, p, rng.choice((1, 5, 20, 50)), rng)
        res = B.run(sc, maxcardinality=True, record=True)
        if any(e.kind == B.EV_EXPAND for e in res.events):
            fired += 1
    print(f"\nIn-Stufe-Aufklappung (delta4) feuerte in {fired}/{total} Zufallsgraphen mindestens einmal.")
    assert fired >= 0  # reine Protokollierung


# --- Zertifikat: Negativkontrolle -----------------------------------------------------------------------------------

def test_certificate_negative_control():
    sc = ws.generate(30, 20, 0, 100000)
    res = B.run(sc, maxcardinality=True, record=False)
    assert cert_of(sc, res)["all_ok"]
    bad_y = list(res.y)
    bad_y[0] += 5
    by_id = {b.id: b.members for b in res.blossoms}
    cert = B.certificate(sc, res.pairs, tuple(bad_y), res.z, by_id, maxcardinality=True)
    assert not cert["all_ok"]


def test_certificate_z_negative_control():
    sc = ws.generate(30, 20, 0, 100010)
    res = B.run(sc, maxcardinality=True, record=False)
    if not res.z:
        pytest.skip("keine Blüte am Ende auf dieser Karte")
    by_id = {b.id: b.members for b in res.blossoms}
    bad_z = tuple((bid, -abs(z) - 1) for bid, z in res.z)
    cert = B.certificate(sc, res.pairs, res.y, bad_z, by_id, maxcardinality=True)
    assert not cert["all_ok"] and not cert["z_nonneg"]


# --- Ereignis-/Schnappschuss-Architektur -----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(100000, 100010)))
def test_events_snapshots_sane(seed):
    sc = ws.generate(30, 20, 0, seed)
    res = B.run(sc, maxcardinality=True, record=True)
    assert len(res.snapshots) == len(res.events) + 1

    m = [-1] * sc.n
    for i, j in res.pairs:
        m[i], m[j] = j, i
    assert res.snapshots[-1].match == tuple(m)

    for k, e in enumerate(res.events, start=1):
        if e.kind != B.EV_AUGMENT:
            continue
        before = res.snapshots[k - 1].match
        path = e.path
        assert len(path) >= 2 and len(path) % 2 == 0 and len(set(path)) == len(path)
        assert before[path[0]] == -1 and before[path[-1]] == -1, "beide Enden waren unmittelbar vorher frei"
        for kk in range(len(path) - 1):
            a, c = path[kk], path[kk + 1]
            assert c in sc.adj[a], "nur echte Kanten"
            if kk % 2 == 1:
                assert before[a] == c, "ungerade Kante war gewählt"
            else:
                assert before[a] != c, "gerade Kante war nicht gewählt"


@pytest.mark.parametrize("seed", list(range(100000, 100010)))
def test_record_false_matches_record_true(seed):
    sc = ws.generate(30, 20, 0, seed)
    rT = B.run(sc, maxcardinality=True, record=True)
    rF = B.run(sc, maxcardinality=True, record=False)
    assert rT.pairs == rF.pairs
    assert rT.cost == rF.cost
    assert rT.scanned_total == rF.scanned_total
    assert rT.augmentations == rF.augmentations
    assert rT.contractions == rF.contractions
    assert rT.contracted_vertices == rF.contracted_vertices
    assert rT.expansions == rF.expansions
    assert rT.y == rF.y
    assert rT.z == rF.z


def test_state_at_roundtrip():
    sc = ws.generate(30, 20, 0, 100000)
    res = B.run(sc, maxcardinality=True, record=True)
    for k in (0, len(res.events) // 2, len(res.events)):
        snap, blossoms = B.state_at(res, k)
        assert snap == res.snapshots[k]
        assert {b.id for b in blossoms} == set(snap.blossoms)


def test_empty_and_trivial_scenarios():
    sc0 = graph_scenario(0, [], [])
    res0 = B.run(sc0, maxcardinality=True, record=True)
    assert res0.pairs == () and res0.cost == 0
    assert cert_of(sc0, res0)["all_ok"]

    sc1 = graph_scenario(3, [[], [], []], [[0, 0, 0]] * 3)
    res1 = B.run(sc1, maxcardinality=True, record=True)
    assert res1.pairs == () and res1.cost == 0
    assert cert_of(sc1, res1)["all_ok"]
