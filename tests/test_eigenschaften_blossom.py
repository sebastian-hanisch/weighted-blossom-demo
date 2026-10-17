"""Eigenschaftstests (metamorph) für den gewichteten Blossom: Aussagen, die für das Verfahren gelten MÜSSEN, ohne dass man die richtige
Paarung kennt (ORACLE-PLAYBOOK §5.2, Verfahrenstyp Optimierung/Matching). Die vorhandenen Orakel-Tests vergleichen das Ergebnis mit
networkx, Brute Force, scipy und einem MILP; hier wird die EINGABE verändert und nur die Beziehung zwischen den beiden Ausgaben geprüft:

1. Umnummerieren der Fahrer (zufällige Permutation, auch die echten Karten und die festen Blüten-Karten über `ws.relabel`) ändert
   (Paarzahl, Kosten) nicht; die zurückübersetzte Paarung ist in der Originalkarte zulässig und gleich teuer.
2. Alle Kosten mit c > 0 multipliziert: dieselbe Paarzahl, Kosten genau c-fach, und die Paarung der skalierten Karte ist auf den
   Originalkosten ebenfalls optimal (beide Zielfunktionen).
3. Alle Kosten um k verschoben (nur lexikografisch / maxcardinality=True, weil dort jede Paarung gleich viele Kanten hat):
   Kosten + k * Paarzahl, Paarzahl gleich, Paarung auf den Originalkosten optimal.
4. Eine zusätzliche zulässige Kante verschlechtert das Optimum nie (lexikografisch: Paarzahl nicht kleiner, bei gleicher Paarzahl Kosten
   nicht größer; ohne Kardinalitätszwang: Kosten nicht größer).
5. Kantenänderungen: Streichen oder Verteuern einer NICHT gewählten Kante ändert den Optimalwert nicht; Verbilligen einer gewählten
   Kante um d senkt die Kosten genau um d (Paarzahl gleich).
6. Disjunkte Vereinigung zweier Karten: Paarzahl und Kosten addieren sich (die Blüten beider Teile stören sich nicht).
7. Primal-dual-Invarianten in JEDEM Zwischenzustand: Blütendualwerte >= 0, alle Kanten dualzulässig (Schlupf >= 0), gewählte Kanten
   straff, der Dualzielwert fällt nie, und er ist eine obere Schranke für jede Paarung (Schwach-Dualität), also auch für das
   Brute-Force-Optimum.
8. Randfälle: leere Karte, eine Ecke, keine Kante, ein einzelner Weg und gleichstandsreiche Karten (alle Kosten gleich).

Annahmen: Kosten sind ganzzahlig; die Aussagen 2 und 3 gelten für den Optimalwert, nicht für die gewählte Paarung (bei Gleichstand
darf eine andere Optimalpaarung herauskommen, deshalb werden nur Werte verglichen und die Paarung neu bewertet).
"""

import random

import numpy as np
import pytest

import wb_blossom as B
import wb_oracle as O
import wb_scenario as ws
from wb_scenario import Scenario

# --- Hilfsfunktionen ---------------------------------------------------------------------------------------------------------


def make(n, edges):
    """Karte aus Kantenliste (i, j, Kosten); Koordinaten sind für den Löser belanglos."""
    cost = np.zeros((n, n), dtype=np.int64)
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        cost[i, j] = cost[j, i] = w
        adj[i].append(j)
        adj[j].append(i)
    return Scenario(tuple((i, 0) for i in range(n)), 0, cost, tuple(tuple(sorted(a)) for a in adj))


def edges_of(sc):
    return [(i, j, int(sc.cost[i, j])) for i in range(sc.n) for j in sc.adj[i] if i < j]


def rand_edges(rng, n, p, wmax):
    return [(i, j, rng.randint(1, wmax)) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def value(sc, maxcard=True):
    r = B.run(sc, maxcardinality=maxcard, record=False)
    return r.count, r.cost


def cost_of(sc, pairs):
    return sum(int(sc.cost[i, j]) for i, j in pairs)


def valid(sc, pairs):
    used = set()
    for i, j in pairs:
        if j not in sc.adj[i] or i in used or j in used:
            return False
        used |= {i, j}
    return True


def instances(seed, count, nmin=2, nmax=10):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        n = rng.randint(nmin, nmax)
        out.append((n, rand_edges(rng, n, rng.choice([0.25, 0.5, 0.9]), rng.choice([1, 2, 3, 9, 40])), rng))
    return out


# --- 1. Umnummerieren ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("maxcard", [True, False])
def test_relabeling_does_not_change_value(maxcard):
    for n, edges, rng in instances(11, 40):
        sc = make(n, edges)
        order = list(range(n))
        rng.shuffle(order)           # neue Ecke k = alte Ecke order[k]
        new_of_old = {old: new for new, old in enumerate(order)}
        sc2 = make(n, [(new_of_old[i], new_of_old[j], w) for i, j, w in edges])
        r1 = B.run(sc, maxcardinality=maxcard, record=False)
        r2 = B.run(sc2, maxcardinality=maxcard, record=False)
        assert (r1.count, r1.cost) == (r2.count, r2.cost)
        back = [(order[i], order[j]) for i, j in r2.pairs]
        assert valid(sc, back) and cost_of(sc, back) == r1.cost
        assert (r1.count, r1.cost) == O.max_weight_matching_brute(sc.cost, sc.adj) or not maxcard


def test_relabeling_real_maps_and_fixed_blossom_maps():
    """Echte Fahrgemeinschaftskarten (viele Gleichstände durch aufgerundete Fahrzeiten) und die festen Blüten-Karten."""
    for seed in range(30):
        n = 6 + seed % 7
        sc = ws.generate(n, 25 + seed % 20, 0, 7000 + seed)
        sc2 = ws.relabel(sc, ws.perm(n, seed))
        assert value(sc) == value(sc2)
        assert value(sc, False) == value(sc2, False)
    for name, build in ws.NETS.items():
        sc = build()
        ref = value(sc)
        contracted = 0
        for seed in range(10):
            order = ws.perm(sc.n, seed)
            inv = {old: new for new, old in enumerate(order)}
            # eigenes Umnummerieren über die Kantenliste: `ws.relabel` baut die Nachbarschaft aus den Punkten neu auf und würde die durch
            # Konstruktion zweiseitige Karte ("bipartit") um Kanten innerhalb der Gruppen erweitern (Hilfsfunktion, in der App unbenutzt)
            sc2 = make(sc.n, [(inv[i], inv[j], w) for i, j, w in edges_of(sc)])
            res = B.run(sc2, maxcardinality=True)
            assert (res.count, res.cost) == ref, (name, seed)
            contracted += res.contractions
        if name in ("verschachtelt", "windmuehle"):
            # Die Karte erzwingt je nach Nummerierung Kontraktionen; der Wert bleibt davon unberührt (Schrumpfung ändert den Wert nicht)
            assert contracted > 0, name


# --- 2. Skalierung ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("maxcard", [True, False])
def test_scaling_costs_scales_value_and_keeps_optimality(maxcard):
    for n, edges, rng in instances(22, 40):
        sc = make(n, edges)
        c = rng.choice([2, 3, 7, 10])
        sc2 = make(n, [(i, j, c * w) for i, j, w in edges])
        r1 = B.run(sc, maxcardinality=maxcard, record=False)
        r2 = B.run(sc2, maxcardinality=maxcard, record=False)
        assert r2.count == r1.count
        assert r2.cost == c * r1.cost
        assert valid(sc, r2.pairs) and cost_of(sc, r2.pairs) == r1.cost    # auf den Originalkosten ebenfalls optimal


# --- 3. Verschiebung (nur bei Kardinalitätszwang) -----------------------------------------------------------------------------


def test_shifting_all_costs_adds_k_per_pair_under_cardinality_first():
    for n, edges, rng in instances(33, 40):
        sc = make(n, edges)
        k = rng.choice([1, 2, 5, 30])
        sc2 = make(n, [(i, j, w + k) for i, j, w in edges])
        r1 = B.run(sc, maxcardinality=True, record=False)
        r2 = B.run(sc2, maxcardinality=True, record=False)
        assert r2.count == r1.count
        assert r2.cost == r1.cost + k * r1.count
        assert valid(sc, r2.pairs) and cost_of(sc, r2.pairs) == r1.cost


# --- 4. Zusätzliche Kante -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("maxcard", [True, False])
def test_adding_a_feasible_edge_never_hurts(maxcard):
    done = 0
    for n, edges, rng in instances(44, 50, nmin=3):
        have = {(i, j) for i, j, _ in edges}
        free = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in have]
        if not free:
            continue
        i, j = rng.choice(free)
        w = rng.randint(1, 40)
        sc1, sc2 = make(n, edges), make(n, edges + [(i, j, w)])
        (c1, k1), (c2, k2) = value(sc1, maxcard), value(sc2, maxcard)
        if maxcard:
            assert c2 >= c1 and (c2 > c1 or k2 <= k1)
        else:
            assert k2 <= k1
        done += 1
    assert done >= 30


# --- 5. Kantenänderungen ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("maxcard", [True, False])
def test_edits_outside_or_inside_the_chosen_matching(maxcard):
    outside = inside = 0
    for n, edges, rng in instances(55, 50, nmin=3):
        sc = make(n, edges)
        r = B.run(sc, maxcardinality=maxcard, record=False)
        chosen = {tuple(sorted(p)) for p in r.pairs}
        unmatched = [e for e in edges if (e[0], e[1]) not in chosen]
        if unmatched:
            e = rng.choice(unmatched)
            removed = make(n, [x for x in edges if x != e])
            raised = make(n, [(i, j, w + (rng.randint(1, 30) if (i, j) == (e[0], e[1]) else 0)) for i, j, w in edges])
            assert value(removed, maxcard) == (r.count, r.cost)
            assert value(raised, maxcard) == (r.count, r.cost)
            outside += 1
        movable = [e for e in edges if (e[0], e[1]) in chosen and e[2] >= 2]
        if movable:
            e = rng.choice(movable)
            d = rng.randint(1, e[2] - 1)
            lowered = make(n, [(i, j, w - (d if (i, j) == (e[0], e[1]) else 0)) for i, j, w in edges])
            r2 = B.run(lowered, maxcardinality=maxcard, record=False)
            assert (r2.count, r2.cost) == (r.count, r.cost - d)
            inside += 1
    assert outside >= 30 and (inside >= 25 or not maxcard)   # ohne Zwang ist bei positiven Kosten die leere Paarung optimal


# --- 6. Disjunkte Vereinigung -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("maxcard", [True, False])
def test_disjoint_union_adds_up(maxcard):
    for k in range(30):
        rng = random.Random(660 + k)
        na, nb = rng.randint(1, 8), rng.randint(1, 8)
        ea = rand_edges(rng, na, 0.6, 9)
        eb = rand_edges(rng, nb, 0.6, 9)
        a, b = make(na, ea), make(nb, eb)
        union = make(na + nb, ea + [(i + na, j + na, w) for i, j, w in eb])
        (ca, ka), (cb, kb) = value(a, maxcard), value(b, maxcard)
        assert value(union, maxcard) == (ca + cb, ka + kb)


# --- 7. Primal-dual-Invarianten in jedem Zwischenzustand ---------------------------------------------------------------------


def _check_every_snapshot(sc, maxcard):
    res = B.run(sc, maxcardinality=maxcard, record=True)
    n = sc.n
    feasible = [(i, j) for i in range(n) for j in sc.adj[i] if i < j]
    maxcost = max((int(sc.cost[i, j]) for i, j in feasible), default=0)
    shift = 2 * (maxcost + 1) if maxcard else 0
    members = {b.id: set(b.members) for b in res.blossoms}
    best_count, best_cost = O.max_weight_matching_brute(sc.cost, sc.adj) if maxcard else (None, O.min_weight_matching_brute(sc.cost, sc.adj))
    # iw des Optimums in der internen Maximierung (ganzzahlig, y ist verdoppelt)
    opt_iw2 = 2 * (best_count * (maxcost + 1) - best_cost) if maxcard else -2 * best_cost
    last_dual = None
    for k, snap in enumerate(res.snapshots):
        zmap = dict(snap.z)
        assert all(z >= 0 for z in zmap.values()), (k, snap.z)
        for i, j in feasible:
            zsum = sum(z for bid, z in zmap.items() if i in members[bid] and j in members[bid])
            slack = snap.y[i] + snap.y[j] + 2 * zsum + 2 * int(sc.cost[i, j]) - shift
            assert slack >= 0, (k, i, j, slack)
            if snap.match[i] == j:
                assert slack == 0, (k, i, j, slack)
        # Dualzielwert (doppelt): sum y + 2 * sum z * floor(|B|/2); fällt nie
        dual2 = sum(snap.y) + 2 * sum(z * (len(members[bid]) // 2) for bid, z in zmap.items())
        if last_dual is not None:
            assert dual2 <= last_dual, (k, dual2, last_dual)
        last_dual = dual2
        # Schwach-Dualität: jede Paarung M hat iw(M) <= 1/2 sum max(y,0) + sum z floor(|B|/2)
        bound2 = sum(max(y, 0) for y in snap.y) + 2 * sum(z * (len(members[bid]) // 2) for bid, z in zmap.items())
        assert bound2 >= opt_iw2, (k, bound2, opt_iw2)
    return res


@pytest.mark.parametrize("maxcard", [True, False])
def test_dual_invariants_hold_at_every_snapshot_random_graphs(maxcard):
    for n, edges, rng in instances(77, 40, nmin=2, nmax=9):
        _check_every_snapshot(make(n, edges), maxcard)


def test_dual_invariants_hold_at_every_snapshot_on_blossom_maps():
    for name, build in ws.NETS.items():
        sc = build()
        if sc.n <= 12:
            _check_every_snapshot(sc, True)
            _check_every_snapshot(sc, False)
    for seed in (18, 151, 165, 100000):          # 18: Aufklappung mitten im Suchlauf, 151: viele bezahlte Blüten
        n = 7 if seed == 18 else 12
        _check_every_snapshot(ws.generate(n, 20 if seed == 18 else 28, 0, seed), True)


def test_final_dual_is_a_certificate_for_relabeled_and_scaled_maps():
    """Das Zertifikat bleibt nach Umnummerieren und Skalieren gültig (die Beweisstruktur ändert sich, die Gültigkeit nicht)."""
    for seed in range(20):
        sc = ws.generate(10 + seed % 4, 28, 0, 9000 + seed)
        sc2 = ws.relabel(sc, ws.perm(sc.n, seed))
        sc3 = make(sc2.n, [(i, j, 3 * w) for i, j, w in edges_of(sc2)])
        for s in (sc, sc2, sc3):
            r = B.run(s, maxcardinality=True)
            cert = B.certificate(s, r.pairs, r.y, r.z, {b.id: b.members for b in r.blossoms}, maxcardinality=True)
            assert cert["all_ok"], cert


# --- 8. Randfälle -------------------------------------------------------------------------------------------------------------


def test_edge_cases():
    assert value(make(0, [])) == (0, 0) and value(make(0, []), False) == (0, 0)
    assert value(make(1, [])) == (0, 0)
    assert value(make(5, [])) == (0, 0)
    assert value(make(2, [(0, 1, 7)])) == (1, 7)
    assert value(make(2, [(0, 1, 7)]), False) == (1, 7) or value(make(2, [(0, 1, 7)]), False) == (0, 0)
    # Weg 0-1-2-3 mit Kosten 1, 100, 1: lexikografisch beide Außenkanten (2 Paare, 2), nicht die Mitte
    path = make(4, [(0, 1, 1), (1, 2, 100), (2, 3, 1)])
    assert value(path) == (2, 2)
    # nur Gleichstände: alle Kosten gleich -> Kosten = Paarzahl * Kostenwert, Paarzahl = Maximum
    for n in range(2, 9):
        k = make(n, [(i, j, 4) for i in range(n) for j in range(i + 1, n)])
        assert value(k) == (n // 2, 4 * (n // 2))
