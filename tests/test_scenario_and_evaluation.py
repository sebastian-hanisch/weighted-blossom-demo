"""Karten, Wächter der kopierten Bausteine und die Plumbing der Auswertung (Verdict, Vergleichstabelle, Verteilungen, Sweeps)."""

import pytest

import wb_blossom as B
import wb_constants as C
import wb_evaluation as ev
import wb_scenario as S


def test_splitmix_vector_and_a_pinned_point_list():
    rng = S.SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]
    sc = S.generate(30, 20, 0, 100000)
    assert sc.pts[:3] == ((8, 69), (86, 39), (66, 62)) and sc.m == 46          # Wächter: derselbe Generator wie blossom-demo


def test_build_fixed_nets_ignore_random_parameters():
    a, b = S.build("windmuehle", 5, 99, 100, 123), S.build("windmuehle", 40, 10, 0, 1)
    assert a.pts == b.pts and a.n == 9
    assert S.build("random", 7, 40, 0, 3).n == 7
    assert set(C.FIXED_NETS) <= set(S.NETS)                                    # die UI zeigt nur eine Auswahl der Lehrbuchkarten aus wb_scenario


def test_bipartite_is_bipartite_by_construction_regardless_of_geometry():
    for k, m, reach, seed in ((9, 9, 40, 1), (5, 5, 10, 7), (3, 12, 60, 3)):
        sc = S.bipartite(k, m, reach, 0, seed)
        assert sc.n == k + m
        for i in range(sc.n):
            for j in sc.adj[i]:
                assert (i < k) != (j < k)                                     # nie eine Kante innerhalb einer Gruppe


def test_the_fixed_maps_have_no_new_geometry_only_new_weights():
    """wb_scenario übernimmt dieselben Koordinaten wie blossom-demo für die Lehrbuchkarten - nur die Kosten zählen jetzt."""
    assert S.flower_stem().pts == ((10, 66), (22, 50), (34, 50), (48, 72), (60, 62), (60, 38), (48, 28), (48, 92), (62, 98), (84, 96))
    assert S.windmill().pts[0] == (50, 50) and S.windmill().n == 9
    assert S.nested_flowers().n == 10


def test_the_engine_reaches_the_optimum_on_every_fixed_map():
    for net in C.NETS:
        sc = S.build(net, C.DEFAULT_N, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED)
        res = B.run(sc, maxcardinality=True, record=False)
        by_id = {b.id: b.members for b in res.blossoms}
        cert = B.certificate(sc, res.pairs, res.y, res.z, by_id, maxcardinality=True)
        assert cert["all_ok"], net


def test_verdict_of_a_normal_run():
    sc = S.generate(30, 20, 0, 165)
    a = ev.analyse(sc, C.OBJ_LEX, 165)
    level, code, d = ev.verdict(a)
    assert (level, code) == ("success", ev.OPTIMAL) and d["cert_ok"] and d["same_size"] and d["premium"] >= 0


def test_verdict_none_without_any_feasible_edge():
    sc = S.from_points([(0, 0), (90, 90), (0, 90)], 10)
    level, code, d = ev.verdict(ev.analyse(sc, C.OBJ_LEX, 0))
    assert (level, code) == ("info", ev.NONE) and d["count"] == 0 and d["isolated"] == 3


def test_verdict_degenerate_without_cardinality_constraint():
    sc = S.generate(30, 20, 0, 100000)
    a = ev.analyse(sc, C.OBJ_COST, 100000)
    level, code, d = ev.verdict(a)
    assert (level, code) == ("warning", ev.DEGENERATE) and (d["count"], d["cost"]) == (0, 0) and d["premium"] == d["blind_cost"]


def test_compare_table_rows():
    sc = S.generate(30, 20, 0, 165)
    a = ev.analyse(sc, C.OBJ_LEX, 165)
    rows = ev.compare_table(a)
    assert [r["label"] for r in rows] == ["Kosten ignoriert (kostenblinde größte Paarung)", "Gewichteter Blossom (bewiesenes Optimum)"]
    assert rows[0]["count"] == rows[1]["count"] and rows[0]["cost"] >= rows[1]["cost"]


def test_blind_scenario_ignores_cost_but_keeps_the_graph():
    sc = S.generate(20, 25, 0, 7)
    blind = ev.blind_scenario(sc)
    assert blind.adj == sc.adj and blind.reach == sc.reach
    feas_costs = {int(blind.cost[i, j]) for i in range(blind.n) for j in blind.adj[i]}
    assert feas_costs == {1}                                                  # jede mögliche Kante kostet genau 1


def test_distribution_fields_and_shapes():
    d = ev.distribution(30, 20, 0)
    assert d["n_seeds"] == 100 == d["n_valid"]
    assert d["same_size_share"] == 1.0                                        # Paarzahl hängt nie von den Kosten ab
    assert d["premium"]["min"] >= 0 and d["cert_ok_share"] == 1.0


def test_distribution_is_repeatable_and_without_feasible_pairs():
    assert ev.distribution(20, 25, 25) == ev.distribution(20, 25, 25)
    lonely = tuple(s for s in range(200) if S.generate(3, 10, 0, s).m == 0)
    assert len(lonely) >= 3
    assert ev.distribution(3, 10, 0, seeds=lonely)["n_valid"] == 0


def test_reach_sweep_and_scale_table_shapes():
    rows = ev.reach_sweep(20, 0, seeds=C.SWEEP_SEEDS[:8], reaches=(15, 25))
    assert [r["reach"] for r in rows] == [15, 25] and all(r["n_valid"] > 0 for r in rows)
    srows = ev.scale_table(ns=(10, 20), seeds=C.SCALE_SEEDS[:3])
    assert [r["n"] for r in srows] == [10, 20] and all(r["scanned"] > 0 for r in srows)


def test_bipartite_check_and_brute_check():
    bc = ev.bipartite_check(seeds=range(5))
    assert bc == {"ok": 5, "total": 5}
    br = ev.brute_check(seeds=range(5))
    assert br == {"ok": 5, "total": 5}
