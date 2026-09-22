"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App und der README ist hier über die 100 festen Karten (DIST_SEEDS) belegt.
Alles rechnet mit ganzen Zahlen und einem eigenen Zufallsgenerator - die Werte sind auf jeder Plattform dieselben; die Toleranzen decken nur die Rundung
auf die im Text genannten Stellen. Mittel und Median stehen zusammen."""

import pytest

import wb_evaluation as ev


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.4f} statt {expected}"


@pytest.fixture(scope="module")
def dist():
    return ev.distribution(30, 20, 0)


def test_the_default_map_family(dist):
    d = dist
    near(d["degree"], 2.9727, 0.0005)
    near(d["count"]["mean"], 13.37, 0.005)
    assert d["count"]["median"] == 13.0 and (d["count"]["min"], d["count"]["max"]) == (11, 15)
    near(d["cost"]["mean"], 147.3, 0.05)
    assert d["cost"]["median"] == 149.5
    near(d["blind_cost"]["mean"], 179.72, 0.05)
    assert d["blind_cost"]["median"] == 178.0
    assert d["same_size_share"] == 1.0                                        # die Paarzahl hängt nie von den Kosten ab


def test_the_premium_of_ignoring_cost(dist):
    d = dist
    near(d["premium"]["mean"], 32.42, 0.005)
    assert d["premium"]["median"] == 31.0 and (d["premium"]["min"], d["premium"]["max"]) == (2, 80)
    assert d["premium_positive_share"] == 1.0                                 # auf allen 100 Karten echt teurer


def test_effort_and_blossoms(dist):
    d = dist
    near(d["scanned"]["mean"], 653.87, 0.05)
    assert d["scanned"]["median"] == 654.0
    near(d["contractions"]["mean"], 6.36, 0.005)
    assert d["contractions"]["median"] == 6.0
    near(d["expansions"]["mean"], 0.44, 0.005)
    assert d["expansions"]["median"] == 0.0 and d["expansions_total"] == 44 and d["expansions"]["max"] == 5
    near(d["depth"]["mean"], 2.76, 0.005)
    assert d["depth"]["median"] == 3.0
    near(d["n_paid"]["mean"], 4.55, 0.005)
    assert d["n_paid"]["median"] == 4.0 and d["paid_share"] == 1.0            # auf allen 100 Karten mindestens eine bezahlte Blüte
    assert d["cert_ok_share"] == 1.0                                          # der Beweis gelingt auf allen 100 Karten


def test_bipartite_matches_scipy():
    r = ev.bipartite_check()
    assert r == {"ok": 20, "total": 20}


def test_brute_force_agreement():
    r = ev.brute_check()
    assert r == {"ok": 20, "total": 20}


def test_reach_sweep_the_premium_grows_with_the_reach():
    rows = {r["reach"]: r for r in ev.reach_sweep(30)}
    expect = {10: (0.825, 4.125, 0.525), 20: (3.035, 31.4, 1.0), 30: (6.1383, 100.475, 1.0), 40: (9.9833, 182.125, 1.0), 60: (17.6033, 342.85, 1.0)}
    for reach, (deg, prem, paid) in expect.items():
        near(rows[reach]["degree"], deg, 0.0006)
        near(rows[reach]["premium_mean"], prem, 0.05)
        near(rows[reach]["paid_share"], paid, 0.0006)
    means = [rows[r]["premium_mean"] for r in (10, 20, 30, 40, 60)]
    assert means == sorted(means)                                             # streng steigend: je größer die Reichweite, desto teurer, Kosten zu ignorieren


def test_effort_against_size():
    rows = {r["n"]: r for r in ev.scale_table()}
    assert [rows[n]["reach"] for n in (10, 20, 40, 80, 160, 320)] == [34, 24, 17, 12, 8, 6]
    expect = {10: 94.0, 20: 306.9, 40: 1087.4, 80: 2805.7, 160: 7660.4, 320: 29544.9}
    for n, scans in expect.items():
        near(rows[n]["scanned"], scans, 0.5)
    assert all(2.5 < rows[n]["degree"] < 3.5 for n in rows)
