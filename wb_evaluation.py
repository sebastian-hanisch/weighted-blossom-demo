"""Auswertung: eine Karte (`analyse`, `verdict`, `compare_table`), viele Karten (`distribution`), Aufwand gegen Größe, Reichweiten-Sweep, Bipartit-Gegenprobe.
Alles ganzzahlig und deterministisch; nur die Anzeige-Statistiken (Anteile, Mittel, Mediane) sind Gleitkomma.

Anders als bei `blossom-demo` gibt es hier kein "Verfahren vs. Optimum": Gewichteter Blossom IST das Optimum, mit Beweis (Zertifikat) auf jeder Karte.
Die Frage ist stattdessen: **was bringt es, die Kosten überhaupt zu berücksichtigen?** Die Gegenprobe ist eine **kostenblinde** größtmögliche Paarung
(alle Kanten gleich teuer, `maxcardinality=True` auf einer Ersatzkarte mit Einheitskosten) - dieselbe Paarzahl (das folgt zwingend aus der Graphstruktur,
Kosten ändern nie, WELCHE Paarzahl maximal ist), aber zu welchem Preis. Die **Prämie** ist der Unterschied in echten Minuten.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import wb_constants as C
from wb_blossom import certificate, run
from wb_scenario import Scenario, build, generate

OPTIMAL, DEGENERATE, NONE = "optimal", "degenerate", "none"


def blind_scenario(sc):
    """Dieselbe Karte, aber jede mögliche Kante kostet 1: eine größtmögliche Paarung, die die echten Kosten nicht kennt."""
    ones = np.where(sc.cost >= 0, 1, 1).astype(sc.cost.dtype)
    return Scenario(sc.pts, sc.reach, ones, sc.adj)


def blind_result(sc):
    return run(blind_scenario(sc), maxcardinality=True, record=False)


def real_cost(sc, pairs):
    return int(sum(sc.cost[i, j] for i, j in pairs))


@dataclass
class Analysis:
    scenario: object
    objective: str
    seed: int
    result: object        # wb_blossom.Result
    blind: object          # wb_blossom.Result auf der kostenblinden Ersatzkarte
    blind_cost: int        # echte Kosten der kostenblinden Paarung
    cert: dict


def analyse(sc, objective=C.OBJ_LEX, seed=0):
    res = run(sc, maxcardinality=(objective == C.OBJ_LEX), record=True)
    by_id = {b.id: b.members for b in res.blossoms}
    cert = certificate(sc, res.pairs, res.y, res.z, by_id, maxcardinality=res.maxcardinality)
    blind = blind_result(sc)
    return Analysis(sc, objective, seed, res, blind, real_cost(sc, blind.pairs), cert)


def verdict(a):
    """(Stufe, Code, Zahlen) für die Anzeige; `Zahlen` enthält jede Zahl, die der Text nennt."""
    sc, r = a.scenario, a.result
    feasible = any(sc.adj[i] for i in range(sc.n))
    if not feasible:
        code = NONE
    elif a.objective == C.OBJ_COST and r.count == 0:
        code = DEGENERATE
    else:
        code = OPTIMAL
    premium = a.blind_cost - r.cost
    data = {"n": sc.n, "count": r.count, "cost": r.cost, "blind_count": a.blind.count, "blind_cost": a.blind_cost, "premium": premium,
            "same_size": a.blind.count == r.count, "events": r.n_events, "scanned": r.scanned_total, "contractions": r.contractions,
            "contracted_vertices": r.contracted_vertices, "expansions": r.expansions, "depth": r.max_depth, "augmentations": r.augmentations,
            "n_paid_blossoms": sum(1 for _, z in r.z if z > 0), "cert": a.cert, "cert_ok": a.cert["all_ok"], "isolated": sum(1 for i in range(sc.n) if not sc.adj[i])}
    level = {NONE: "info", OPTIMAL: "success", DEGENERATE: "warning"}[code]
    return level, code, data


def compare_table(a):
    sc, r, b = a.scenario, a.result, a.blind
    return [{"label": "Kosten ignoriert (kostenblinde größte Paarung)", "count": b.count, "cost": a.blind_cost, "events": b.n_events},
            {"label": "Gewichteter Blossom (bewiesenes Optimum)", "count": r.count, "cost": r.cost, "events": r.n_events}]


# --- viele Karten --------------------------------------------------------------------------------------------------------------

def _row(sc):
    res = run(sc, maxcardinality=True, record=False)
    by_id = {b.id: b.members for b in res.blossoms}
    cert = certificate(sc, res.pairs, res.y, res.z, by_id, maxcardinality=True)
    b = blind_result(sc)
    bc = real_cost(sc, b.pairs)
    return {"count": res.count, "cost": res.cost, "blind_count": b.count, "blind_cost": bc, "premium": bc - res.cost, "same_size": b.count == res.count,
            "scanned": res.scanned_total, "contractions": res.contractions, "contracted_vertices": res.contracted_vertices, "expansions": res.expansions,
            "depth": res.max_depth, "n_paid": sum(1 for _, z in res.z if z > 0), "cert_ok": cert["all_ok"], "degree": sum(len(a) for a in sc.adj) / max(sc.n, 1)}


@lru_cache(maxsize=64)
def cell_rows(n, reach, ballung, seeds):
    return tuple(_row(generate(n, reach, ballung, sd)) for sd in seeds)


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _med(v):
    return float(np.median(v)) if len(v) else None


def _stat(values):
    return {"mean": _mean(values), "median": _med(values), "min": min(values) if len(values) else None, "max": max(values) if len(values) else None}


def distribution(n, reach, ballung, seeds=C.DIST_SEEDS):
    """Verteilung über viele Karten: Paarzahl, Kosten, Prämie gegenüber der kostenblinden Paarung, Aufwand, bezahlte Blüten (Mittel, Median, Extrem)."""
    rows = [r for r in cell_rows(n, reach, ballung, tuple(seeds)) if r["count"] > 0 or r["blind_count"] > 0]
    if not rows:
        return {"n_seeds": len(seeds), "n_valid": 0}
    return {"n_seeds": len(seeds), "n_valid": len(rows), "count": _stat([r["count"] for r in rows]), "cost": _stat([r["cost"] for r in rows]),
            "blind_cost": _stat([r["blind_cost"] for r in rows]), "premium": _stat([r["premium"] for r in rows]),
            "premium_positive_share": _mean([r["premium"] > 0 for r in rows]), "same_size_share": _mean([r["same_size"] for r in rows]),
            "scanned": _stat([r["scanned"] for r in rows]), "contractions": _stat([r["contractions"] for r in rows]),
            "expansions": _stat([r["expansions"] for r in rows]), "expansions_total": sum(r["expansions"] for r in rows),
            "depth": _stat([r["depth"] for r in rows]), "n_paid": _stat([r["n_paid"] for r in rows]), "paid_share": _mean([r["n_paid"] > 0 for r in rows]),
            "cert_ok_share": _mean([r["cert_ok"] for r in rows]), "degree": _mean([r["degree"] for r in rows])}


def reach_sweep(n, ballung=0, seeds=C.SWEEP_SEEDS, reaches=C.REACH_SWEEP):
    """Gegen die Reichweite: mittlerer Grad, Paarzahl, Prämie, bezahlte Blüten, Aufwand."""
    out = []
    for r in reaches:
        d = distribution(n, r, ballung, seeds)
        if d["n_valid"] == 0:
            continue
        out.append({"reach": r, "degree": d["degree"], "count": d["count"]["mean"], "premium_mean": d["premium"]["mean"], "premium_median": d["premium"]["median"],
                    "paid_share": d["paid_share"], "contractions": d["contractions"]["mean"], "n_valid": d["n_valid"]})
    return out


@lru_cache(maxsize=8)
def scale_table(ns=C.SCALE_NS, seeds=C.SCALE_SEEDS):
    """Angesehene Kanten gegen die Größe (Reichweite so, dass der mittlere Grad etwa gleich bleibt)."""
    out = []
    for n in ns:
        reach = max(3, math.isqrt(C.SCALE_DEGREE_AREA // n))
        scans, contr, deg, edges = [], [], [], []
        for sd in seeds:
            sc = generate(n, reach, 0, sd)
            res = run(sc, maxcardinality=True, record=False)
            scans.append(res.scanned_total)
            contr.append(res.contractions)
            deg.append(sum(len(a) for a in sc.adj) / n)
            edges.append(sum(len(a) for a in sc.adj) // 2)
        out.append({"n": n, "reach": reach, "scanned": _mean(scans), "contractions": _mean(contr), "degree": _mean(deg), "edges": _mean(edges)})
    return out


def bipartite_check(k=9, m=9, reach=40, ballung=0, seeds=range(20)):
    """Bipartit gebaut (keine Kante innerhalb einer Gruppe) gegen scipy.optimize.linear_sum_assignment: Übereinstimmung in Paarzahl und Kosten."""
    from scipy.optimize import linear_sum_assignment

    from wb_scenario import bipartite
    ok = 0
    total = 0
    for sd in seeds:
        sc = bipartite(k, m, reach, ballung, sd)
        res = run(sc, maxcardinality=True, record=False)
        big = 10 ** 6
        cm = np.full((k, m), big, dtype=np.int64)
        for i in range(k):
            for j in sc.adj[i]:
                cm[i, j - k] = sc.cost[i, j]
        row, col = linear_sum_assignment(cm)
        chosen = [(i, j) for i, j in zip(row, col) if cm[i, j] < big]
        total += 1
        ok += (len(chosen) == res.count and sum(int(cm[i, j]) for i, j in chosen) == res.cost)
    return {"ok": ok, "total": total}


def brute_check(n=10, reach=25, seeds=range(20)):
    """Gegenprobe gegen den eigenen Brute-Force-Löser (`wb_oracle`) für kleine Karten."""
    from wb_oracle import max_weight_matching_brute
    ok = 0
    total = 0
    for sd in seeds:
        sc = generate(n, reach, 0, sd)
        res = run(sc, maxcardinality=True, record=False)
        bsize, bcost = max_weight_matching_brute(sc.cost, sc.adj)
        total += 1
        ok += (res.count == bsize and res.cost == bcost)
    return {"ok": ok, "total": total}


def scenario_from_settings(net, n, reach, ballung, seed):
    return build(net, n, reach, ballung, seed)
