"""Startpaarungen: leer oder eine der beiden Greedy-Regeln (wie in der Wurzel-Demo, jetzt auf einem allgemeinen Graphen).

- edge: alle möglichen Paare nach (Kosten, i, j) sortieren, jedes nehmen, dessen beide Enden frei sind (billigste Kante zuerst).
- order: Fahrer für Fahrer in Indexreihenfolge; jeder nimmt seinen ersten freien Nachbarn (kleinster Index).
Rückgabe: `match`-Liste mit match[i] = Partner oder -1.
"""

START_EMPTY, START_EDGE, START_ORDER = "empty", "edge", "order"
STARTS = (START_EDGE, START_ORDER, START_EMPTY)


def greedy_edge(sc):
    match = [-1] * sc.n
    for _c, i, j in sc.edges():
        if match[i] < 0 and match[j] < 0:
            match[i], match[j] = j, i
    return match


def greedy_order(sc):
    match = [-1] * sc.n
    for i in range(sc.n):
        if match[i] >= 0:
            continue
        for j in sc.adj[i]:
            if match[j] < 0:
                match[i], match[j] = j, i
                break
    return match


def start_matching(sc, start):
    if start == START_EMPTY:
        return [-1] * sc.n
    if start == START_EDGE:
        return greedy_edge(sc)
    if start == START_ORDER:
        return greedy_order(sc)
    raise ValueError(start)


def pairs_of(match):
    """Paare (i, j) mit i < j aus einer match-Liste."""
    return tuple((i, j) for i, j in enumerate(match) if j > i)
