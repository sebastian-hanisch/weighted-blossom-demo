"""Orakel für Tests: Brute Force per Bitmasken-Dynamik, jetzt mit Gewichten (Nachfolger von `blossom-demo/bl_oracle.py`, ohne dessen Tutte-Berge-Teil - der war
Beweismaschinerie fürs ungewichtete Ergebnis, hier reicht die Zielfunktion selbst als Orakel).

`max_weight_matching_brute(cost, adj)`: DP über Teilmengen (Ecke mit kleinstem Index: allein lassen oder mit einem freien Nachbarn paaren); jede Maske trägt
das Paar (größte erreichbare Anzahl Paare, darunter die kleinsten Kosten) - lexikografisch, genau das Ziel von `wb_blossom.run(sc, maxcardinality=True)`.
Brauchbar bis etwa n = 14 (2^n Masken).

`min_weight_matching_brute(cost, adj)`: dieselbe DP, aber ohne die Nebenbedingung an die Kardinalität, rein nach Kosten - für die Gegenprobe zu
`maxcardinality=False` (dort ist meist die leere Paarung mit Kosten 0 die billigste, da alle Kosten >= 0 sind: die Kardinalität ist "entartet").
"""

from functools import lru_cache


def max_weight_matching_brute(cost, adj):
    """(größte Anzahl Paare, kleinste Kosten unter den größten Paarungen) über den ganzen Graphen."""
    n = len(adj)
    nb = [sum(1 << j for j in adj[i]) for i in range(n)]

    @lru_cache(None)
    def f(mask):
        if mask == 0:
            return (0, 0)
        i = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << i)
        best = f(rest)                         # i bleibt allein
        c = nb[i] & rest
        while c:
            j = (c & -c).bit_length() - 1
            c &= c - 1
            size2, cost2 = f(rest & ~(1 << j))
            cand = (size2 + 1, cost2 + int(cost[i][j]))
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand
        return best

    result = f((1 << n) - 1)
    f.cache_clear()
    return result


def min_weight_matching_brute(cost, adj):
    """Kleinste Kosten über ALLE Paarungen (auch die leere), ohne Nebenbedingung an die Kardinalität."""
    n = len(adj)
    nb = [sum(1 << j for j in adj[i]) for i in range(n)]

    @lru_cache(None)
    def f(mask):
        if mask == 0:
            return 0
        i = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << i)
        best = f(rest)
        c = nb[i] & rest
        while c:
            j = (c & -c).bit_length() - 1
            c &= c - 1
            cand = f(rest & ~(1 << j)) + int(cost[i][j])
            if cand < best:
                best = cand
        return best

    result = f((1 << n) - 1)
    f.cache_clear()
    return result
