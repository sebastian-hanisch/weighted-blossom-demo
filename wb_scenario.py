"""Szenario: Fahrgemeinschaften - eine Gruppe von Fahrern auf einer Karte, je zwei können zusammen fahren.

Anders als in den Vorgängerdemos gibt es hier **keine zwei Seiten**: jeder Fahrer kann mit jedem gepaart werden, dessen Fahrzeit (auf ganze Minuten aufgerundete Entfernung, per `isqrt`, ohne
Gleitkomma) höchstens die Reichweite beträgt. Das ist ein allgemeiner Graph, und in ihm gibt es ungerade Kreise (drei Fahrer, die einander erreichen) - die "Blüten". Die Fahrzeit trägt jetzt
zusätzlich ein Gewicht: `wb_blossom` sucht nicht mehr nur die größtmögliche, sondern die (unter den größtmöglichen) billigste Paarung.

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt über `numpy.random`: numpy garantiert keine über Versionen stabilen Zufallsströme, die CI installiert
aber wöchentlich die neueste Version. So sind Voreinstellungen, Seeds und jede im Text genannte Zahl auf Windows und Linux dieselben.

`bipartite(k, m, ...)` baut zusätzlich eine echte zweiseitige Karte (Fahrzeuge 0..k-1 gegen Aufträge k..k+m-1, wie bei `hungarian-demo`/`auction-algorithm-demo`), aber auf demselben
allgemeinen-Graph-Fundament: nach der üblichen Entfernungs-Zulässigkeit werden alle Paare INNERHALB derselben Gruppe zusätzlich als nicht zulässig markiert, damit der Graph unabhängig von der
Geometrie garantiert zweiseitig ist (nicht nur zufällig, weil die beiden Gruppen geometrisch getrennt liegen)."""

from dataclasses import dataclass
from math import isqrt

import numpy as np

_MASK = (1 << 64) - 1
MAP_SIZE = 100
N_CENTRES = 3


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 101 liegt um 1e-17)."""
        return self.next() % n


def travel_cost(dx, dy):
    """Aufgerundete Entfernung in Minuten und das Quadrat der Entfernung (beides ganzzahlig)."""
    d2 = dx * dx + dy * dy
    r = isqrt(d2)
    return r + (1 if d2 > r * r else 0), d2


@dataclass(frozen=True)
class Scenario:
    pts: tuple           # ((x, y), ...) - die Fahrer
    reach: int
    cost: np.ndarray     # (n, n) int64, Fahrzeit in Minuten (symmetrisch)
    adj: tuple           # adj[i] = aufsteigend sortierte Nachbarn von i (Entfernung <= Reichweite)

    @property
    def n(self):
        return len(self.pts)

    @property
    def m(self):
        """Zahl der möglichen Paare (Kanten)."""
        return sum(len(a) for a in self.adj) // 2

    def edges(self):
        """Alle möglichen Paare als (Kosten, i, j) mit i < j, aufsteigend sortiert (Gleichstand: kleinster Index)."""
        return sorted((int(self.cost[i, j]), i, j) for i in range(self.n) for j in self.adj[i] if i < j)

    def degrees(self):
        return tuple(len(a) for a in self.adj)


def from_points(pts, reach):
    n = len(pts)
    cost = np.zeros((n, n), dtype=np.int64)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c, d2 = travel_cost(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            cost[i, j] = c
            if d2 <= reach * reach:
                adj[i].append(j)
    return Scenario(tuple(map(tuple, pts)), int(reach), cost, tuple(map(tuple, adj)))


def generate(n, reach, ballung, seed):
    """Zufällige Karte mit n Fahrern. `ballung` in ganzen Prozent: 0 = gleichmäßig verteilt, 100 = alle Punkte um drei Zentren."""
    rng = SplitMix64(seed)
    lo, hi = 15, MAP_SIZE - 15
    centres = [(lo + rng.below(hi - lo + 1), lo + rng.below(hi - lo + 1)) for _ in range(N_CENTRES)]

    def point():
        ux, uy = rng.below(MAP_SIZE + 1), rng.below(MAP_SIZE + 1)
        cx, cy = centres[rng.below(N_CENTRES)]
        jx, jy = rng.below(21) - 10, rng.below(21) - 10
        x = ((100 - ballung) * ux + ballung * (cx + jx)) // 100
        y = ((100 - ballung) * uy + ballung * (cy + jy)) // 100
        return min(max(x, 0), MAP_SIZE), min(max(y, 0), MAP_SIZE)

    return from_points([point() for _ in range(n)], reach)


def bipartite(k, m, reach, ballung, seed):
    """Wie `generate`, aber mit zwei erzwungenen Gruppen: Fahrer 0..k-1 ("Fahrzeuge"), Fahrer k..k+m-1 ("Aufträge"). Kanten innerhalb einer Gruppe werden nach der
    normalen Entfernungs-Zulässigkeit zusätzlich auf nicht zulässig gesetzt, unabhängig davon, wie nah sie geometrisch liegen - der Graph ist dadurch durch Konstruktion
    zweiseitig, nicht nur zufällig durch die Geometrie."""
    n = k + m
    rng = SplitMix64(seed)
    lo, hi = 15, MAP_SIZE - 15
    centres = [(lo + rng.below(hi - lo + 1), lo + rng.below(hi - lo + 1)) for _ in range(N_CENTRES)]

    def point():
        ux, uy = rng.below(MAP_SIZE + 1), rng.below(MAP_SIZE + 1)
        cx, cy = centres[rng.below(N_CENTRES)]
        jx, jy = rng.below(21) - 10, rng.below(21) - 10
        x = ((100 - ballung) * ux + ballung * (cx + jx)) // 100
        y = ((100 - ballung) * uy + ballung * (cy + jy)) // 100
        return min(max(x, 0), MAP_SIZE), min(max(y, 0), MAP_SIZE)

    pts = [point() for _ in range(n)]
    sc = from_points(pts, reach)
    adj = [[] for _ in range(n)]
    for i in range(n):
        gi = 0 if i < k else 1
        for j in sc.adj[i]:
            gj = 0 if j < k else 1
            if gi != gj:
                adj[i].append(j)
    return Scenario(sc.pts, sc.reach, sc.cost, tuple(tuple(a) for a in adj))


def relabel(sc, order):
    """Dieselbe Karte mit anderer Nummerierung: Fahrer i der neuen Karte ist Fahrer order[i] der alten. Der Graph ist derselbe, nur die Reihenfolge, in der die Suche ihn sieht, ist anders."""
    return from_points([sc.pts[k] for k in order], sc.reach)


def perm(n, seed):
    """Zufällige Permutation von 0..n-1 (Fisher-Yates mit SplitMix64)."""
    rng = SplitMix64(seed)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        k = rng.below(i + 1)
        p[i], p[k] = p[k], p[i]
    return p


# --- feste Karten (Lehrbuchfälle): kleine Reichweiten, damit der Graph genau der gewünschte ist -------------------------------------

def triangle_pendant():
    """A: Dreieck A-B-C mit Anhängsel D an B (4 Fahrer). Greedy nimmt die billigste Kante B-C und findet 1 Paar, das Optimum sind 2 (A-C und B-D).
    Der Weg D-B=C-A führt durch das Dreieck: die kleinste Blüte."""
    return from_points([(34, 38), (50, 46), (50, 30), (60, 60)], 20)


def flower_stem():
    """B: Blüte mit Stiel (10 Fahrer): Stiel A-B, Blüte {C,D,E,F,G} (Fünfeck, Basis C), Ausgang D-H, danach H=I-J. Der Weg A B C G F E D H I J hat 9 Kanten, 4 davon im Inneren der Blüte
    (der Weg um die Blüte herum hat gerade Länge)."""
    return from_points([(10, 66), (22, 50), (34, 50), (48, 72), (60, 62), (60, 38), (48, 28), (48, 92), (62, 98), (84, 96)], 27)


def nested_flowers():
    """C: verschachtelte Blüten (10 Fahrer): eine Blüte {C,D,E} liegt in der größeren {A,B,C,D,E,F,G}; der Verbesserungsweg J I H F G D E C B A (9 Kanten) läuft durch beide.
    Das Layout ist maschinell gefittet (die Adjazenz wird im Test exakt geprüft)."""
    return from_points([(51, 13), (76, 11), (85, 21), (82, 43), (90, 43), (46, 38), (57, 47), (22, 48), (15, 61), (22, 83)], 28)


def windmill():
    """D: Windmühle (9 Fahrer): der mittlere Fahrer ist mit allen acht äußeren verbunden, die äußeren paarweise (B-C, D-E, F-G, H-I). Greedy findet schon das Optimum (4),
    aber der Beweis braucht vier verschachtelte Kontraktionen: der ganze Graph ist eine einzige Blüte."""
    return from_points([(50, 50), (76, 46), (76, 54), (54, 76), (46, 76), (24, 54), (24, 46), (46, 24), (54, 24)], 30)


def star():
    """E: Stern (4 Fahrer): der mittlere ist mit den drei äußeren verbunden, die äußeren untereinander nicht. Optimum 1; A = {mittlerer}, D = {die drei äußeren}: 3 - 1 = 2 = 4 - 2."""
    return from_points([(50, 50), (50, 24), (27, 63), (73, 63)], 27)


def bipartit_default():
    """F: feste zweiseitige Karte (9 Fahrzeuge, 9 Aufträge, 24 mögliche Paare, kein isolierter Fahrer), Reichweite 40, Seed 1 - für die bipartite Gegenprobe
    gegen `scipy.optimize.linear_sum_assignment`."""
    return bipartite(9, 9, 40, 0, 1)


NETS = {
    "dreieck": triangle_pendant,
    "bluete": flower_stem,
    "verschachtelt": nested_flowers,
    "windmuehle": windmill,
    "stern": star,
    "bipartit": bipartit_default,
}


def build(net, n, reach, ballung, seed):
    """Karte zu den Einstellungen; feste Karten ignorieren die Zufallsparameter."""
    if net in NETS:
        return NETS[net]()
    return generate(n, reach, ballung, seed)
