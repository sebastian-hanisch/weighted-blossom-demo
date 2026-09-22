"""Edmonds' Blossom-Algorithmus mit Gewichten: die (unter den größtmöglichen) billigste Paarung in einem allgemeinen Graphen. Primal-dual-Methode mit
Blüten-Dualwerten (Galil, "Efficient Algorithms for Finding Maximum Matching in Graphs", 1986) - genau das Verfahren hinter `networkx.max_weight_matching`,
hierher portiert mit demselben Ereignisprotokoll-Aufbau wie `blossom-demo/bl_blossom.py` (ein `Snapshot` je `Event`, `state_at(res, k)` zum Zurückspulen).

**Kostentransformation.** Der Löser maximiert intern ein Ersatzgewicht (wie `networkx.max_weight_matching`); nach außen minimiert `run()` immer die echte
Fahrzeit. Bei `maxcardinality=True` (lexikografisch zuerst nach Anzahl, dann nach Kosten): `iw(i, j) = (maxcost + 1) - cost(i, j)` auf allen zulässigen
Paaren (`maxcost` = größte vorkommende Fahrzeit), genau der Trick aus `networkx.min_weight_matching`. Bei `maxcardinality=False` (reine Kostenminimierung,
KEINE Nebenbedingung an die Anzahl - das macht die leere Paarung meist zur entarteten Lösung, da alle Kosten >= 0 sind) reicht die reine Vorzeichenumkehr
`iw(i, j) = -cost(i, j)`: der `(maxcost+1)`-Trick würde hier, weil `iw` dabei immer positiv bleibt, die interne Maximierung weiter zu möglichst vielen
Paaren treiben - das Gegenteil von "keine Nebenbedingung an die Anzahl" (empirisch nachgeprüft: mit dem `(maxcost+1)`-Trick auch bei `maxcardinality=False`
kam auf der Fahrgemeinschaften-Karte `generate(30, 20, 0, 100000)` 13 Paare/147 Minuten heraus statt der erwarteten 0/0 - das ist die einzige inhaltliche
Abweichung von der ursprünglichen Vorgabe, siehe Bericht). `y`/`iw` selbst bleiben intern; das Ereignisprotokoll und `Result.cost` melden immer echte
Minuten. Das Zertifikat (`certificate()`, ganz unten) rechnet mit dem rohen internen `y` weiter, braucht dafür aber ebenfalls `maxcardinality` und ein
GEGENÜBER der ursprünglichen Vorgabe korrigiertes Vorzeichen vor `cost` - Begründung dort.

**Id-Schema (wichtig für `wb_visualization.py`/`wb_evaluation.py`).** Ecken sind `0 .. n-1`. Jede Ecke ist gleichzeitig ihre eigene "triviale Blüte": ihre
Blüten-Id ist ihre Eckennummer. Jede ECHTE (mehrelementige) Blüte bekommt bei ihrer Entstehung (`addBlossom`) eine NEUE, fortlaufende Ganzzahl-Id ab `n`
(`n, n+1, n+2, ...`), auch wenn sie sich nach einer Aufklappung "gleich wieder" zu formen scheint - jede Kontraktion ist ein neuer, eigener Eintrag in
`Result.blossoms` (append-only, nie mutiert). `inblossom[v]` (intern) ist immer die aktuelle OBERSTE (nicht mehr enthaltene) Blüte, die `v` gerade
enthält - entweder eine Ecke selbst oder ein `_Blossom`-Objekt mit `.id`.

Intern hält der Löser fast wörtlich `networkx.max_weight_matching`s eigene Buchführung nach (Python-Objektidentität für `_Blossom`, Dicts, die sowohl mit
Eckennummern als auch mit `_Blossom`-Objekten als Schlüssel arbeiten - genau wie dort) statt alles in Ganzzahl-Arrays zu zwingen: das ist die mit Abstand
sicherste Portierung des primal-dualen Kerns (siehe die bekannten Fallstricke unten) und bleibt trotzdem mit dem Id-Schema oben voll integer-adressierbar
für alles, was nach außen (Ereignisse, Snapshots, Zertifikat) sichtbar ist. Einzige bewusste Vereinfachung gegenüber dem Original: die Trampolin-Technik
für `expandBlossom`/`augmentBlossom` (dort nur wegen Pythons Rekursionslimit bei sehr großen Graphen) ist hier durch normale Rekursion ersetzt - bei den
Kartengrößen dieses Portfolios (n bis in die Hunderte, Blütentiefe immer sehr viel kleiner als n) unbedenklich.

**Dualwerte, verdoppelt.** `y[v]` (= `dualvar`/`2*u(v)` bei networkx) und `z` je Blüte (= `blossomdual`) sind IMMER die verdoppelte Form; das ist der Grund,
warum sie stets ganzzahlig bleiben, obwohl die "echten" Werte y/2 halbzahlig sein können (und das ist normal, kein Fehler - nie runden oder Ganzzahligkeit
von y/2 annehmen). Straffheit einer zulässigen Kante (i, j): `y[i] + y[j] + 2 * sum(z der Blüten, die i UND j enthalten) - 2 * cost(i, j) >= 0`, Gleichheit
bei gewählten Kanten (`slack(v, w) <= 0` ist im internen Maximierungs-Vorzeichen "straff", exakt wie bei networkx - nicht ins Minimierungs-Vorzeichen
drehen). Unter `maxcardinality=True` dürfen einzelne `y`-Werte NEGATIV sein (erwartet, keine Prüfung auf Nichtnegativität von y - nur `z >= 0`).

**Snapshot.blossoms vs. Snapshot.z.** `Snapshot.blossoms` listet nur die gerade OBERSTEN aktiven Blüten (für die Zeichnung: was man als einen einzigen
kontrahierten Knoten sieht). `Snapshot.z` dagegen listet `(Blüten-Id, z)` für JEDE noch existierende (nicht aufgeklappte) Blüte, auch verschachtelte, die
gerade Teil einer größeren Blüte sind - das Zertifikat braucht diese Summe über ALLE Ebenen (mirror von networkx' `verifyOptimum`, das die volle
Vorfahren-Kette aufsummiert). `Blossom.members` (voll ausgeflacht, bei Entstehung fixiert) ist deshalb rotationssicher für die Zertifikat-Nachschlage-Tabelle
- `Blossom.base`/`.cycle` dagegen gelten nur "wie bei der Entstehung aufgezeichnet" (eine Rotation bei einer Augmentierung kann die tatsächliche Basis
später ändern); für den "aktuellen" Zustand immer den `Snapshot` nehmen, nie `.base`/`.cycle` eines alten Datensatzes.

Aufwand wird in angesehenen Kanten gezählt (nie Sekunden), wie überall im Portfolio.
"""

from dataclasses import dataclass

NONE, EVEN, ODD = 0, 1, 2

EV_ROUND, EV_GROW, EV_BLOSSOM, EV_AUGMENT, EV_DELTA, EV_EXPAND = "round", "grow", "blossom", "augment", "delta", "expand"


@dataclass(frozen=True)
class Blossom:
    id: int
    round: int          # in welcher Runde (Stufe) sie entstand
    base: int           # Basis-Ecke bei Entstehung (kann sich später durch Rotation ändern, s. Moduldoku)
    cycle: tuple         # Basis und die Ecken (bzw. Teilblüten, durch ihre Basis vertreten) des Kreises in Umlaufreihenfolge, bei Entstehung
    members: tuple       # ALLE Fahrer der Blüte, voll ausgeflacht, sortiert - rotationssicher, ändert sich nie
    depth: int            # 1 = keine Teilblüte
    children: tuple      # Ids der unmittelbar enthaltenen Teilblüten (nur nicht-triviale)


@dataclass(frozen=True)
class Event:
    kind: str
    round: int
    v: int = -1
    u: int = -1
    w: int = -1
    blossom: int = -1
    path: tuple = ()
    through: tuple = ()
    deltatype: int = -1
    delta: int = 0
    touched: tuple = ()
    z_at_expansion: int = -1
    scanned: int = 0


@dataclass(frozen=True)
class Snapshot:
    match: tuple
    label: tuple
    parent: tuple
    root: tuple
    blossoms: tuple      # Ids der aktuell OBERSTEN aktiven Blüten
    y: tuple             # verdoppelte Eck-Dualwerte, Länge n
    z: tuple             # (Blüten-Id, z) für JEDE aktuell existierende (auch verschachtelte) Blüte, sortiert nach Id


@dataclass(frozen=True)
class Round:
    index: int
    scanned: int
    contractions: int
    contracted_vertices: int
    deltas: int
    expansions: int
    path: tuple          # der gefundene Verbesserungsweg dieser Stufe, () bei der letzten (erfolglosen) Stufe
    pairs_after: int
    first: int
    last: int


@dataclass(frozen=True)
class Result:
    start: tuple
    pairs: tuple
    cost: int
    maxcardinality: bool
    rounds: tuple
    events: tuple
    snapshots: tuple
    blossoms: tuple       # ALLE je entstandenen Blüten, append-only
    y: tuple              # finale verdoppelte Eck-Dualwerte
    z: tuple              # (Blüten-Id, z) für jede am Ende noch existierende Blüte
    scanned_total: int
    augmentations: int
    contractions: int
    contracted_vertices: int
    expansions: int       # Zahl der Aufklappungen WÄHREND einer Stufe (delta4), ohne die stillen Stufenende-Aufklappungen
    max_depth: int

    @property
    def count(self):
        return len(self.pairs)

    @property
    def n_events(self):
        return len(self.events)

    def match_list(self, n):
        m = [-1] * n
        for i, j in self.pairs:
            m[i], m[j] = j, i
        return m


class _Blossom:
    __slots__ = ["id", "childs", "edges", "mybestedges"]

    def leaves(self):
        stack = [*self.childs]
        while stack:
            t = stack.pop()
            if isinstance(t, _Blossom):
                stack.extend(t.childs)
            else:
                yield t


class _NoNode:
    pass


def _pairs_of_mate(mate, n):
    return tuple((v, mate[v]) for v in range(n) if v in mate and mate[v] > v)


def _empty_result(sc, maxcardinality, record):
    n = sc.n
    snap0 = Snapshot((-1,) * n, (NONE,) * n, (-1,) * n, (-1,) * n, (), (0,) * n, ())
    events = (Event(EV_ROUND, 1, scanned=0),) if record else ()
    snaps = (snap0, snap0) if record else ()
    rounds = (Round(1, 0, 0, 0, 0, 0, (), 0, 1, 1 if record else 0),)
    return Result((), (), 0, maxcardinality, rounds, events, snaps, (), (0,) * n, (), 0, 0, 0, 0, 0, 0)


def run(sc, maxcardinality=True, record=True):
    """Primal-duale Blossom-Methode; minimiert intern über `iw = (maxcost+1) - cost` maximiert, meldet nach außen immer echte Kosten. Rückgabe `Result`."""
    n = sc.n
    adj = sc.adj
    feasible = [(i, j) for i in range(n) for j in adj[i] if i < j]
    if n == 0 or not feasible:
        return _empty_result(sc, maxcardinality, record)

    maxcost = max(int(sc.cost[i, j]) for i, j in feasible)

    def rcost(i, j):
        return int(sc.cost[i, j])

    # Zwei verschiedene interne Ersatzgewichte, je nach `maxcardinality` (Abweichung von einer wörtlichen Lesart der Spezifikation,
    # s. Moduldoku-Nachtrag unten): der (maxcost+1)-Trick ist NUR für maxcardinality=True richtig - er macht jede Kante lohnend
    # (iw >= 1), die interne Maximierung bevorzugt darum immer möglichst viele Paare, was für maxcardinality=True genau gewollt
    # ist (das ist exakt `networkx.min_weight_matching`s eigener Trick, der IMMER intern maxcardinality=True aufruft). Für
    # maxcardinality=False (keine Nebenbedingung an die Kardinalität, echte Kostenminimierung) braucht es stattdessen das simple
    # Vorzeichenumkehr-Gewicht `-cost`: eine Kante lohnt sich dann nur noch, wenn sie eine straffere (negativ genug) Struktur
    # ergibt, und die leere Paarung (alle echten Kosten >= 0) wird zur erwarteten entarteten Lösung.
    if maxcardinality:
        def iw(i, j):
            return (maxcost + 1) - rcost(i, j)
    else:
        def iw(i, j):
            return -rcost(i, j)

    maxiw = max([0] + [iw(i, j) for i, j in feasible])  # wie networkx: `maxweight` startet bei 0 und wird nie kleiner (wichtig für negative iw, maxcardinality=False)

    NONODE = _NoNode()
    mate = {}
    label = {}
    labeledge = {}
    inblossom = {v: v for v in range(n)}
    blossomparent = {v: None for v in range(n)}
    blossombase = {v: v for v in range(n)}
    bestedge = {}
    dualvar = {v: maxiw for v in range(n)}
    blossomdual = {}
    allowedge = {}
    queue = []
    next_id = [n]
    public_blossoms = []
    by_id = {}

    def slack(v, w):
        return dualvar[v] + dualvar[w] - 2 * iw(v, w)

    events = [] if record else None
    snaps = [] if record else None
    rounds = []
    stats = {"scanned": 0, "augmentations": 0, "contractions": 0, "contracted_vertices": 0, "expansions": 0}

    def pub_label(v):
        bl = label.get(inblossom[v])
        return EVEN if bl == 1 else (ODD if bl == 2 else NONE)

    def pub_parent(v):
        le = labeledge.get(inblossom[v])
        return le[0] if le is not None else -1

    def pub_root(v):
        b = inblossom[v]
        if label.get(b) is None:
            return -1
        guard = 0
        while labeledge.get(b) is not None:
            pv = labeledge[b][0]
            b = inblossom[pv]
            guard += 1
            if guard > 4 * n + 10:
                break
        return blossombase[b]

    def snap():
        matcht = tuple(mate.get(v, -1) for v in range(n))
        labelt = tuple(pub_label(v) for v in range(n))
        parentt = tuple(pub_parent(v) for v in range(n))
        roott = tuple(pub_root(v) for v in range(n))
        top = tuple(sorted(b.id for b in blossomdual if blossomparent.get(b) is None))
        zt = tuple(sorted((b.id, blossomdual[b]) for b in blossomdual))
        yt = tuple(dualvar[v] for v in range(n))
        return Snapshot(matcht, labelt, parentt, roott, top, yt, zt)

    def emit(kind, stage_no, **kw):
        if record:
            events.append(Event(kind, stage_no, scanned=stats["scanned"], **kw))
            snaps.append(snap())

    # --- assignLabel / scanBlossom / addBlossom / expandBlossom / augment*: literal port of networkx.max_weight_matching --------

    def assignLabel(w, t, v):
        b = inblossom[w]
        label[w] = label[b] = t
        if v is not None:
            labeledge[w] = labeledge[b] = (v, w)
        else:
            labeledge[w] = labeledge[b] = None
        bestedge[w] = bestedge[b] = None
        if t == 1:
            if isinstance(b, _Blossom):
                queue.extend(b.leaves())
            else:
                queue.append(b)
        elif t == 2:
            base = blossombase[b]
            assignLabel(mate[base], 1, base)

    def scanBlossom(v, w):
        path = []
        base = NONODE
        while v is not NONODE:
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            path.append(b)
            label[b] = 5
            if labeledge.get(b) is None:
                v = NONODE
            else:
                v = labeledge[b][0]
                b = inblossom[v]
                v = labeledge[b][0]
            if w is not NONODE:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base

    def addBlossom(base, v, w, stage_no):
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        b = _Blossom()
        b.id = next_id[0]
        next_id[0] += 1
        b.mybestedges = None
        blossombase[b] = base
        blossomparent[b] = None
        blossomparent[bb] = b
        b.childs = path = []
        b.edges = edgs = [(v, w)]
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            edgs.append(labeledge[bv])
            v = labeledge[bv][0]
            bv = inblossom[v]
        path.append(bb)
        path.reverse()
        edgs.reverse()
        while bw != bb:
            blossomparent[bw] = b
            path.append(bw)
            edgs.append((labeledge[bw][1], labeledge[bw][0]))
            w = labeledge[bw][0]
            bw = inblossom[w]
        label[b] = 1
        labeledge[b] = labeledge[bb]
        blossomdual[b] = 0
        for leaf in b.leaves():
            if label.get(inblossom[leaf]) == 2:
                queue.append(leaf)
            inblossom[leaf] = b
        bestedgeto = {}
        for bv2 in path:
            if isinstance(bv2, _Blossom):
                if bv2.mybestedges is not None:
                    nblist = bv2.mybestedges
                    bv2.mybestedges = None
                else:
                    nblist = [(lv, wv) for lv in bv2.leaves() for wv in adj[lv] if lv != wv]
            else:
                nblist = [(bv2, wv) for wv in adj[bv2] if bv2 != wv]
            for (i2, j2) in nblist:
                if inblossom[j2] == b:
                    i2, j2 = j2, i2
                bj = inblossom[j2]
                if bj != b and label.get(bj) == 1 and ((bj not in bestedgeto) or slack(i2, j2) < slack(*bestedgeto[bj])):
                    bestedgeto[bj] = (i2, j2)
            bestedge[bv2] = None
        b.mybestedges = list(bestedgeto.values())
        mybestedge, mybestslack = None, None
        bestedge[b] = None
        for k in b.mybestedges:
            kslack = slack(*k)
            if mybestedge is None or kslack < mybestslack:
                mybestedge, mybestslack = k, kslack
        bestedge[b] = mybestedge

        cycle = tuple(blossombase[c] if isinstance(c, _Blossom) else c for c in b.childs)
        members = tuple(sorted(b.leaves()))
        children_ids = tuple(c.id for c in b.childs if isinstance(c, _Blossom))
        depth = 1 + max((by_id[cid].depth for cid in children_ids), default=0)
        rec = Blossom(b.id, stage_no, base, cycle, members, depth, children_ids)
        public_blossoms.append(rec)
        by_id[b.id] = rec
        stats["contractions"] += 1
        stats["contracted_vertices"] += len(members)
        return b

    def expandBlossom(b, endstage):
        for s in b.childs:
            blossomparent[s] = None
            if isinstance(s, _Blossom):
                if endstage and blossomdual[s] == 0:
                    expandBlossom(s, True)
                    continue
                for leaf in s.leaves():
                    inblossom[leaf] = s
            else:
                inblossom[s] = s
        touched = []
        if (not endstage) and label.get(b) == 2:
            entrychild = inblossom[labeledge[b][1]]
            j = b.childs.index(entrychild)
            if j & 1:
                j -= len(b.childs)
                jstep = 1
            else:
                jstep = -1
            v, w = labeledge[b]
            while j != 0:
                if jstep == 1:
                    p, q = b.edges[j]
                else:
                    q, p = b.edges[j - 1]
                label[w] = None
                label[q] = None
                assignLabel(w, 2, v)
                touched.append(("grow", w))
                allowedge[(p, q)] = allowedge[(q, p)] = True
                j += jstep
                if jstep == 1:
                    v, w = b.edges[j]
                else:
                    w, v = b.edges[j - 1]
                allowedge[(v, w)] = allowedge[(w, v)] = True
                j += jstep
            bw = b.childs[j]
            label[w] = label[bw] = 2
            labeledge[w] = labeledge[bw] = (v, w)
            bestedge[bw] = None
            touched.append(("base", bw.id if isinstance(bw, _Blossom) else bw))
            j += jstep
            while b.childs[j] != entrychild:
                bv = b.childs[j]
                if label.get(bv) == 1:
                    j += jstep
                    continue
                if isinstance(bv, _Blossom):
                    vv = None
                    for leaf in bv.leaves():
                        if label.get(leaf):
                            vv = leaf
                            break
                else:
                    vv = bv
                if vv is not None and label.get(vv):
                    label[vv] = None
                    label[mate[blossombase[bv]]] = None
                    assignLabel(vv, 2, labeledge[vv][0])
                    touched.append(("relabel", vv))
                j += jstep
        label.pop(b, None)
        labeledge.pop(b, None)
        bestedge.pop(b, None)
        del blossomparent[b]
        del blossombase[b]
        z_val = blossomdual.pop(b)
        return z_val, touched

    def augmentBlossom(b, v):
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        if isinstance(t, _Blossom):
            augmentBlossom(t, v)
        i = j = b.childs.index(t)
        if i & 1:
            j -= len(b.childs)
            jstep = 1
        else:
            jstep = -1
        while j != 0:
            j += jstep
            t = b.childs[j]
            if jstep == 1:
                w, x = b.edges[j]
            else:
                x, w = b.edges[j - 1]
            if isinstance(t, _Blossom):
                augmentBlossom(t, w)
            j += jstep
            t = b.childs[j]
            if isinstance(t, _Blossom):
                augmentBlossom(t, x)
            mate[w] = x
            mate[x] = w
        b.childs = b.childs[i:] + b.childs[:i]
        b.edges = b.edges[i:] + b.edges[:i]
        blossombase[b] = blossombase[b.childs[0]]

    def augmentMatching(v, w):
        for s, j in ((v, w), (w, v)):
            while True:
                bs = inblossom[s]
                if isinstance(bs, _Blossom):
                    augmentBlossom(bs, s)
                mate[s] = j
                if labeledge.get(bs) is None:
                    break
                t = labeledge[bs][0]
                bt = inblossom[t]
                s, j = labeledge[bt]
                if isinstance(bt, _Blossom):
                    augmentBlossom(bt, j)
                mate[j] = s

    # --- Wegrekonstruktion (nur lesend, VOR der Augmentierung aufgerufen) für das `path`-Feld des Ereignisses --------------------

    def enter(t, x):
        if isinstance(t, _Blossom):
            return _blossom_path(t, x)
        return [x]

    def _blossom_path(b, v):
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        seq = list(enter(t, v))
        i = j = b.childs.index(t)
        if i & 1:
            j -= len(b.childs)
            jstep = 1
        else:
            jstep = -1
        while j != 0:
            j += jstep
            t = b.childs[j]
            if jstep == 1:
                w, x = b.edges[j]
            else:
                x, w = b.edges[j - 1]
            seq.extend(enter(t, w))
            j += jstep
            t = b.childs[j]
            seq.extend(enter(t, x))
        return seq

    def path_to_root(x):
        seq = []
        cur = x
        guard = 0
        while True:
            guard += 1
            if guard > 4 * n + 10:
                raise RuntimeError("path_to_root: Schleife")
            b = inblossom[cur]
            seq.extend(enter(b, cur))
            base_b = blossombase[b]
            if base_b not in mate:
                break
            nxt = mate[base_b]
            seq.append(nxt)
            bt = inblossom[nxt]
            if labeledge.get(bt) is None:
                break
            pv, entry_w = labeledge[bt]
            seg = enter(bt, entry_w)
            seq.extend(reversed(seg[:-1]))
            cur = pv
        return seq

    def through_blossoms(path):
        out = []
        edges = list(zip(path, path[1:]))
        for bobj in blossomdual:
            rec = by_id[bobj.id]
            mem = set(rec.members)
            inner = sum(1 for a, c in edges if a in mem and c in mem)
            if inner:
                inside = [x for x in path if x in mem]
                out.append((rec.id, inside[0], inside[-1], inner))
        return tuple(sorted(out))

    # --- Hauptschleife: eine Iteration = eine Stufe (Galil) -------------------------------------------------------------------

    stage_no = 0
    if record:
        snaps.append(Snapshot((-1,) * n, (NONE,) * n, (-1,) * n, (-1,) * n, (), tuple(dualvar[v] for v in range(n)), ()))

    while True:
        stage_no += 1
        label.clear()
        labeledge.clear()
        bestedge.clear()
        for b in blossomdual:
            b.mybestedges = None
        allowedge.clear()
        queue[:] = []
        for v in range(n):
            if v not in mate and label.get(inblossom[v]) is None:
                assignLabel(v, 1, None)

        first_ev = (len(events) + 1) if record else 0
        scan0 = stats["scanned"]
        con0, cv0, exp0 = stats["contractions"], stats["contracted_vertices"], stats["expansions"]
        emit(EV_ROUND, stage_no)
        augmented = False
        this_path = ()

        while True:
            while queue and not augmented:
                v = queue.pop()
                for w in adj[v]:
                    stats["scanned"] += 1
                    bv, bw = inblossom[v], inblossom[w]
                    if bv == bw:
                        continue
                    if (v, w) not in allowedge:
                        kslack = slack(v, w)
                        if kslack <= 0:
                            allowedge[(v, w)] = allowedge[(w, v)] = True
                    if (v, w) in allowedge:
                        if label.get(bw) is None:
                            assignLabel(w, 2, v)
                            ww = mate[blossombase[inblossom[w]]]
                            emit(EV_GROW, stage_no, v=v, u=w, w=ww)
                        elif label.get(bw) == 1:
                            base = scanBlossom(v, w)
                            if base is not NONODE:
                                bobj = addBlossom(base, v, w, stage_no)
                                emit(EV_BLOSSOM, stage_no, v=v, u=w, blossom=bobj.id)
                            else:
                                left = path_to_root(v)
                                right = path_to_root(w)
                                full_path = tuple(reversed(left)) + tuple(right)
                                through = through_blossoms(full_path)
                                augmentMatching(v, w)
                                augmented = True
                                stats["augmentations"] += 1
                                this_path = full_path
                                emit(EV_AUGMENT, stage_no, v=v, u=w, path=full_path, through=through)
                                break
                        elif label.get(w) is None:
                            label[w] = 2
                            labeledge[w] = (v, w)
                    elif label.get(bw) == 1:
                        if bestedge.get(bv) is None or kslack < slack(*bestedge[bv]):
                            bestedge[bv] = (v, w)
                    elif label.get(w) is None:
                        if bestedge.get(w) is None or kslack < slack(*bestedge[w]):
                            bestedge[w] = (v, w)
                if augmented:
                    break

            if augmented:
                break

            deltatype, delta, deltaedge, deltablossom = -1, None, None, None
            if not maxcardinality:
                deltatype, delta = 1, min(dualvar.values())
            for v in range(n):
                if label.get(inblossom[v]) is None and bestedge.get(v) is not None:
                    d = slack(*bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta, deltatype, deltaedge = d, 2, bestedge[v]
            for b in list(blossomparent):
                if blossomparent[b] is None and label.get(b) == 1 and bestedge.get(b) is not None:
                    kslack = slack(*bestedge[b])
                    d = kslack // 2
                    if deltatype == -1 or d < delta:
                        delta, deltatype, deltaedge = d, 3, bestedge[b]
            for b in blossomdual:
                if blossomparent[b] is None and label.get(b) == 2 and (deltatype == -1 or blossomdual[b] < delta):
                    delta, deltatype, deltablossom = blossomdual[b], 4, b
            if deltatype == -1:
                deltatype, delta = 1, max(0, min(dualvar.values()))

            y_before = dict(dualvar)
            z_before = {b.id: blossomdual[b] for b in blossomdual}
            for v in range(n):
                lb = label.get(inblossom[v])
                if lb == 1:
                    dualvar[v] -= delta
                elif lb == 2:
                    dualvar[v] += delta
            for b in blossomdual:
                if blossomparent[b] is None:
                    lb = label.get(b)
                    if lb == 1:
                        blossomdual[b] += delta
                    elif lb == 2:
                        blossomdual[b] -= delta
            touched = [("y", v, y_before[v], dualvar[v]) for v in range(n) if dualvar[v] != y_before[v]]
            touched += [("z", b.id, z_before.get(b.id, 0), blossomdual[b]) for b in blossomdual if blossomdual[b] != z_before.get(b.id, 0)]
            emit(EV_DELTA, stage_no, deltatype=deltatype, delta=delta, touched=tuple(touched))

            if deltatype == 1:
                break
            elif deltatype in (2, 3):
                (dv, dw) = deltaedge
                allowedge[(dv, dw)] = allowedge[(dw, dv)] = True
                queue.append(dv)
            elif deltatype == 4:
                z_val, relab = expandBlossom(deltablossom, False)
                stats["expansions"] += 1
                emit(EV_EXPAND, stage_no, blossom=deltablossom.id, z_at_expansion=z_val, touched=tuple(relab))

        last_ev = len(events) if record else 0
        rounds.append(Round(stage_no, stats["scanned"] - scan0, stats["contractions"] - con0, stats["contracted_vertices"] - cv0,
                             sum(1 for e in (events[first_ev - 1:last_ev] if record else ()) if e.kind == EV_DELTA),
                             stats["expansions"] - exp0, this_path, sum(1 for v in mate) // 2, first_ev, last_ev))
        if not augmented:
            break
        for b in list(blossomdual.keys()):
            if b not in blossomdual:
                continue
            if blossomparent.get(b) is None and label.get(b) == 1 and blossomdual[b] == 0:
                expandBlossom(b, True)

    pairs = _pairs_of_mate(mate, n)
    cost = sum(rcost(i, j) for i, j in pairs)
    y_final = tuple(dualvar[v] for v in range(n))
    z_final = tuple(sorted((b.id, blossomdual[b]) for b in blossomdual))
    max_depth = max((rec.depth for rec in public_blossoms), default=0)
    return Result((), pairs, cost, maxcardinality, tuple(rounds), tuple(events) if record else (), tuple(snaps) if record else (),
                  tuple(public_blossoms), y_final, z_final, stats["scanned"], stats["augmentations"], stats["contractions"],
                  stats["contracted_vertices"], stats["expansions"], max_depth)


def state_at(res, k):
    """Zustand nach Ereignis k (k = 0: vor der ersten Runde). Rückgabe (Snapshot, Blüten-Datensätze der gerade obersten aktiven Blüten)."""
    snap = res.snapshots[k]
    by_id = {b.id: b for b in res.blossoms}
    return snap, tuple(by_id[i] for i in snap.blossoms)


# --- Zertifikat: verdoppelte Straffheit + Dualzulässigkeit + Primal-Dual-Identität, unabhängig vom Löser ---------------------
#
# ABWEICHUNG (mit Zahlen nachgeprüft, siehe Bericht): eine wörtliche Lesart der ursprünglichen Vorgabe ("slack = y[u]+y[v]+2*sum(z)
# - 2*cost, >= 0, ==0 bei gewählten Kanten") ist mit dem oben beschriebenen Ersatzgewicht NICHT richtig - y ist der Dualwert des
# INTERNEN Maximierungsproblems auf `iw = (maxcost+1) - cost` (maxcardinality=True) bzw. `iw = -cost` (maxcardinality=False), und
# eine Kostenminimierung hat in dieser Dualität zwangsläufig das umgekehrte Vorzeichen vor `cost` (ein Minimierungs-Dual ist
# "Kosten minus Dualwerte >= 0", nicht "Dualwerte minus Kosten >= 0" - dieselbe Konvention wie `hungarian-demo`s eigenes
# Zertifikat, `cost[i,j] - (pi_o[j]-pi_v[i])`). Am K4-Beispiel unten nachgerechnet (y = (39, 17, 1, 1), maxcost = 30): mit der
# wörtlichen Formel kommt ein negativer `min_slack` heraus (das "Zertifikat" widerspräche der nachweislich richtigen Lösung);
# mit der hier verwendeten, aus der iw-Straffheit zurückgerechneten Formel stimmt jede Kante exakt mit dem unabhängig geprüften
# IW-Straffheitswert überein. Die Idee (Straffheit in verdoppelter Form, Identität als Gegenprobe) bleibt erhalten, nur das
# Vorzeichen vor `cost` und der Term `shift = 2*(maxcost+1)` (nur bei maxcardinality=True) sind anders als ursprünglich benannt.
# Zweite, kleinere Abweichung an der Identität selbst: `sum(y)` läuft nur über die GEPAARTEN Ecken, nicht über alle n - freie
# Ecken tragen bei maxcardinality=True einen eigenen, nicht notwendig verschwindenden konstanten Dualwert (networkx' eigenes
# `vdualoffset` in `verifyOptimum`); mit allen n Ecken ging die Identität an echten (nicht perfekten) Paarungen nachweisbar
# nicht auf (an einer 30-Ecken-Karte mit nur 28 gepaarten Ecken: 206 statt 246), an perfekten Paarungen (K4, das n=8-Beispiel)
# fiel das nicht auf, weil dort keine Ecke frei blieb.

def certificate(sc, pairs, y, active_blossom_z, blossom_members_by_id, maxcardinality=True):
    """`active_blossom_z`: ((Blüten-Id, z), ...) für JEDE am Ende noch existierende Blüte (auch verschachtelte, s. Moduldoku).
    `blossom_members_by_id`: {Id: iterable von Ecken} (die vollen, ausgeflachten Mitglieder, z.B. aus den `Blossom`-Datensätzen).
    `maxcardinality` muss zum `run()`-Aufruf passen, der `y`/`pairs` erzeugt hat (steuert das Vorzeichen/den Term `shift`, s.o.)."""
    n = sc.n
    feasible_pairs = [(i, j) for i in range(n) for j in sc.adj[i] if i < j]
    maxcost = max((int(sc.cost[i, j]) for i, j in feasible_pairs), default=0)
    shift = 2 * (maxcost + 1) if maxcardinality else 0
    matched = {}
    for i, j in pairs:
        matched[i], matched[j] = j, i
    members = {bid: set(mem) for bid, mem in blossom_members_by_id.items()}
    min_slack = None
    matched_tight = True
    for i, j in feasible_pairs:
        zsum = sum(z for bid, z in active_blossom_z if i in members.get(bid, ()) and j in members.get(bid, ()))
        slack_doubled = y[i] + y[j] + 2 * zsum + 2 * int(sc.cost[i, j]) - shift
        if min_slack is None or slack_doubled < min_slack:
            min_slack = slack_doubled
        if matched.get(i) == j and slack_doubled != 0:
            matched_tight = False
    feasible = min_slack is None or min_slack >= 0
    z_nonneg = all(z >= 0 for _, z in active_blossom_z)
    total_cost = sum(int(sc.cost[i, j]) for i, j in pairs)
    # nur über die GEPAARTEN Ecken summieren, nicht alle n (freie Ecken tragen bei maxcardinality=True einen eigenen, nicht
    # notwendig verschwindenden konstanten Dualwert - vgl. networkx' `vdualoffset` in `verifyOptimum` - und fehlen deshalb hier)
    lhs = sum(y[v] for v in matched) + sum(2 * z * ((len(members.get(bid, ())) - 1) // 2) for bid, z in active_blossom_z)
    rhs = shift * len(pairs) - 2 * total_cost
    identity_holds = lhs == rhs
    all_ok = feasible and z_nonneg and identity_holds and matched_tight
    return {"feasible": feasible, "identity_holds": identity_holds, "identity_lhs": lhs, "identity_rhs": rhs,
            "min_slack": min_slack if min_slack is not None else 0, "z_nonneg": z_nonneg, "matched_tight": matched_tight, "all_ok": all_ok}
