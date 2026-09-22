"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Gewichteter Blossom"."""

# --- Regler (wie in den Vorgängerdemos) ---------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 4, 40, 30          # Fahrer
REACH_MIN, REACH_MAX, DEFAULT_REACH = 10, 150, 20   # Reichweite in Minuten
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG = 0, 100, 0   # ganze Prozent, Schritt 25
DEFAULT_SEED = 165                           # dieselbe Karte wie blossom-demo's Standard (direkter Vergleich)
SEED_MAX = 2_000_000_000

NETS = {"random": "Zufällige Karte", "verschachtelt": "Bezahlte, verschachtelte Blüten (10)", "bluete": "Blüte mit Stiel (10)", "windmuehle": "Windmühle (9)", "bipartit": "Bipartit (9+9)"}
DEFAULT_NET = "random"
FIXED_NETS = ("verschachtelt", "bluete", "windmuehle", "bipartit")

OBJ_LEX, OBJ_COST = "lex", "cost"
OBJECTIVE_LABELS = {OBJ_LEX: "Erst Paarzahl, dann Kosten", OBJ_COST: "Nur Kosten (keine Nebenbedingung)"}
DEFAULT_OBJECTIVE = OBJ_LEX

# --- feste Seed-Mengen (dieselben wie in den Vorgängerdemos; unabhängig vom Nutzer-Seed) --------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
SCALE_NS = (10, 20, 40, 80, 160, 320)
SCALE_SEEDS = DIST_SEEDS[:10]
SCALE_DEGREE_AREA = 12000                   # reach = isqrt(SCALE_DEGREE_AREA // n): mittlerer Grad bleibt ungefähr konstant
REACH_SWEEP = (10, 15, 20, 25, 30, 40, 60)
BRUTE_MAX_N = 12

COLORS = {"matched": "#1f77b4", "even": "#2ca02c", "odd": "#d62728", "none": "#c8c8c8", "path": "#ff7f0e", "inner": "#9467bd", "blossom": "#7f7f7f", "tree": "#555555", "blind": "#8c564b"}

# --- Presets ------------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", objective=OBJ_LEX, n=DEFAULT_N, reach=DEFAULT_REACH, ballung=DEFAULT_BALLUNG, seed=DEFAULT_SEED)
PRESETS = {
    "🪆 Bezahlte, verschachtelte Blüten": {**_BASE, "net": "verschachtelt"},
    "🌸 Blüte mit Stiel": {**_BASE, "net": "bluete"},
    "🌀 Windmühle": {**_BASE, "net": "windmuehle"},
    "🔗 Bipartit": {**_BASE, "net": "bipartit"},
    "🗺️ Mittlere Karte": {**_BASE},
    "🌪️ Braucht Aufklappung": {**_BASE, "n": 7, "reach": 20, "seed": 18},
    "💸 Kosten ignoriert": {**_BASE, "seed": 151},
    "🚫 Ohne Kardinalitätszwang": {**_BASE, "objective": OBJ_COST},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py und tests/test_presets.py belegt
PRESET_HELP = {
    "🪆 Bezahlte, verschachtelte Blüten": "Eine Blüte aus drei Fahrern liegt in einer größeren aus sieben. Beide werden bezahlt: die innere hat Dualwert 9, die äußere 2 (Tiefe 2) - der Beweis braucht sie, obwohl am Ende 5 Paare für 125 Minuten herauskommen, dieselben Kosten wie eine kostenblinde Paarung.",
    "🌸 Blüte mit Stiel": "Dieselbe Karte wie bei Blossom (unbezahlt), aber mit echten Kosten: hier reicht der Wald allein, ganz ohne eine einzige Kontraktion - 5 Paare für 114 Minuten. Ob eine Blüte gebraucht wird, hängt von den Kosten ab, nicht nur vom Graphen.",
    "🌀 Windmühle": "Der ganze Graph zieht sich zu einer einzigen, vierfach verschachtelten Blüte zusammen (Tiefe 4) - aber ihr Dualwert steht am Ende bei 0: verschachtelt heißt nicht bezahlt. Kosten ignorieren wäre hier trotzdem 19 Minuten teurer (32 gegen 51).",
    "🔗 Bipartit": "Neun Fahrzeuge, neun Aufträge, keine Kante innerhalb einer Gruppe (durch Konstruktion garantiert): 7 Paare für 108 Minuten, ohne eine einzige Kontraktion - stimmt exakt mit `scipy.optimize.linear_sum_assignment` überein, also mit der Ungarischen Methode.",
    "🗺️ Mittlere Karte": "30 Fahrer, Reichweite 20: 13 Paare für 145 Minuten, mit Beweis (2 bezahlte, verschachtelte Blüten). Eine Paarung, die die Kosten ignoriert, hätte dieselben 13 Paare für 178 Minuten gebraucht - 33 Minuten mehr.",
    "🌪️ Braucht Aufklappung": "Eine kleine Karte (7 Fahrer, 2 isoliert), auf der eine Blüte mitten im Suchlauf wieder aufgeklappt werden muss (nicht erst am Ende einer Stufe) - der algorithmisch anspruchsvollste Schritt des Verfahrens, live zu sehen.",
    "💸 Kosten ignoriert": "Eine Karte mit besonders großer Prämie: 14 Paare für 143 Minuten, aber 225 Minuten, wenn man die Kosten ignoriert - 82 Minuten mehr, mit 5 bezahlten Blüten unterwegs.",
    "🚫 Ohne Kardinalitätszwang": "Ohne Nebenbedingung an die Anzahl ist die leere Paarung (0 Paare, 0 Minuten) fast immer die billigste, weil jede Fahrzeit ≥ 0 ist - eine kostenblinde größte Paarung hätte 13 Paare für 178 Minuten gefunden. Genau deshalb gilt in der ganzen Matching-Linie „erst Paarzahl, dann Kosten“.",
}
