"""Gewichteter Blossom - die billigste Paarung in einem allgemeinen Graphen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Edmonds' gewichteten
Blossom-Algorithmus - und lässt stattdessen das Beispiel wachsen. Neuntes Stück der Matching-Linie der "Konzepte"-Reihe, der Konvergenzpunkt:
Ungarische Methode (gewichtet, bipartit) + Blossom (ungewichtet, allgemein) -> Gewichteter Blossom (gewichtet, allgemein). Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import wb_constants as C
import wb_evaluation as ev
from wb_blossom import state_at
from wb_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from wb_scenario import build
from wb_visualization import build_hist, build_map, build_progress, build_reach_sweep, build_scale

st.set_page_config(page_title="Gewichteter Blossom – Sebastian Hanisch", layout="wide")


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return f"{x:,.0f}".replace(",", " ")


def _ms(s, digits=1):
    return f"{_f(s['mean'], digits)} | {_f(s['median'], digits)}"


def _share(x):
    return f"{100 * x:.0f} %"


def _premium_values(n, reach, ballung):
    rows = ev.cell_rows(n, reach, ballung, C.DIST_SEEDS)
    return [r["premium"] for r in rows]


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    net, n, reach, ballung, seed, objective = params
    return ev.analyse(build(net, n, reach, ballung, seed), objective, seed)


@st.cache_data(show_spinner=False)
def _distribution(n, reach, ballung):
    return ev.distribution(n, reach, ballung)


@st.cache_data(show_spinner=False)
def _scale():
    return ev.scale_table()


@st.cache_data(show_spinner=False)
def _reach_sweep(n, ballung):
    return ev.reach_sweep(n, ballung)


@st.cache_data(show_spinner=False)
def _brute_check(reach):
    return ev.brute_check(reach=reach)


st.title("💍 Gewichteter Blossom – die billigste Paarung in einem allgemeinen Graphen")
st.markdown(
    """
**Blossom** findet die *größte* Paarung in einem allgemeinen Graphen, ohne auf Kosten zu achten. Die **Ungarische Methode** findet die *billigste* Paarung, aber nur zwischen zwei Seiten.
Diese Demo ist der **Konvergenzpunkt** beider: **Edmonds' gewichteter Blossom-Algorithmus** findet die billigste Paarung in einem **allgemeinen** Graphen (Fahrgemeinschaften, wie in `blossom-demo`) - mit **Dualwerten**
je Fahrer UND je Blüte (eine Blüte kann etwas *kosten*), die dieselbe Rolle spielen wie die Potentiale der Ungarischen Methode. Anders als dort ist dieses Verfahren **immer beweisbar optimal**: es gibt hier keine Heuristik, die scheitern könnte - nur die Frage, wie viel es bringt, die Kosten überhaupt zu berücksichtigen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - neuntes und letztes Stück des Verbesserungswege-Asts der Matching-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Der Vergleich ist deshalb nicht \"Verfahren gegen Optimum\" (es IST das Optimum, mit Beweis auf jeder Karte), sondern \"Kosten berücksichtigt gegen ignoriert\": die **Prämie** gegenüber einer größtmöglichen Paarung, die alle Kanten für gleich teuer hält. "
    "Danach ist der ganze Verbesserungswege-Ast der Matching-Linie fertig; offen bleibt in der ganzen Linie nur noch **Stabile Mitbewohner**."
)

with st.expander("So funktioniert Gewichteter Blossom", expanded=True):
    st.markdown(
        """
1. **Dualwerte statt fester Kosten:** jeder Fahrer bekommt einen Dualwert `y` (wie die Potentiale der Ungarischen Methode), jede kontrahierte Blüte einen eigenen Dualwert `z >= 0`. Eine Kante ist **straff** (darf im Wald wachsen), wenn `y[i] + y[j] + 2·Σz = 2·Kosten(i, j)` gilt - sonst ist noch "Luft" (Schlupf) drin.
2. **Wald wachsen, wie bei Blossom:** von allen freien Fahrern aus wächst ein Wald über straffe Kanten; eine straffe Kante zwischen zwei geraden Fahrern verschiedener Bäume ist ein Verbesserungsweg, zwischen zwei geraden Fahrern desselben Baums schließt sie eine **Blüte** (mit neuem Dualwert `z = 0`).
3. **Kein Fortschritt mehr? Die Dualwerte drehen** (ein `delta`-Schritt): der kleinste Schritt, der irgendwo eine neue Kante straff macht (oder - wenn nichts mehr geht - die Suche beendet), wird auf alle Dualwerte angewendet. Manchmal fällt dabei der Dualwert einer Blüte auf 0: sie wird **mitten im Suchlauf wieder aufgeklappt**.
4. **Ende und Beweis:** ist die Paarung größtmöglich UND lässt sich kein Dualwert mehr sinnvoll bewegen, ist sie **bewiesen billigste**. Der Beweis prüft, dass jede Kante straff oder locker genug ist (nie negativ) und dass eine Identität aus Dualwerten genau die Kosten der Paarung ergibt.
        """
    )

st.caption("🎯 Schnellstart – eine Beispielkarte laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Karte", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Eine zufällige Karte mit Fahrern oder eine feste Lehrbuchkarte: verschachtelte bezahlte Blüten, eine Blüte mit Stiel (die hier ganz ohne Kontraktion auskommt), die Windmühle, oder eine echte bipartite Karte für die Gegenprobe.",
    )
    if net_key == "random":
        seed_widget("n_slider")
        n = st.slider("Fahrer", *bounds("n_slider"), key="n_slider", help="Anzahl der Fahrer.")
        st.session_state[KEPT["n_slider"]] = n
        seed_widget("reach_slider")
        reach = st.slider("Reichweite [min]", *bounds("reach_slider"), key="reach_slider", step=5, help="Wie weit zwei Fahrer höchstens auseinander liegen dürfen. Je größer, desto teurer wird es, Kosten zu ignorieren.")
        st.session_state[KEPT["reach_slider"]] = reach
        seed_widget("ballung_slider")
        ballung = st.slider("Ballung [%]", *bounds("ballung_slider"), key="ballung_slider", step=25, help="0 = Fahrer gleichmäßig verteilt, 100 = alle um drei Stadtteile gruppiert.")
        st.session_state[KEPT["ballung_slider"]] = ballung
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neue Karte generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilung über 100 feste Karten weiter unten ändert sich dabei nicht.")
    else:
        n = int(st.session_state.get(KEPT["n_slider"], C.DEFAULT_N))
        reach = int(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        ballung = int(st.session_state.get(KEPT["ballung_slider"], C.DEFAULT_BALLUNG))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Diese Karte ist fest - es gibt nichts zu erzeugen.")

fixed = net_key in C.FIXED_NETS

# --- Ziel und Ablauf ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ziel und Ablauf")
step_col, obj_col, play_col = st.columns([4, 4, 2])
with obj_col:
    objective = st.radio("Ziel", list(C.OBJECTIVE_LABELS), key="objective_radio", format_func=lambda k: C.OBJECTIVE_LABELS[k],
                         help="Erst Paarzahl, dann Kosten: die Regel der ganzen Matching-Linie. Nur Kosten: keine Nebenbedingung an die Anzahl - da jede Fahrzeit >= 0 ist, wird dann fast immer niemand gepaart (die leere Paarung kostet 0).")
params = (net_key, int(n), int(reach), int(ballung), int(seed), objective)
if fixed:
    params = (net_key, C.DEFAULT_N, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, C.DEFAULT_SEED, objective)
with st.spinner("Rechne..."):
    a = _analysis(params)
sc, res = a.scenario, a.result
level, code, d = ev.verdict(a)
n_events = res.n_events
if st.session_state.get("wb_step_owner") != params:
    st.session_state["wb_step"] = n_events
    st.session_state["wb_step_owner"] = params
with step_col:
    if n_events > 0:
        step = st.slider("Ereignis", 0, n_events, key="wb_step", help="Wie viele Ereignisse der Suche schon geschehen sind (Runde beginnt, Fahrer entdeckt, Blüte gefunden, Dualwerte gedreht, Weg umgeklappt, Blüte aufgeklappt). Ganz rechts das Ergebnis mit dem Beweis.")
    else:
        step = 0
        st.caption("Kein Ereignis: keine Paare möglich.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch", disabled=n_events == 0)
sync_query_params({"net_select": net_key, "objective_radio": objective, "n_slider": int(n), "reach_slider": int(reach), "ballung_slider": int(ballung), "seed_input": int(seed)})
view_slot = st.empty()
lab = lambda v: f"{v + 1}"


def _event_text(k):
    if k == 0:
        return f"Anfang: alle {sc.n} Fahrer sind frei, alle Dualwerte stehen auf {res.snapshots[0].y[0] if sc.n else 0}. Noch keine Suche."
    e = res.events[k - 1]
    if e.kind == "round":
        roots = sum(1 for x in state_at(res, k - 1)[0].match if x < 0)
        return f"Runde {e.round}: eine neue Stufe beginnt. Alle {roots} freien Fahrer sind Wurzeln (gerade)."
    if e.kind == "grow":
        return f"Fahrer {lab(e.v)} (gerade) erreicht {lab(e.u)} über eine straffe Kante: {lab(e.u)} wird ungerade, sein Partner {lab(e.w)} gerade."
    if e.kind == "blossom":
        blo = next(b for b in res.blossoms if b.id == e.blossom)
        return f"Fahrer {lab(e.v)} und {lab(e.u)} sind beide gerade und im selben Baum: die straffe Kante schließt eine Blüte mit {len(blo.members)} Fahrern (Dualwert startet bei 0)."
    if e.kind == "augment":
        thr = ""
        if e.through:
            thr = " Der Weg läuft durch " + " und ".join(f"die Blüte {b} (Eintritt {lab(i)}, Austritt {lab(o)}, {inner} Kanten im Inneren)" for b, i, o, inner in e.through) + "."
        return f"Fahrer {lab(e.v)} und {lab(e.u)} liegen in verschiedenen Bäumen: der Weg {' - '.join(lab(x) for x in e.path)} ist ein Verbesserungsweg.{thr} Umklappen."
    if e.kind == "delta":
        names = {1: "δ1 (ein freier Fahrer erreicht Dualwert 0 - Suche endet)", 2: "δ2 (kleinster Schlupf gerade↔frei)", 3: "δ3 (kleinster Schlupf gerade↔gerade)", 4: "δ4 (Dualwert einer Blüte erreicht 0 - sie wird aufgeklappt)"}
        touched = f" {len(e.touched)} Dualwerte ändern sich um {e.delta}." if e.touched else ""
        return f"Kein weiterer Fortschritt ohne die Dualwerte zu drehen: {names.get(e.deltatype, e.deltatype)}, Betrag {e.delta}.{touched}"
    if e.kind == "expand":
        return f"Die Blüte {e.blossom} erreicht Dualwert 0 und wird mitten im Suchlauf wieder aufgeklappt; ihre Teilblüten werden neu beschriftet."
    return "unbekanntes Ereignis"


def _cert_table():
    c = d["cert"]
    ok = lambda b: "✅" if b else "❌"
    rows = [("Straffheit (alle Kanten)", f"{ok(c['feasible'])} kleinster Schlupf {c['min_slack']}"),
            ("Gewählte Kanten straff", ok(c["matched_tight"])),
            ("Blütendualwerte ≥ 0", ok(c["z_nonneg"])),
            ("Primal-Dual-Identität", f"{ok(c['identity_holds'])} {c['identity_lhs']} = {c['identity_rhs']}")]
    return {"Bestandteil": [r[0] for r in rows], "Ergebnis": [r[1] for r in rows]}


def _render(k):
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Nach {k} von {n_events} Ereignissen** - " + _event_text(k))
        c1.plotly_chart(build_map(sc, res, k), width="stretch", key=f"wb_map_{k}")
        c2.markdown("**Verlauf**")
        c2.plotly_chart(build_progress(res, k), width="stretch", key=f"wb_progress_{k}")
        snap, blossoms = state_at(res, k)
        if blossoms:
            c2.table({"Blüte": [b.id for b in blossoms], "Fahrer": [len(b.members) for b in blossoms], "Dualwert z": [dict(snap.z).get(b.id, 0) for b in blossoms], "Tiefe": [b.depth for b in blossoms]})
        if k > 0 and res.events[k - 1].kind == "delta" and res.events[k - 1].touched:
            c2.table({"Art": ["Fahrer" if t[0] == "y" else "Blüte" for t in res.events[k - 1].touched[:10]], "Id": [t[1] + 1 if t[0] == "y" else t[1] for t in res.events[k - 1].touched[:10]],
                      "vorher": [t[2] for t in res.events[k - 1].touched[:10]], "nachher": [t[3] for t in res.events[k - 1].touched[:10]]})
        if k == n_events:
            st.markdown("**Beweis:** ist das Ergebnis bewiesen billigste?")
            st.table(_cert_table())
            st.caption("Straffheit in verdoppelter Form: y[i] + y[j] + 2·Σz + 2·Kosten(i,j) ≥ eine Konstante, Gleichheit auf jeder gewählten Kante (Herleitung im Mathe-Teil unten).")


if auto_play:
    frames = sorted({int(round(x)) for x in np.linspace(0, n_events, min(n_events, 40) + 1)})
    for k in frames:
        _render(k)
        time.sleep(min(0.6, 6.0 / max(len(frames), 1)))
    step = n_events
else:
    _render(step)

st.caption("Fahrer: grün = gerade, rot = ungerade, grau = noch nicht im Wald (Hover zeigt den Dualwert). Blaue Linien sind gewählte Paare, dünne dunkle Linien der Wald. Eine violette Hülle ist eine aktive Blüte (Hover zeigt ihren Dualwert). "
           "Beim letzten Ereignis: grün = neu entdeckte Kante, violett gestrichelt = die Kante, die eine Blüte schließt, orange = der Verbesserungsweg (Stücke im Inneren einer Blüte violett).")

st.markdown("---")

# --- Wie gut? --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was bringt es, die Kosten zu berücksichtigen?")
if code == "none":
    st.info("ℹ️ Kein einziges Paar ist möglich – die Reichweite ist zu klein.")
elif code == "degenerate":
    st.warning(f"„Nur Kosten“ ohne Nebenbedingung an die Anzahl: {d['count']} Paare für {d['cost']} Minuten. Da jede Fahrzeit ≥ 0 ist, ist die leere Paarung fast immer die billigste - genau deshalb gilt in der ganzen Matching-Linie „erst Paarzahl, dann Kosten“.")
else:
    st.success(f"✅ {d['count']} Paare für {d['cost']} Minuten - bewiesen billigste. Eine Paarung, die die Kosten ignoriert, hätte dieselbe Paarzahl ({d['blind_count']}, das folgt allein aus der Graphstruktur) für {d['blind_cost']} Minuten gebraucht: {d['premium']} Minuten mehr.")
    if net_key == "bipartit":
        st.caption("Auf 20 von 20 getesteten bipartiten Karten stimmt das exakt mit `scipy.optimize.linear_sum_assignment` überein - demselben Verfahren wie die Ungarische Methode (Testorakel, keine Laufzeitabhängigkeit dieser App). Ist der Graph bipartit, entstehen nie Blüten: der Gewichtete Blossom ist dann nichts anderes als die Ungarische Methode - der stärkste Beleg für die Konvergenz.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Paare", f"{d['count']}", delta=f"Kosten ignoriert: {d['blind_count']}", delta_color="off", help="Beide Regeln finden dieselbe Paarzahl - sie hängt nur vom Graphen ab, nie von den Kosten.")
m2.metric("Kosten", f"{d['cost']} min", delta=f"Kosten ignoriert: {d['blind_cost']} min" if d["blind_count"] else "–", delta_color="off")
m3.metric("Prämie des Ignorierens", f"{d['premium']} min", delta=f"{d['n_paid_blossoms']} bezahlte Blüte(n)", delta_color="off", help="Wie viel eine größtmögliche Paarung mehr kostet, wenn sie die Fahrzeit gar nicht kennt (alle Kanten gleich teuer).")
m4.metric("Aufwand (angesehene Kanten)", _int(d["scanned"]), delta=f"{d['contractions']} Kontraktionen, {d['expansions']} Aufklappungen", delta_color="off", help="Aufwand in Kanten (Kopfzahl, nie Sekunden); Kontraktionen und Aufklappungen mitten im Suchlauf.")
cmp_rows = ev.compare_table(a)
st.table({"": [r["label"] for r in cmp_rows], "Paare": [str(r["count"]) for r in cmp_rows], "Kosten [min]": [str(r["cost"]) for r in cmp_rows], "Ereignisse": [str(r["events"]) for r in cmp_rows]})

if fixed:
    st.info("Feste Karte: es gibt nur diese eine Ziehung. Für die Verteilung über viele Karten eine zufällige Karte wählen.")
elif code == "optimal":
    st.markdown(f"**Nicht nur diese eine Karte:** {len(C.DIST_SEEDS)} feste Karten mit denselben Einstellungen (Fahrer {n}, Reichweite {reach}, Ballung {ballung} %), getrennt vom Seed oben.")
    dist = _distribution(int(n), int(reach), int(ballung))
    if dist["n_valid"] == 0:
        st.info("ℹ️ Bei dieser Reichweite gibt es auf keiner der Karten ein mögliches Paar.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Paare (Mittel | Median)", _ms(dist["count"]), delta=f"{dist['n_valid']} von {dist['n_seeds']} Karten mit Paaren", delta_color="off")
        p2.metric("Prämie (Mittel | Median)", f"{_f(dist['premium']['mean'], 1)} | {_f(dist['premium']['median'], 1)} min", delta=f"auf {_share(dist['premium_positive_share'])} der Karten positiv", delta_color="off")
        p3.metric("Bezahlte Blüten", _share(dist["paid_share"]), delta=f"im Mittel {_f(dist['n_paid']['mean'], 2)} je Karte", delta_color="off")
        p4.metric("Aufklappungen mitten im Suchlauf", _int(dist["expansions_total"]), delta=f"höchstens {dist['expansions']['max']} auf einer Karte", delta_color="off")
        h1, h2 = st.columns(2)
        h1.plotly_chart(build_hist(_premium_values(int(n), int(reach), int(ballung)), mean=dist["premium"]["mean"], median=dist["premium"]["median"], current=d["premium"],
                                   x_title="Prämie [min]", bin_size=5, color=C.COLORS["blind"]), width="stretch", key="wb_hist_premium")
        h1.caption("Prämie über die Karten; die rote Linie ist Ihre Ziehung.")
        h2.table({"Kennzahl": ["Kosten", "Kosten ignoriert", "Kontraktionen", "Aufwand (Kanten)"], "Mittel": [_f(dist["cost"]["mean"], 0), _f(dist["blind_cost"]["mean"], 0), _f(dist["contractions"]["mean"], 1), _f(dist["scanned"]["mean"], 0)],
                  "Median": [_f(dist["cost"]["median"], 0), _f(dist["blind_cost"]["median"], 0), _f(dist["contractions"]["median"], 1), _f(dist["scanned"]["median"], 0)]})


st.markdown("---")

# --- Experimente auf Abruf ------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wovon hängt es ab?")
if not fixed:
    if st.button("Reichweiten-Sweep (40 Karten je Wert)", key="reach_start"):
        st.session_state["reach_on"] = (int(n), int(ballung))
    if st.session_state.get("reach_on") == (int(n), int(ballung)):
        with st.spinner("Rechne 7 Reichweiten × 40 Karten..."):
            rrows = _reach_sweep(int(n), int(ballung))
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_reach_sweep(rrows), width="stretch", key="wb_reach_chart")
        c2.table({"Reichweite": [r["reach"] for r in rrows], "Grad": [_f(r["degree"], 1) for r in rrows], "Prämie (Mittel)": [_f(r["premium_mean"], 1) for r in rrows], "bezahlte Blüte": [_share(r["paid_share"]) for r in rrows]})
        st.caption("Je größer die Reichweite, desto mehr Alternativen hat jeder Fahrer - und desto teurer wird es, die Kosten zu ignorieren (die Prämie wächst mit der Reichweite, nicht umgekehrt).")

if st.button("Aufwand gegen die Größe (n = 10 bis 320)", key="scale_start"):
    st.session_state["scale_on"] = True
if st.session_state.get("scale_on"):
    with st.spinner("Rechne 6 Größen × 10 Karten..."):
        rows = _scale()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_scale(rows), width="stretch", key="wb_scale_chart")
    c2.table({"n": [r["n"] for r in rows], "Reichweite": [r["reach"] for r in rows], "angesehene Kanten": [_int(r["scanned"]) for r in rows], "Kontraktionen": [_f(r["contractions"], 1) for r in rows]})
    st.caption("Mittel über 10 feste Karten, Reichweite so gewählt, dass der mittlere Grad bei jeder Größe etwa 3 bleibt.")

if st.button("Brute-Force-Beweis auf kleinen Karten", key="brute_start"):
    st.session_state["brute_on"] = True
if st.session_state.get("brute_on"):
    with st.spinner("Probiere alle Teilmengen..."):
        bc = _brute_check(int(reach) if net_key == "random" else 25)
    st.success(f"✅ {bc['ok']} von {bc['total']} kleinen Karten (10 Fahrer) stimmen mit dem eigenen Brute-Force-Orakel exakt überein.")

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jeder Fahrer nimmt genau einen Partner** | Größere Gruppen (Kapazitäten, b-Matching) brauchen andere Verfahren. | (nicht in der Linie) |
| **Größe zählt, nicht Zufriedenheit** | Haben die Fahrer Vorlieben, ist die stabile Paarung das Ziel - und in einer Gruppe muss keine existieren. | **Stabile Mitbewohner** (gebaut) |
| **Alles ist vorab bekannt** | Kommen die Fahrer nacheinander und sind Zusagen bindend, ist nur Online-Matching möglich. | **Online-Matching** (gebaut) |
| **Kleine bis mittlere Graphen** | Diese Version braucht O(n³); für riesige Graphen gibt es asymptotisch schnellere Verfahren (Gabows Skalierung). | (nicht in der Linie) |
"""
)
st.caption("Damit ist der ganze Verbesserungswege-Ast der Matching-Linie fertig (Wurzel, Verbesserungswege, Hopcroft–Karp, Ungarische Methode, Auktionsalgorithmus, Blossom, Gewichteter Blossom). Offen in der ganzen Linie bleibt nur noch Stabile Mitbewohner.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Allgemeiner Graph $G=(V,E)$ mit Kosten $c_{ij}\ge 0$. Gesucht: eine Paarung $M$, erst mit größtmöglicher Kardinalität, unter diesen mit kleinster Kostensumme.

**Ersatzgewicht.** Der Löser maximiert intern; bei „erst Paarzahl, dann Kosten" $iw(i,j)=(c_{\max}+1)-c_{ij}$ (immer $\ge 1$, macht jede Kante lohnend - die interne Maximierung bevorzugt darum automatisch möglichst viele Paare); ohne Nebenbedingung an die Anzahl $iw(i,j)=-c_{ij}$.

**Dualwerte.** $y_v$ je Ecke (verdoppelt: $y_v=2u(v)$, deshalb immer ganzzahlig, auch wenn $u(v)$ halbzahlig ist), $z_B\ge 0$ je Blüte. Straffheit im internen Maximum: $y_i+y_j+2\sum_{B\ni i,j} z_B - 2\,iw(i,j)\ge 0$, Gleichheit auf gewählten Kanten.

**Zurückgerechnet auf echte Kosten** (Minimierungs-Dualität, dieselbe Konvention wie bei der Ungarischen Methode - Kosten minus Dualwerte $\ge 0$): mit „erst Paarzahl, dann Kosten" gilt für jede mögliche Kante
$$y_i+y_j+2\sum_{B\ni i,j} z_B + 2\,c_{ij} \;\ge\; 2\,(c_{\max}+1),$$
Gleichheit auf jeder gewählten Kante; ohne Nebenbedingung an die Anzahl entfällt die rechte Seite (sie wird 0). Die **Primal-Dual-Identität** (Gegenprobe, wie bei der Ungarischen Methode die Kostenidentität): summiert man diese Gleichheit über alle gewählten Kanten, ergibt sich
$$\sum_{v\in M} y_v + \sum_B z_B\cdot\frac{|B|-1}{2}\cdot 2 \;=\; (c_{\max}+1)\cdot 2|M| - 2\sum_{(i,j)\in M} c_{ij}$$
(nur über gepaarte Ecken - freie Ecken tragen bei „erst Paarzahl" einen eigenen, nicht notwendig verschwindenden Dualwert). Stimmt die Summe, ist $M$ bewiesen die billigste Paarung ihrer Größe.

**Delta-Schritte.** Kommt der Wald nicht weiter, wird der kleinste von vier möglichen Schritten angewendet: δ1 (ein freier Dualwert erreicht 0 - Suche endet, nur ohne Nebenbedingung an die Anzahl), δ2 (kleinster Schlupf gerade↔frei), δ3 (kleinster Schlupf/2 gerade↔gerade über Bäume), δ4 (kleinster Dualwert über Blüten, die dabei mitten im Suchlauf wieder aufgeklappt werden).

**Konvergenz.** Ist $G$ bipartit (zwei getrennte Seiten), entstehen nie Blüten - der Algorithmus ist dann genau die Ungarische Methode. Ohne Gewichte (alle Kosten gleich) reduziert er sich auf den ungewichteten Blossom-Algorithmus.

**Aufwand.** $O(n^3)$ für die einfache Version mit Basis-Feldern; mit Mitgliederlisten je Blüte $O(n\cdot|E|)$ für die Wald-Suche allein, dazu $O(n)$ Delta-Schritte.

Implementiert in `wb_scenario.py` (Karten, eigener Zufallsgenerator, feste Karten, bipartite Konstruktion), `wb_greedy.py`, `wb_blossom.py` (Primal-Dual-Kern mit Dualwerten, Ereignisprotokoll, Beweis), `wb_oracle.py` (Brute Force), `wb_evaluation.py` (Kennzahlen, Verteilungen, Sweeps).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
