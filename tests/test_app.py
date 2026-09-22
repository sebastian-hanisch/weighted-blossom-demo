"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, beide Ziele (auch beim Abspielen auf mehrbildrigen Karten),
Randgrößen, Schritt-Zustand, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import wb_constants as C
from wb_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

EXPECTED = {
    "🪆 Bezahlte, verschachtelte Blüten": "5 Paare für 125 Minuten - bewiesen billigste",
    "🌸 Blüte mit Stiel": "5 Paare für 114 Minuten - bewiesen billigste",
    "🌀 Windmühle": "4 Paare für 32 Minuten - bewiesen billigste",
    "🔗 Bipartit": "7 Paare für 108 Minuten - bewiesen billigste",
    "🗺️ Mittlere Karte": "13 Paare für 145 Minuten - bewiesen billigste",
    "🌪️ Braucht Aufklappung": "2 Paare für 20 Minuten - bewiesen billigste",
    "💸 Kosten ignoriert": "14 Paare für 143 Minuten - bewiesen billigste",
    "🚫 Ohne Kardinalitätszwang": "„Nur Kosten“ ohne Nebenbedingung an die Anzahl: 0 Paare für 0 Minuten",
}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _set(**kw):
    def setup(at):
        for k, v in kw.items():
            at.session_state[k] = v
    return setup


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _step(at):
    found = [s for s in at.slider if s.key == "wb_step"]
    return found[0] if found else None


def _keys(at, kind):
    return [w.key for w in getattr(at, kind)]


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_default_renders_without_exception():
    at = _run()
    assert any("Ziel und Ablauf" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🗺️ Mittlere Karte"]) and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _has(at, EXPECTED[name]), _texts(at)
    if C.PRESETS[name]["net"] in C.FIXED_NETS:
        assert any(t.startswith("Feste Karte") for t in _texts(at))
    elif C.PRESETS[name]["objective"] == C.OBJ_LEX:
        assert any(m.label.startswith("Paare (Mittel") for m in at.metric)


@pytest.mark.parametrize("objective", list(C.OBJECTIVE_LABELS))
def test_every_objective_renders_every_kind_of_step(objective):
    at = _run(_set(objective_radio=objective))
    assert not at.error
    last = int(_step(at).max)
    assert last > 0 and _step(at).value == last
    for k in (0, 1, 2, last // 3, last // 2, last - 1, last):
        _step(at).set_value(k)
        at.run()
        assert not at.exception, (objective, k, [e.value for e in at.exception])


@pytest.mark.parametrize("objective", list(C.OBJECTIVE_LABELS))
def test_play_runs_through_all_frames_without_duplicate_chart_keys(objective):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression:
    StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run(_set(objective_radio=objective))
    assert _step(at).max > 0
    _play(at)


@pytest.mark.parametrize("net", list(C.FIXED_NETS))
def test_play_on_fixed_maps(net):
    at = _run(_set(net_select=net))
    assert _step(at).max >= 2
    _play(at)


def test_no_feasible_pair_renders():
    from wb_scenario import generate
    seed = next(k for k in range(500) if generate(C.N_MIN, C.REACH_MIN, 0, k).m == 0)
    at = _run(_set(n_slider=C.N_MIN, reach_slider=C.REACH_MIN, seed_input=seed))
    assert any(t.startswith("Kein einziges Paar ist möglich") for t in _texts(at))


def test_extreme_sizes_render():
    for n, reach in ((C.N_MIN, C.REACH_MIN), (C.N_MAX, C.REACH_MAX), (C.N_MIN, C.REACH_MAX), (C.N_MAX, C.REACH_MIN)):
        at = _run(_set(n_slider=n, reach_slider=reach))
        step = _step(at)
        assert step is not None and step.value == step.max


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(_set(net_select=net)))
    random_labels, fixed = labels_for("random"), labels_for("windmuehle")
    assert {"Karte", "Fahrer", "Reichweite [min]", "Ballung [%]", "Zufalls-Seed"} <= random_labels
    assert fixed == {"Karte"}


def test_hidden_slider_values_come_back_when_the_random_map_is_shown_again():
    at = _run(_set(reach_slider=90, n_slider=24))
    at.session_state["net_select"] = "windmuehle"
    at.run()
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="reach_slider").value == 90 and at.slider(key="n_slider").value == 24


def test_step_slider_returns_to_the_last_step_when_the_map_or_the_objective_changes():
    at = _run()
    last = int(_step(at).max)
    assert _step(at).value == last
    _step(at).set_value(3)
    at.run()
    assert _step(at).value == 3
    at.session_state["net_select"] = "windmuehle"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max and _step(at).max < last
    at.session_state["net_select"] = "random"
    at.run()
    _step(at).set_value(2)
    at.run()
    at.session_state["objective_radio"] = "cost"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max


def test_first_step_shows_the_start_and_the_last_the_proof():
    at = _run()
    _step(at).set_value(0)
    at.run()
    assert not at.exception and any(m.value.startswith("**Nach 0 von") for m in at.markdown)
    assert not any("Primal-Dual-Identität" in t.value.to_dict("list").get("Bestandteil", []) for t in at.table)
    _step(at).set_value(int(_step(at).max))
    at.run()
    proof = next(t.value.to_dict("list") for t in at.table if "Primal-Dual-Identität" in t.value.to_dict("list").get("Bestandteil", []))
    assert len(proof["Bestandteil"]) == 4 and all(v.startswith("✅") for v in proof["Ergebnis"])


def test_a_delta_step_shows_a_touched_table():
    at = _run(_set(net_select="verschachtelt"))
    seen_delta_table = False
    for k in range(1, int(_step(at).max) + 1):
        _step(at).set_value(k)
        at.run()
        assert not at.exception
        if any("Art" in t.value.columns and "vorher" in t.value.columns for t in at.table):
            seen_delta_table = True
    assert seen_delta_table


def test_events_of_every_kind_appear_on_the_expansion_map():
    at = _run(_set(n_slider=7, reach_slider=20, seed_input=18))
    seen = set()
    for k in range(1, int(_step(at).max) + 1):
        _step(at).set_value(k)
        at.run()
        assert not at.exception
        text = " ".join(m.value for m in at.markdown)
        for key, marker in (("round", "eine neue Stufe beginnt"), ("grow", "wird ungerade"), ("blossom", "schließt eine Blüte"), ("augment", "ist ein Verbesserungsweg"),
                            ("delta", "Kein weiterer Fortschritt"), ("expand", "wird mitten im Suchlauf wieder aufgeklappt")):
            if marker in text:
                seen.add(key)
    assert seen == {"round", "grow", "blossom", "augment", "delta", "expand"}


def test_permalink_parameters_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "9999"
    at.query_params["ballung"] = "abc"
    at.query_params["n"] = "-5"
    at.run()
    assert not at.exception
    assert at.slider(key="reach_slider").value == C.REACH_MAX and at.slider(key="ballung_slider").value == C.DEFAULT_BALLUNG and at.slider(key="n_slider").value == C.N_MIN
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "42"
    at.query_params["ballung"] = "60"
    at.run()
    assert at.slider(key="reach_slider").value == 40 and at.slider(key="ballung_slider").value == 50


def test_permalink_keeps_valid_objectives_and_falls_back_for_unknown_ones():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["objective"] = "cost"
    at.run()
    assert not at.exception and at.radio(key="objective_radio").value == "cost"
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.query_params["objective"] = "chaos"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET and at.radio(key="objective_radio").value == C.DEFAULT_OBJECTIVE


def test_randomize_moves_the_seed_but_not_the_distribution():
    at = _run()
    before = {m.label: m.value for m in at.metric if "Mittel" in m.label}
    seed_before = at.number_input(key="seed_input").value
    [b for b in at.sidebar.button if "Neue Karte" in b.label][0].click()
    at.run()
    assert not at.exception and at.number_input(key="seed_input").value != seed_before
    after = {m.label: m.value for m in at.metric if "Mittel" in m.label}
    assert before and before == after


def test_experiments_run_on_demand():
    at = _run()
    markers = ("desto teurer wird es, die Kosten zu ignorieren", "mittlere Grad bei jeder Größe etwa 3 bleibt")
    assert not any(any(m in c.value for m in markers) for c in at.caption)
    for key in ("reach_start", "scale_start", "brute_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert all(m in text for m in markers)


def _calls(src, name):
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_the_play_loop_keys_carry_the_step():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    assert len(calls) == 5 and all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    stepped = [c for c in calls if 'key=f"' in c]
    assert len(stepped) == 2 and {re.search(r'key=f"([a-z_]+?)_\{', c).group(1) for c in stepped} == {"wb_map", "wb_progress"}
    unstepped = [k for c, k in zip(calls, keys) if 'key=f"' not in c]
    assert len(set(unstepped)) == len(unstepped) == 3, unstepped
    viz = (ROOT / "wb_visualization.py").read_text(encoding="utf-8")
    bodies = [b for b in viz.split(chr(10) + "def ") if b.startswith("build_")]
    assert "fixedrange=True" in viz and all("_base(" in b or "_map_layout(" in b for b in bodies)


def test_maps_have_no_legend_and_keep_equal_scale_inside_the_domain():
    import wb_blossom as B
    import wb_scenario as S
    import wb_visualization as V
    sc = S.build("verschachtelt", 0, 0, 0, 0)
    res = B.run(sc, maxcardinality=True, record=True)
    for k in (0, res.n_events // 2, res.n_events):
        fig = V.build_map(sc, res, k)
        assert fig.layout.showlegend is False and fig.layout.xaxis.constrain == "domain" and fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_footer_is_verbatim():
    src = APP.read_text(encoding="utf-8")
    assert "https://sebastianhanisch.net/kontakt.html" in src and "Interesse an einer maßgeschneiderten Lösung für" in src and "Operations Research und Machine Learning" in src


def test_runtime_needs_only_numpy_pandas_plotly_streamlit():
    """Konvention der Konzepte-Wurzeln und -Stücke: Referenzbibliotheken (scipy, networkx) nur als Testorakel."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" not in req and "networkx" not in req
    for path in ROOT.glob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+(networkx)\b", path.read_text(encoding="utf-8"), re.M), path.name
    ev_src = (ROOT / "wb_evaluation.py").read_text(encoding="utf-8")
    assert "from scipy" in ev_src                                             # scipy ist hier absichtlich Testorakel-artig genutzt (Bipartit-Gegenprobe), nicht Laufzeitpflicht
