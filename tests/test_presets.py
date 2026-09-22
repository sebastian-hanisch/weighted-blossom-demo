"""Presets: vollständig, in den Grenzen, und jede Beispielkarte zeigt, was ihr Hilfetext behauptet."""

import pytest

import wb_constants as C
import wb_evaluation as ev
import wb_presets as P
from wb_scenario import build

KEYS = set(P.PRESET_KEYS)


def _a(p):
    return ev.analyse(build(p["net"], p["n"], p["reach"], p["ballung"], p["seed"]), p["objective"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["objective"] in C.OBJECTIVE_LABELS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["reach"] - C.REACH_MIN) % 5 == 0 and p["ballung"] % 25 == 0


def test_setting_specs_have_room_to_move():
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)
    assert set(P.KEPT) <= set(P.SETTING_SPECS)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_the_default_map_is_the_medium_preset():
    p = C.PRESETS["🗺️ Mittlere Karte"]
    assert (p["n"], p["reach"], p["ballung"], p["seed"], p["objective"]) == (30, 20, 0, C.DEFAULT_SEED, C.OBJ_LEX)
    level, code, d = ev.verdict(_a(p))
    assert (level, code) == ("success", ev.OPTIMAL) and (d["count"], d["cost"]) == (13, 145)


def test_the_presets_show_what_their_help_says():
    v = {name: ev.verdict(_a(p))[2] for name, p in C.PRESETS.items()}
    assert (v["🪆 Bezahlte, verschachtelte Blüten"]["n_paid_blossoms"], v["🪆 Bezahlte, verschachtelte Blüten"]["depth"]) == (2, 2)
    assert (v["🌸 Blüte mit Stiel"]["contractions"], v["🌸 Blüte mit Stiel"]["count"]) == (0, 5)          # kommt ganz ohne Kontraktion aus
    wind = v["🌀 Windmühle"]
    assert (wind["depth"], wind["contractions"], wind["n_paid_blossoms"]) == (4, 4, 0)                   # tief verschachtelt, aber am Ende unbezahlt
    bip = v["🔗 Bipartit"]
    assert bip["contractions"] == 0 and bip["premium"] > 0
    med = v["🗺️ Mittlere Karte"]
    assert (med["count"], med["cost"], med["premium"]) == (13, 145, 33)
    exp = v["🌪️ Braucht Aufklappung"]
    assert exp["expansions"] >= 1 and exp["cert_ok"]
    ign = v["💸 Kosten ignoriert"]
    assert ign["premium"] > med["premium"]
    off = v["🚫 Ohne Kardinalitätszwang"]
    assert (off["count"], off["cost"]) == (0, 0) and off["blind_count"] > 0


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"🪆 Bezahlte, verschachtelte Blüten", "🌸 Blüte mit Stiel", "🌀 Windmühle", "🔗 Bipartit"}
