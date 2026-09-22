# Gewichteter Blossom – die billigste Paarung in einem allgemeinen Graphen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-weighted-edmonds-matching-demo.streamlit.app/)**

Neuntes Stück der **Matching-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", die **Konvergenz** zweier Vorgänger: [hungarian-demo](https://github.com/sebastian-hanisch/hungarian-demo) (Ungarische Methode: Gewichte, aber nur bipartit) und [blossom-demo](https://github.com/sebastian-hanisch/blossom-demo) (Blüten, aber nur ungewichtet). **Gewichteter Blossom** (Edmonds' Primal-Dual-Algorithmus, in der Literatur auch Galils Algorithmus – genau das, was `networkx.max_weight_matching` implementiert) trägt beides zusammen: dasselbe Fahrgemeinschaften-Szenario wie `blossom-demo` bekommt jetzt echte Kosten (Fahrzeit), und die Suche liefert nicht nur ein größtmögliches, sondern das **billigste** Matching – mit **Beweis**: zu jeder Ecke und jeder (womöglich verschachtelten) Blüte ein Dualwert, straff auf jeder gewählten Kante, nie negativ.

Wie überall in der Linie ist das Ziel **lexikografisch**: erst größtmögliches Matching, dann geringste Kosten. Nach diesem Stück (der Konvergenz von Ungarisch und Blossom) folgte noch der Gale-Shapley-Ast mit seinen vier Erweiterungen (Stabile Mitbewohner, Krankenhaus-Zulassung, Top Trading Cycles, Nierentausch) – die gesamte Matching-Linie ist inzwischen vollständig gebaut.
```
greedy-matching-demo (Wurzel: eine gewählte Zuordnung bleibt)                     [gebaut]
  ├─ augmenting-path-demo (Verbesserungswege: Paare optimal, Kosten blind)        [gebaut]
  │    ├─ hopcroft-karp-demo (viele kürzeste Wege je Phase)                       [gebaut]
  │    ├─ hungarian-demo (Ungarische Methode: Paare zuerst, dann Kosten)           [gebaut]
  │    │    └─ auction-algorithm-demo (Auktionsalgorithmus: dezentral)             [gebaut]
  │    └─ blossom-demo (allgemeine Graphen: ungerade Kreise, Kontraktion)          [gebaut]
  │   Ungarisch + Blossom → weighted-blossom-demo (Gewichteter Blossom)            [dieses Stück]
  ├─ gale-shapley-demo (Vorlieben statt Kosten, stabil)                            [gebaut]
  │    ├─ stabile-mitbewohner-demo (eine Gruppe statt zwei Seiten)                 [gebaut]
  │    ├─ krankenhaus-zulassung-demo (many-to-one, Kapazitäten)                    [gebaut]
  │    ├─ top-trading-cycles-demo (Tausch ohne Geld, Wohnungsmarkt)                [gebaut]
  │    └─ nierentausch-demo (Kompatibilität statt Präferenz, kurze Zyklen)         [gebaut]
  └─ online-matching-demo (Aufträge kommen nacheinander)                           [gebaut]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` (und, wo es feste Karten sind, in `tests/test_blossom.py`) über die 100 festen Karten (Seeds 100000–100099) belegt: 30 Fahrer, Reichweite 20 (mittlerer Grad 2,97). Aufwand = **angesehene Kanten**, nie Sekunden. Mittel und Median stehen zusammen.

| Frage | Ergebnis |
|---|---|
| Die Karten | ✅ Optimum im Mittel **13,37** Paare (Median 13, 11 bis 15), Kosten im Mittel **147,3** Minuten (Median 149,5). Die Paarzahl hängt nie von den Kosten ab – auf allen 100 Karten dieselbe Größe wie eine kostenblinde Paarung. |
| Was bringt es, die Kosten zu berücksichtigen? | ✅ Eine kostenblinde größtmögliche Paarung (alle Kanten gleich teuer) kostet im Mittel **179,72** Minuten (Median 178,0) – **32,42** Minuten (Median 31,0) mehr, auf **allen 100** Karten teurer, im Extrem 2 bis 80 Minuten. |
| Je größer die Reichweite, desto teurer wird Ignorieren | ✅ Reichweite 10 / 20 / 30 / 40 / 60 (Grad 0,83 / 3,04 / 6,14 / 9,98 / 17,60): Prämie im Mittel **4,13 / 31,40 / 100,48 / 182,13 / 342,85** Minuten – streng steigend. |
| Der Beweis | ✅ Auf **allen 100** Karten straff und die Primal-Dual-Identität exakt: im Mittel 6,36 Kontraktionen (Median 6), 4,55 bezahlte Blüten (Median 4, auf jeder Karte mindestens eine), Verschachtelungstiefe im Mittel 2,76 (Median 3). |
| Aufklappung mitten im Suchlauf | ⚠️ Selten, aber real: im Mittel 0,44 je Karte (Median 0), auf **44** der 100 Karten mindestens eine, höchstens 5 auf einer einzelnen Karte. |
| Aufwand | ⚠️ Im Mittel **653,87** angesehene Kanten (Median 654,0) – deutlich mehr als das ungewichtete `blossom-demo` (dort im Mittel 52,32 vom Greedy-Start), weil jede Runde jetzt vier Delta-Typen statt einer einzigen Wachstumsregel abwägt. |
| Aufwand gegen Größe | ⚠️ Grad ≈ 3 konstant, n = 10 / 20 / 40 / 80 / 160 / 320: **94 / 307 / 1087 / 2806 / 7660 / 29545** angesehene Kanten – ungefähr n^1,66, steiler als das ungewichtete Blossom. |
| Ohne Kardinalitätszwang ist das Ergebnis leer | ✅ Reines Minimum ohne Nebenbedingung an die Anzahl: Seed 100000, n = 30, Reichweite 20 – mit Zwang **14 Paare für 171** Minuten, ohne Zwang **0 Paare für 0** Minuten (keine Nullkosten-Kante auf dieser Karte). Fahrzeit ist immer positiv, also schlägt „niemanden paaren" fast immer jede echte Paarung – genau deshalb gilt „erst Paarzahl, dann Kosten" in der ganzen Linie. |
| Bipartit = Ungarisch | ✅ 100 zufällige bipartite Karten (5 Größen × 20 Seeds) stimmen exakt mit `scipy.optimize.linear_sum_assignment` überein (Paarzahl **und** Kosten): **100 von 100**. |
| Gegen Brute Force | ✅ 20 kleine Karten (n = 10): **20 von 20** stimmen mit dem eigenen Bitmasken-Löser überein. |

## Was nicht funktioniert hat / widerlegte Vorab-Hypothesen

- **„Kosten verdoppeln" ist ein Vorverarbeitungsschritt."** Nein: die Verdopplung steckt allein in der internen Speicherung des Eckendualwerts (`y[v] = 2 · y_echt(v)`), nie in den Eingabekosten und nie im Blütendualwert `z[B]` (echter Maßstab). Für zwei gerade Ecken in verschiedenen Bäumen ist `δ3 = slack(u,v) // 2` – die Ganzzahligkeit folgt aus den Invarianten des Verfahrens selbst, nicht aus einer Konstruktion der Eingabe.
- **Die K4-Kostenprobe: 37 statt 33.** Ohne die `mybestedges`-Konsolidierung beim Kontrahieren einer Blüte stürzt nichts ab – das Ergebnis hat die richtige Paarzahl, aber falsche (zu hohe) Kosten. Auf K4 mit Kosten `01:3, 02:13, 03:11, 12:22, 13:24, 23:30` ist **33** die nachweislich billigste Paarung (`{0,1}+{2,3}`), **37** wäre die stille, zu teure Fehlantwort. Der gefährlichste Fehlerpfad des ganzen Stücks, weil „stürzt nicht ab" ihn nicht fängt.
- **Der ursprüngliche n=8-Beispielgraph löst mitten im Suchlauf eine Aufklappung aus.** Nicht unter der hier verwendeten Minimierungs-Konvention – die Aufklappung fällt bei diesem konkreten Graphen mit den angegebenen Kosten immer erst ans Stufenende. Das kleinste tatsächlich verifizierte Beispiel ist eine kleine, eigens dafür gebaute Karte (7 Fahrer, Reichweite 20, Seed 18, 2 isolierte Fahrer).
- **Die Prämie gegen ein cost-blindes Matching lässt sich einfach aus dem ungewichteten `blossom-demo`-Ergebnis herleiten (Kreuz-Import).** Verworfen: CI checkt pro Lauf nur ein Repo aus, ein Import über Repo-Grenzen wäre nicht lauffähig. Stattdessen läuft `blind_scenario` mit derselben Engine (`wb_blossom.run`) auf einer Kopie der Karte mit Einheitskosten – eine andere, **strengere** Vergleichsgröße (im Mittel 32,42 statt der zuvor grob geschätzten ≈ 5,55 Minuten), weil sie nicht nur „eine andere größte Paarung", sondern konsequent die **teuerste unter den größtmöglichen** vergleicht.
- **Reine Zulässigkeit der Dualwerte reicht als Beweis.** Nein: bei `maxcardinality=True` können Eckendualwerte negativ werden, ein zulässiges, aber nicht paarzahl-optimales Dualpaar besteht die reine Zulässigkeitsprüfung trotzdem. Zusätzlich prüft das Zertifikat die **Primal-Dual-Identität** (`Σ y[matched] + Σ z[B]·(|B|-1) = shift·|Paare| − 2·Kosten`) – das fängt, was reine Zulässigkeit nicht fängt.

## Was die Demo zeigt

- **Suche und Ablauf:** Schritt-Slider und ▶️ über die Ereignisse (Runde, Wald wächst, Blüte kontrahiert, Delta-Schritt, Aufklappung, Verbesserung): die Karte mit dualwertbeschrifteten Ecken, Blüten als durchscheinenden konvexen Hüllen mit ihrem Dualwert, und einer Tabelle je Delta-Schritt (welche Ecke/Blüte betroffen, Dualwert vorher/nachher); am Ende der Beweis (Straffheit, gewählte Kanten straff, alle Blütendualwerte ≥ 0, Primal-Dual-Identität).
- **Wie gut?** Paare, Kosten, angesehene Kanten, Kontraktionen, Aufklappungen, Verschachtelungstiefe, bezahlte Blüten; Vergleich gegen die kostenblinde größtmögliche Paarung (Prämie in Minuten); Verteilung über 100 feste Karten.
- **Wovon hängt es ab?** Reichweiten-Sweep (Prämie gegen mittleren Grad), Aufwand gegen Größe (n = 10 bis 320), Brute-Force-Beweis auf 20 kleinen Karten.
- **Feste Karten:** Bezahlte, verschachtelte Blüten · Blüte mit Stiel · Windmühle · Bipartit (= Ungarisch); dazu Presets, die gezielt eine Aufklappung mitten im Suchlauf, eine besonders große Prämie und die Entartung ohne Kardinalitätszwang zeigen. **Wo die Annahmen enden.**

## Modell und Verfahren

- **Graph:** dieselbe Fahrgemeinschaften-Karte wie `blossom-demo` (Fahrer als Punkte, Kante bei Fahrzeit ≤ Reichweite, `SplitMix64`), jetzt mit ihren echten Fahrzeiten als Kosten statt nur als Kante/keine Kante. Neu: eine **bipartite** Konstruktion (zwei Punktgruppen, keine Kante innerhalb einer Gruppe, garantiert durch Konstruktion) für den direkten Ungarisch-Vergleich.
- **Gewichteter Blossom (`wb_blossom.py`):** interne Zielumkehr `iw = (max_kosten+1) − Kosten` (bei Kardinalitätszwang) bzw. `iw = −Kosten` (ohne Zwang), damit das Verfahren wie üblich maximiert; Zustand: Eckendualwert `y[v]` (verdoppelt), Blütendualwert `z[B]` (unverdoppelt, je auch verschachtelter Blüte), Label (gerade/ungerade/keins), `base[]`/`members[base]`, `bestedge`/`mybestedges`-Konsolidierung beim Kontrahieren. Wald auf straffen Kanten (`slack==0`) wachsen lassen; wächst nichts mehr, den kleinsten von vier Delta-Schritten anwenden (δ1 freier Eckendualwert, nur ohne Kardinalitätszwang; δ2 gerade↔frei; δ3 gerade↔gerade über Bäume; δ4 Blütendualwert erreicht 0 → **Aufklappung mitten im Suchlauf**).
- **Zertifikat:** für jede Kante straff oder locker (`y[i]+y[j]+2·Σz+2·Kosten(i,j) ≥ shift`, Gleichheit auf jeder gewählten Kante, alle `z[B] ≥ 0`), plus die Primal-Dual-Identität – reine Zulässigkeit reicht nicht, s. o.
- **Aufwand:** angesehene Kanten, Kontraktionen, kontrahierte Ecken, Aufklappungen, Verschachtelungstiefe – nie Sekunden.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `wb_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `wb_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik (Standardmuster des Portfolios) |
| `wb_scenario.py` | Fahrer-Karten (aus `blossom-demo` übernommen), **neu:** garantiert bipartite Konstruktion |
| `wb_greedy.py` | Startpaarungen (aus `blossom-demo` übernommen) |
| `wb_blossom.py` | **Neu:** Gewichteter Blossom (Wald, vier Delta-Schritte, Kontraktion mit `bestedge`-Konsolidierung, Aufklappung, Zertifikat), Ereignisprotokoll mit Schnappschüssen |
| `wb_oracle.py` | **Neu:** Brute Force (Bitmasken-Dynamik) für Tests |
| `wb_evaluation.py` | Einordnung, kostenblinder Vergleich, Verteilung, Reichweiten-Sweep, Aufwand gegen Größe, Bipartit-/Brute-Force-Gegenprobe |
| `wb_visualization.py` | Plotly-Abbildungen (Blüten-Hüllen wie `blossom-demo`, Dualwert-Anzeige, Delta-Schritt-Hervorhebung; Achsen gesperrt, keine Legende auf Karten) |
| `tests/` | Algorithmus (networkx, Brute Force, Bipartit gegen scipy, Zertifikat-Prüfer, Negativkontrollen inkl. der beiden dokumentierten Fehlerpfade), Auswertung, Presets, belegte Zahlen, AppTest-Rauchtests |

Die kopierten Bausteine werden durch Tests bewacht (Zufallsgenerator-Vektor, eine festgeschriebene Punktliste). Alle Daten sind synthetisch; die Laufzeit braucht nur numpy, pandas, plotly und streamlit (scipy und networkx sind reine Testorakel, nie zur Laufzeit importiert).

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Die Logik rechnet ausschließlich mit ganzen Zahlen; die im Text genannten Anteile und Mittelwerte sind deshalb auf jeder Plattform identisch.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
