# Objavovanie spoločne použiteľných bezpečnostných vzorov

Systém na analýzu a klasifikáciu vzťahov medzi bezpečnostnými vzormi prostrednictvom kombinovaného prístupu využívajúceho textovú analýzu, detekciu menovaní a Bayesovskú inferenciu. Aplikácia umožňuje identifikáciu explicitných vzťahov medzi vzormi, aposteriórne zaradenie vzťahov do kategórií a generovanie stochastických sekvencií vzorov na základe Markovovho reťazca.

## Požiadavky a inštalácia

### Systémové požiadavky
- Python 3.8+
- pip (správca balíkov)

### Závislosť

```bash
pip install streamlit numpy pandas PyMuPDF
```

**Verzie komponentov:**
- Streamlit ≥ 1.0
- NumPy ≥ 1.19
- Pandas ≥ 1.2
- PyMuPDF ≥ 1.18

### Spustenie aplikácie

```bash
streamlit run app.py
```

Webové rozhranie je prístupné na adrese `http://localhost:8501`

## Popis systému

### Účel a aplikácia

Systém je určený na podporu analýzy bezpečnostných vzorov (security patterns) prostredníctvom automatizovanej detekcie vzťahov. Primárnym prínosom je:

- Identifikácia skrytých vzťahov medzi vzormi na základe textovej analýzy
- Rozlíšenie sily vzťahov pomocou Bayesovskej klasifikácie
- Generovanie sekvenčných kombinácií vzorov na základe pravdepodobnostného modelu

### Používateľské rozhranie

Aplikácia poskytuje tri primárne funkcie:

1. **Voľba katalógu** – Záložka pre výber preddefiniovaného katalógu alebo nahratie vlastného súboru `parsed_catalog.json`
2. **Analýza** – Inicializácia spracovateľského potrubia
   - Automatická extrakcia vzťahov
   - Výpočet aposteriórnych pravdepodobností
   - Konštrukcia Markovovho reťazca
3. **Interaktívne dopytovanie** – Zadanie zoznamu vzorov na analýzu:
   - **Páry vzťahov** – Obojsmerné vzťahy s kategorizáciou sily (🟢 silný ≥ 0.3, 🟠 stredný 0.1–0.3, 🔴 slabý < 0.1)
   - **Súbežné vzory** – Identifikácia Top-5 vzorov s najvyššou spoločnou frekvenciou
   - **Generovaná sekvencia** – Stochastická prechádzka Markovovým reťazcom

## Architektúra a komponentný model

### Štruktúra projektového súboru

```
├── app.py                      # Streamlit webové rozhranie
├── analyze_patterns.py         # Modul analýzy vzťahov
├── pdf_extract.py             # Modul extrakcii z PDF
├── parsed_catalog.json        # Katálog vzorov (vstup)
├── pattern_edges.json         # Graf vzťahov (výstup)
└── app/                        # Demonštračný balík
    ├── streamlit_app.py
    ├── utils.py
    └── [...demo móduly]
```

### Komponentný popis

| Komponent | Typ | Funkcia | Iniciátor |
|-----------|-----|---------|-----------|
| `app.py` | Modul | Orchestrácia celého spracovateľského potrubia | Užívateľ |
| `analyze_patterns.py` | Modul | Detekcia vzťahov a generovanie hrán grafu | `app.py` |
| `pdf_extract.py` | Modul | Parseovanie PDF a extrakcia textovej reprezentácie | Administrátor (jednorazovo) |
| `parsed_catalog.json` | Dátový artefakt | Štruktúrovaný katalóg vzorov | `pdf_extract.py` |
| `pattern_edges.json` | Dátový artefakt | Detektovaná množina vzťahov s váhami | `analyze_patterns.py` |

### Spracovateľské potrubie

```
parsed_catalog.json
         ↓
   [ANALÝZA VZŤAHOV]
   (analyze_patterns.py)
         ↓
   pattern_edges.json
         ↓
   [BAYESOVSKÁ INFERENCIE]
         ↓
   [MARKOVOV REŤAZEC]
         ↓
   [WEBOVÉ ROZHRANIE]
```

## Príprava a inicializácia dát

### Prvotná extrakcia z PDF

Proces prípravy dát je vykonávaný jednorazovo pred spustením hlavnej aplikácie:

```bash
# 1. Umiestnite PDF súbor katalógu do projektovej zložky:
#    (predvolané meno: catalog.pdf)

# 2. Spustite extraktor:
python pdf_extract.py

# 3. Výsledný súbor:
#    parsed_catalog.json
```

**Výstup modulu `pdf_extract.py`:**
- Štruktúrovaný JSON súbor s extrahovanými textovými opismi
- Zachovanie pôvodnej hierarchie a metadát dokumentu

### Formát vstupného katalógu

Modul `pdf_extract.py` očakáva PDF súbor obsahujúci:
- Štitky vzorov ako názvy sekcií
- Textové opisy vzoru ako obsah sekcií
- Konzistentnú formátovanie dokumentu

## Metodológia spracovania a analýzy

Spracovateľský proces je rozdelený do päť nezávislých fáz, z ktorých každá má špecifickú zodpovednosť v rámci detekcie a klasifikácie vzťahov.

### Fáza 1: Extrakcia a štruktúrácia textových dát

**Zodpovedný modul:** `pdf_extract.py`

Extraktor aplikuje heuristické pravidlá založené na metrických vlastnostiach textu (veľkosť písma, pozícia, formátovanie):

1. Identifikácia názvov vzorov (sekundárne nadpisy)
2. Agregácia textového obsahu patriaceho danému vzoru
3. Normalizácia a čistenie textových reťazcov
4. Serializácia do štandardného JSON formátu

**Výstupný formát:**
```json
{
  "pattern_names": ["Vzor A", "Vzor B", "..."],
  "textual_descriptions": [
    {
      "title": "Vzor A",
      "content": "Komplexný textový opis vzoru..."
    }
  ]
}
```

### Fáza 2: Detekcia explicitných vzťahov

**Zodpovedný modul:** `analyze_patterns.py`

Detekcia vychádza z kombinácie dvoch nezávislých signálov:

#### Signal 1: Analýza relačných kľúčových slov

Modul analyzuje výskyt predefinovaných relačných fráz (45+ položiek):
- Frázy naznačujúce prím direktnú závislosť („vyžaduje", „predpokladá", „implementuje")
- Frázy naznačujúce vzájomné vzťahy („pracuje s", „kombinuje", „rozširuje")
- Frázy naznačujúce komplementaritu („alternácia k", „kontrast s")

Frekvencia výskytov je vážená podľa kontextuálneho typu frázy.

#### Signal 2: Počítanie menovaní

Detekcia exaktného výskytu názvov ostatných vzorov v textovej reprezentácii daného vzoru:

$$\text{mention\_score}_{i,j} = \frac{\text{count}_{\text{name}_j \text{ in description}_i}}{\text{total\_mentions}_i}$$

#### Kombinovaná váha

Finálna váha vzťahu je linárna kombinácia oboch signálov:

$$\text{combined\_score}_{i,j} = w_k \times \text{keyword\_weight}_{i,j} + w_m \times \text{mention\_score}_{i,j}$$

kde:
- $w_k = 1.0$ – koeficient váhy relačných kľúčových slov
- $w_m = 0.3$ – koeficient váhy menovaní

### Fáza 3: Bayesovská inferencie a posteriórne pravdepodobnosti

**Vstup:** Množina vzťahov s kombinovanými váhami z Fázy 2

**Postup:**

Aplikácia Bayesovho pravidla s jednotným prior:

$$P(j|i) = \frac{L_{i,j}}{\sum_{k=1}^{n} L_{i,k}}$$

kde:
- $L_{i,j}$ = kombinovaná váha vzťahu od vzoru $i$ k vzoru $j$ (likelihood)
- $P(j|i)$ = aposteriórna pravdepodobnosť vzoru $j$ daného vzoru $i$
- $n$ = počet vzorov v katalógu

**Vlastnosti modelu:**
- Jednotný prior: $P(j) = \frac{1}{n}$ pre všetky $j$
- Normalizácia: $\sum_{j} P(j|i) = 1$ pre všetky $i$
- Symetrizácia: $S_{i,j} = \max(P(j|i), P(i|j))$ pre párové vzťahy

### Fáza 4: Klasifikácia sily vzťahov

**Zodpovedný modul:** Funkcia `classify_posterior_strength()` v `app.py`

Automatická klasifikácia aposteriórnych pravdepodobností do troch kategórií:

| Kategória | Prah | Interpretácia | Indikátor |
|-----------|------|---------------|-----------|
| **Silný** | $P(j\|i) \geq 0.3$ | Výrazne podporovaný vzťah | 🟢 Zelená |
| **Stredný** | $0.1 \leq P(j\|i) < 0.3$ | Umierneného podporovaný vzťah | 🟠 Oranžová |
| **Slabý** | $P(j\|i) < 0.1$ | Nízko podporovaný vzťah | 🔴 Červená |

**Logika klasifikácie:**

```
IF P(j|i) >= STRONG_THRESHOLD:
    category = STRONG
ELIF P(j|i) >= MEDIUM_THRESHOLD:
    category = MEDIUM
ELSE:
    category = WEAK
```

### Fáza 5: Markovov reťazec a stochastická generácia

**Model:**

Markovov reťazec je reprezentovaný ako orientovaný vážený graf:
- **Stavy:** Bezpečnostné vzory
- **Prechody:** Aposteriórne pravdepodobnosti $P(j|i)$
- **Inicializácia:** Rovnomerné rozdelenie cez všetky vzory

**Generovanie sekvencií:**

1. Náhodný výber počiatočného vzoru
2. Iterácia podľa prechodných pravdepodobností
3. Filtrujúci prah: Eliminácia prechodov s pravdepodobnosťou $< 0.15$
4. Zastavovacia podmienka: Maximálna dĺžka sekvencie alebo stagnácia

**Generovanie zlúčenín:**

Vzory sú klasifikované ako zlúčeniny podľa:
- Obojsmerný vzťah: $S_{i,j} = \max(P(j|i), P(i|j))$
- Prah: $S_{i,j} \geq 0.1$

## Konfigurácia a nastaviteľné parametre

### Globálne konškanty

Konfigurovateľné hodnoty sú umiestnené v jednotlivých moduloch:

#### `analyze_patterns.py`
```python
KEYWORD_COEF = 1.0      # Váha signálu z kľúčových slov (skúmania kvázi-lineárne)
MENTION_COEF = 0.3      # Váha signálu z počítania menovaní
```

#### `app.py`
```python
STRONG_THRESHOLD = 0.3      # Prah klasifikácie: silný vzťah
MEDIUM_THRESHOLD = 0.1      # Prah klasifikácie: stredný vzťah
COMPOUND_THRESHOLD = 0.1    # Minimálna pravdepodobnosť pre zlúčeniny
SEQUENCE_THRESHOLD = 0.15   # Minimálna pravdepodobnosť pre prechody v sekvencii
```

### Možnosti prispôsobenia

Zvýšenie váhy menovaní zlepšuje detekciu explicitných referencií:
```python
MENTION_COEF = 0.5  # Zvýšená váha → viac vzťahov z menovaní
```

Zníženie SEQUENCE_THRESHOLD umožňuje generovanie dlhších sekvencií:
```python
SEQUENCE_THRESHOLD = 0.10   # Zníženej prah → dlhšie sekvencie
```

## Dátové formáty

### Vstupný formát: `parsed_catalog.json`

```json
{
  "pattern_names": [
    "Vzor 1",
    "Vzor 2"
  ],
  "textual_descriptions": [
    {
      "title": "Vzor 1",
      "content": "Detailný textový opis vzoru..."
    },
    {
      "title": "Vzor 2",
      "content": "Detailný textový opis vzoru..."
    }
  ]
}
```

### Výstupný formát: `pattern_edges.json`

```json
{
  "edges": [
    {
      "source": "Vzor A",
      "target": "Vzor B",
      "keyword_weight": 2.5,
      "mention_score": 0.75,
      "combined_score": 2.725,
      "posterior_probability": 0.156,
      "strength": "medium"
    }
  ]
}
```

**Popis polí:**
- `source`, `target`: Identifikátory vzorov
- `keyword_weight`: Váha z kľúčových slov (Fáza 2, Signal 1)
- `mention_score`: Váha z menovaní (Fáza 2, Signal 2)
- `combined_score`: Kombinovaná váha (Fáza 2)
- `posterior_probability`: Bayesovská aposteoriorna pravdepodobnosť (Fáza 3)
- `strength`: Klasifikačná kategória (Fáza 4)

## Rozšírené technické aspekty

### Komplexnosť algoritmu

- **Fáza 2 – Detekcia vzťahov:** $O(n^2 \cdot m)$, kde $n$ = počet vzorov, $m$ = priemerná dĺžka textu
- **Fáza 3 – Normalizácia:** $O(n^2)$
- **Fáza 5 – Generovanie sekvencií:** $O(k)$, kde $k$ = dĺžka sekvencie

### Stabilita a reprodukovateľnosť

- Textová analýza je deterministická (nie je potrebný seed random generátora)
- Bayesovská inferencie je stabilná za predpokladu konzistentného vstupu
- Markovov reťazec vyžaduje nastavenie seed generátora pre reprodukovateľnosť

## Odporúčané postupy a best practices

1. **Validácia katalógu:** Pred spustením analýzy skontrolujte súlad a konzistenciu `parsed_catalog.json`
2. **Monitorovanie váh:** Analyzujte distribúciu `combined_score` v `pattern_edges.json` na detekciu anomálií
3. **Testovanie prahov:** Experiment s hodnotami `STRONG_THRESHOLD` a `MEDIUM_THRESHOLD` za účelom optimalizácie klasifikácie
4. **Dokumentácia verzií:** Uchovávania verzie katalógu a generovaných súborov pre audovateľnosť

## Licencia a autorstvo

Projekt je implementáciou metodológie z bakalárskej práce zameranej na objavovanie vzťahov medzi bezpečnostnými vzormi. Metodológia kombinuje textovú analýzu s Bayesovským pravdepodobnostným modelom na vytvorenie opakovateľného a transparentného systému.