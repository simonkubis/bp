import streamlit as st
import json
import random
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# =============================================================================
# KONFIGURÁCIA
# =============================================================================

DEFAULT_PARSED_CATALOG = "parsed_catalog.json"
ANALYZE_SCRIPT        = "analyze_patterns.py"
EDGES_FILE            = "pattern_edges.json"


# Prahy pre zaradenie kandidátov do zlúčenín a sekvencií
COMPOUND_THRESHOLD  = 0.1
SEQUENCE_THRESHOLD  = 0.05

# Prahy pre klasifikáciu vzťahov na silné, stredné a slabé
STRONG_THRESHOLD   = 0.6
MEDIUM_THRESHOLD   = 0.3

# Maximálna dĺžka generovanej sekvencie vzorov
MAX_SEQUENCE_LENGTH = 10

# Seed pre reprodukovateľnosť náhodnej prechádzky
RANDOM_SEED = 42

# =============================================================================
# INICIALIZÁCIA STAVU RELÁCIE
# =============================================================================

# Všetky stavové premenné sa inicializujú raz pri prvom spustení aplikácie
if "markov_chain"    not in st.session_state:
    st.session_state.markov_chain    = None
if "pattern_names"   not in st.session_state:
    st.session_state.pattern_names   = []
if "bayesian_matrix" not in st.session_state:
    st.session_state.bayesian_matrix = None
if "log_messages"    not in st.session_state:
    st.session_state.log_messages    = []

# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def log_message(msg):
    """Uloží interný ladiaci výpis do stavu relácie (nezobrazuje sa používateľovi)."""
    st.session_state.log_messages.append(msg)


def classify_posterior_strength(posterior_prob,
                                 strong_threshold=STRONG_THRESHOLD,
                                 medium_threshold=MEDIUM_THRESHOLD):
    """
    Klasifikuje silu vzťahu na základe hodnoty posteriornej pravdepodobnosti.
    Vracia trojicu (sila, farebný indikátor v Markdown, slovenský popis).
    """
    if posterior_prob >= strong_threshold:
        return "strong", ":green[●]", "silný"
    elif posterior_prob >= medium_threshold:
        return "medium", ":orange[●]", "stredný"
    else:
        return "weak",   ":red[●]",   "slabý"

# =============================================================================
# NAČÍTAVANIE DÁT
# =============================================================================

def load_parsed_catalog(file_path):
    """
    Načíta súbor parsed_catalog.json s názvami a opismi vzorov.
    Vráti slovník alebo None pri chybe.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_message(f"Katalóg načítaný: {file_path}")
        return data
    except Exception as e:
        log_message(f"Chyba pri načítavaní katalógu: {e}")
        return None


def run_analyze_script(catalog_path):
    """
    Spustí externý skript analyze_patterns.py, ktorý z katalógu vypočíta
    hrany (vzťahy) medzi vzormi a uloží ich do pattern_edges.json.
    Vráti True pri úspechu, False pri chybe.
    """
    try:
        result = subprocess.run(
            ["python", ANALYZE_SCRIPT],
            capture_output=True,
            text=True,
            cwd=Path(".").resolve()
        )
        if result.returncode == 0:
            log_message("Skript analýzy bol úspešne spustený.")
            return True
        else:
            log_message(f"Chyba skriptu: {result.stderr}")
            return False
    except Exception as e:
        log_message(f"Nepodarilo sa spustiť skript: {e}")
        return False


def load_pattern_edges(file_path=EDGES_FILE):
    """
    Načíta súbor pattern_edges.json vygenerovaný skriptom analyze_patterns.py.
    Vráti slovník s kľúčom 'edges' alebo None pri chybe.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_message(f"Načítaných {len(data.get('edges', []))} hrán.")
        return data
    except Exception as e:
        log_message(f"Chyba pri načítavaní hrán: {e}")
        return None

# =============================================================================
# BAYESOVSKÁ ANALÝZA
# =============================================================================

def calculate_uniform_prior(num_patterns):
    """
    Vypočíta rovnomernú apriórnu pravdepodobnosť pre každý vzor.
    Predpokladáme, že pred analýzou sú si všetky vzory rovnako pravdepodobné.
    """
    return 1.0 / num_patterns if num_patterns > 0 else 0.0


def build_likelihood_matrix(pattern_names, edges):
    """
    Zostaví maticu vierohodnosti L[i, j] = P(B | A) z načítaných hrán.
    Hodnoty pochádzajú z poľa combined_score, ktoré skript analyze_patterns.py
    priradí každej hrane na základe textovej analýzy opisov vzorov.
    Riadky = zdrojový vzor A, stĺpce = cieľový vzor B.
    """
    n           = len(pattern_names)
    pattern_idx = {p.lower(): i for i, p in enumerate(pattern_names)}
    likelihood  = np.zeros((n, n))

    for edge in edges:
        src   = edge["source"].lower()
        tgt   = edge["target"].lower()
        score = edge.get("combined_score", 0.0)

        if src in pattern_idx and tgt in pattern_idx:
            i = pattern_idx[src]
            j = pattern_idx[tgt]
            likelihood[i, j] = score

    return likelihood, pattern_idx


def calculate_posterior(prior, likelihood):
    """
    Vypočíta posteriorné pravdepodobnosti pomocou Bayesovho pravidla.

    Pre každý zdrojový vzor i a cieľový vzor j platí:
        P(j | i) = L[i,j] * P(j) / Z
    kde Z je normalizačná konštanta (suma čitateľov cez všetky j).

    Keďže prior je rovnomerný (rovnaká hodnota pre každé j), zjednoduší sa na:
        P(j | i) = L[i,j] / sum_k( L[i,k] )

    Diagonála sa vynuluje (vzor sa neodkazuje sám na seba) a riadky sa
    renormalizujú, aby súčet pravdepodobností zostal 1.
    """
    n         = likelihood.shape[0]
    posterior = np.zeros_like(likelihood, dtype=float)

    for i in range(n):
        row_lik    = likelihood[i, :]
        numerator  = row_lik * prior          # prior je skalár, aplikuje sa na celý riadok
        denominator = np.sum(numerator)

        if denominator > 0:
            posterior[i, :] = numerator / denominator
        else:
            # Záložné riešenie: ak pre vzor neexistujú žiadne dôkazy, použijeme rovnomernú distribúciu
            posterior[i, :] = np.ones(n) / n

    # Odstraňujeme samoprechody (vzor → rovnaký vzor)
    np.fill_diagonal(posterior, 0.0)

    # Renormalizácia po vynulovaní diagonály
    row_sums = posterior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0             # Ochrana pred delením nulou
    posterior = posterior / row_sums

    return posterior


def create_markov_chain(posterior, pattern_names):
    """
    Prevedie posteriornu maticu na Markovov reťazec reprezentovaný
    ako vnorený slovník: chain[zdrojový_vzor][cieľový_vzor] = pravdepodobnosť.
    """
    chain = {}
    for i, src in enumerate(pattern_names):
        src_lower        = src.lower()
        chain[src_lower] = {}
        for j, tgt in enumerate(pattern_names):
            if i != j:
                prob = posterior[i, j]
                if prob > 0:
                    chain[src_lower][tgt.lower()] = prob

    log_message(f"Markovov reťazec vytvorený s {len(chain)} zdrojovými vzormi.")
    return chain

# =============================================================================
# ANALÝZA VZOROV – ZLÚČENINY A SEKVENCIE
# =============================================================================

def find_compound_patterns(chain, input_patterns, threshold=COMPOUND_THRESHOLD):
    """
    Nájde vzory, ktoré majú obojsmerný vzťah so všetkými zadanými vzormi.

    Podmienka: pre každý vstupný vzor musí existovať nenulová pravdepodobnosť
    v oboch smeroch (vpred aj vzad) s kandidátom, a priemer musí presiahnuť prah.
    """
    input_lower = [p.lower() for p in input_patterns]
    compounds   = {}

    for tgt in chain:
        if tgt in input_lower:
            continue

        scores = []
        for src in input_lower:
            if src in chain:
                fwd = chain[src].get(tgt, 0)
                bwd = chain.get(tgt, {}).get(src, 0)
                if fwd > 0 and bwd > 0:
                    scores.append((fwd + bwd) / 2)

        # Kandidát musí spĺňať podmienku pre každý vstupný vzor
        if len(scores) > 0:
            avg = sum(scores) / len(scores)
            if avg >= threshold:
                compounds[tgt] = avg

    return compounds


def find_sequence_patterns(chain, input_patterns, max_length=MAX_SEQUENCE_LENGTH):
    random.seed(RANDOM_SEED)
    input_lower     = [p.lower() for p in input_patterns]
    sequence        = list(input_lower)
    visited         = set(input_lower)

    # Zozbierame všetkých kandidátov prvého kroku zo všetkých vstupných vzorov
    seed_candidates = []
    seed_weights    = []

    for p in input_lower:
        if p in chain:
            for tgt, prob in chain[p].items():
                if tgt not in visited and prob >= SEQUENCE_THRESHOLD:
                    seed_candidates.append(tgt)
                    seed_weights.append(prob)

    if not seed_candidates:
        return sequence

    # Stochastický výber prvého kroku — konzistentné s popisom náhodnej prechádzky v práci
    current = random.choices(seed_candidates, weights=seed_weights, k=1)[0]

    # Zvyšok prechádzky 
    while current and len(sequence) < max_length:
        sequence.append(current)
        visited.add(current)

        if current not in chain:
            break

        options = [
            t for t in chain[current]
            if t not in visited and chain[current][t] >= SEQUENCE_THRESHOLD
        ]
        if not options:
            break

        weights = [chain[current][t] for t in options]
        current = random.choices(options, weights=weights, k=1)[0]

    return sequence

# =============================================================================
# POUŽÍVATEĽSKÉ ROZHRANIE – BOČNÝ PANEL
# =============================================================================

st.set_page_config(layout="wide", page_title="Objavovanie vzorov")
st.title("Objavovanie spoločne použiteľných vzorov")

with st.sidebar:
    st.subheader("Výber textových opisov vzorov")

    # Prepínač medzi predvoleným katalógom a vlastným súborom
    use_default = st.checkbox(
        "Použiť predvolené textové opisy bezpečnostných vzorov",
        value=True
    )

    if use_default:
        catalog_path = DEFAULT_PARSED_CATALOG
        st.info(f"Používa sa: `{DEFAULT_PARSED_CATALOG}`")
    else:
        uploaded_file = st.file_uploader(
            "Nahraj vlastný súbor textových opisov vzorov pomenovaný parsed_catalog.json",
            type=["json"]
        )
        if uploaded_file:
            catalog_path = uploaded_file.name
            with open(catalog_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success(f"Nahrané: {uploaded_file.name}")
        else:
            catalog_path = None

    st.divider()

# =============================================================================
# POUŽÍVATEĽSKÉ ROZHRANIE – NAČÍTAVANIE A SPRACOVANIE
# =============================================================================

if st.button("Načítaj", use_container_width=True):
    st.session_state.log_messages = []

    if catalog_path and Path(catalog_path).exists():

        with st.status("Prebieha analýza...", expanded=True) as status:

            # Krok 1: Načítanie katalógu vzorov
            st.write("Načítavam katalóg...")
            catalog = load_parsed_catalog(catalog_path)

            if not catalog:
                status.update(label="Chyba pri načítavaní katalógu.", state="error")
                st.stop()

            # Krok 2: Spustenie skriptu na identifikáciu explicitných vzťahov
            st.write("Identifikujem explicitné vzťahy medzi vzormi...")
            if not run_analyze_script(catalog_path):
                status.update(label="Chyba pri analýze.", state="error")
                st.stop()

            # Krok 3: Načítanie hrán vygenerovaných skriptom
            st.write("Načítavam generované hrany...")
            edges_data = load_pattern_edges()

            if not edges_data:
                status.update(label="Chyba pri načítavaní hrán.", state="error")
                st.stop()

            pattern_names = catalog.get("pattern_names", [])
            edges         = edges_data.get("edges", [])

            # Krok 4: Bayesovské výpočty
            st.write("Počítam bayesovskú posterioriu...")
            prior                    = calculate_uniform_prior(len(pattern_names))
            likelihood, pattern_idx  = build_likelihood_matrix(pattern_names, edges)
            posterior                = calculate_posterior(prior, likelihood)

            # Krok 5: Zostrojenie Markovovho reťazca
            st.write("Vytváram Markovov reťazec...")
            chain = create_markov_chain(posterior, pattern_names)

            # Uloženie výsledkov do stavu relácie pre ďalšie použitie v UI
            st.session_state.markov_chain    = chain
            st.session_state.pattern_names   = pattern_names
            st.session_state.bayesian_matrix = posterior

            status.update(label="Hotovo!", state="complete")
            st.success(f"Hotovo! {len(pattern_names)} vzorov načítaných.")

    else:
        st.error("Súbor nebol nájdený alebo nebol vybraný.")

st.divider()

# =============================================================================
# POUŽÍVATEĽSKÉ ROZHRANIE – INTERAKTÍVNA ANALÝZA VZOROV
# =============================================================================

if st.session_state.markov_chain:
    st.subheader("Zakladanie sekvencií a zlúčenín vzorov")

    input_text = st.text_area(
        "Zadajte názvy vzorov (oddelené čiarkami)",
        placeholder="napr. Mutual Authentication, Asymmetric Encryption, Secure Channels",
        height=50
    )
    input_patterns = [p.strip() for p in input_text.split(",") if p.strip()]

    if input_patterns:
        chain            = st.session_state.markov_chain
        known_patterns   = [p for p in input_patterns if p.lower() in chain]
        unknown_patterns = [p for p in input_patterns if p.lower() not in chain]

        if unknown_patterns:
            st.warning(f"Nenájdené v dátovej sade: {', '.join(unknown_patterns)}")

        if known_patterns:

            # --- Párová analýza (len pri viac ako jednom vzore) ---
            if len(known_patterns) >= 2:
                st.markdown("### Dvojice vzorov")

                for i, p1 in enumerate(known_patterns):
                    for p2 in known_patterns[i + 1:]:
                        p1_lower = p1.lower()
                        p2_lower = p2.lower()

                        fwd = chain.get(p1_lower, {}).get(p2_lower, 0)
                        bwd = chain.get(p2_lower, {}).get(p1_lower, 0)

                        if fwd > 0 or bwd > 0:
                            # Klasifikujeme smer vpred aj vzad osobitne
                            _, fwd_color, _ = classify_posterior_strength(fwd)
                            _, bwd_color, _ = classify_posterior_strength(bwd)
                            st.markdown(
                                f"{fwd_color} **{p1}** ↔ **{p2}**: → {fwd:.4f} | ← {bwd:.4f}"
                            )

                st.divider()

            # --- Zlúčeniny (vzory kompatibilné so všetkými zadanými) ---
            st.markdown("### Zlúčeniny vzorov (Top 5)")
            compounds = find_compound_patterns(chain, known_patterns)

            if compounds:
                top_5 = sorted(compounds.items(), key=lambda x: x[1], reverse=True)[:5]
                for pattern, score in top_5:
                    _, color_indicator, strength_label = classify_posterior_strength(score)
                    st.caption(
                        f"{color_indicator} ↔ **{pattern.title()}** "
                        f"(skóre: {score:.4f} – {strength_label})"
                    )
            else:
                st.info("Žiadne zlúčeniny vzorov nenájdené.")

            st.divider()

            # --- Sekvenčné rozšírenie zadaných vzorov ---
            st.markdown("### Sekvencia vzorov (max 5)")
            sequence = find_sequence_patterns(chain, known_patterns, max_length=5)

            if len(sequence) > len(known_patterns):
                st.code(" -> ".join([p.title() for p in sequence]))
            else:
                st.info("Žiadna rozšírená sekvencia nenájdená.")

    else:
        st.info("Zadajte názvy vzorov.")

else:
    st.info(
        "Vyberte si vlastný alebo predvolený súbor textových opisov vzorov "
        "a kliknite na tlačidlo 'Načítaj'."
    )