import json
import re
from collections import defaultdict

# =============================================================================
# KONŠTANTY
# =============================================================================

INPUT_FILE = "parsed_catalog.json"
OUTPUT_PATH = "pattern_edges.json"

# Koeficienty pre kombináciu signálov pri výpočte skóre
KEYWORD_COEF = 1.0
MENTION_COEF = 0.3

# Kľúčové slová naznačujúce vzťah medzi vzormi
KEYWORDS = [
    "related to", "related pattern", "see also", "similar to",
    "alternative to", "variant of", "variation of", "analogous to",
    "comparable to", "can be combined with", "can be used",
    "often used with", "in conjunction with", "together with",
    "along with", "pairs well with", "frequently paired with",
    "commonly paired with", "may be used alongside", "can complement",
    "complements", "is complementary to", "contrast with",
    "compare with", "compared to", "see", "see pattern", "see also pattern", "is an alternative to",
    "always use", "must use", "should use", "is recommended",
    "recommended to use", "best practice", "is required",
    "preferred over", "is preferable to", "makes use", "pattern uses",
    "requires", "uses", "is based on", "builds upon", "builds on",
    "extends", "relies on", "depends on", "calls", "employs",
    "implements", "is combined with", "is used", "works with",
    "works alongside", "leverages", "makes use of",
    "is built on top of", "is built on", "is an extension of",
    "is a specialization of", "is a refinement of", "enables",
    "facilitates", "supports", "is supported by", "is enabled by",
    "is facilitated by", "is powered by", "is composed with",
    "is composed of", "encapsulates", "wraps", "decorates",
    "delegates to", "is delegated to", "integrates with",
    "is integrated with", "is typically combined with",
    "is commonly used with", "is naturally used with",
    "orchestrates", "is orchestrated by", "coordinates with",
    "mediates between", "abstracts", "is abstracted by",
    "is composed into", "composes", "to implement"
]


# =============================================================================
# POMOCNÉ FUNKCIE
# =============================================================================

def normalize_text(text):
    # Prevedenie na malé písmená a zjednotenie medzier
    return " ".join(text.lower().split())


def contains_phrase(phrase, sentence):
    # Overenie, či sa fráza nachádza v texte ako celé slovo
    return re.search(rf"\b{re.escape(phrase)}\b", sentence)


# =============================================================================
# POČÍTANIE VÝSKYTOV
# =============================================================================

def compute_mention_counts(document_structure, valid_patterns):
    # Počítanie, koľkokrát sa názov každého vzoru objaví v texte ostatných vzorov
    pattern_titles = [p.lower() for p in valid_patterns]

    pattern_texts = {}
    for doc in document_structure:
        title = doc["title"].lower()
        if title not in pattern_titles:
            continue
        content = doc.get("content", "")
        # Odstránenie interpunkcie pred vyhľadávaním
        cleaned = re.sub(r"[^\w\s]", " ", normalize_text(content))
        pattern_texts[title] = cleaned

    raw_counts = defaultdict(lambda: defaultdict(int))
    for src, text in pattern_texts.items():
        for tgt in pattern_titles:
            if tgt == src:
                continue
            tgt_clean = re.sub(r"[^\w\s]", " ", tgt).strip()
            count = len(re.findall(rf"\b{re.escape(tgt_clean)}\b", text))
            if count > 0:
                raw_counts[src][tgt] = count

    return raw_counts


def normalize_mention_counts(raw_counts):
    # Normalizácia počtov výskytov do rozsahu 0–1 (relatívne k maximu pre daný zdroj)
    normalized = defaultdict(dict)
    for src, targets in raw_counts.items():
        if not targets:
            continue
        max_count = max(targets.values())
        for tgt, count in targets.items():
            normalized[src][tgt] = count / max_count
    return normalized

def normalize_keyword_weights(keyword_relationships):
    """
    Normalizuje surové počty zhôd kľúčových slov do rozsahu [0, 1]
    voči globálnemu maximu — rovnaká škála ako mention_score.
    """
    all_values = [
        w for targets in keyword_relationships.values()
        for w in targets.values()
    ]
    if not all_values:
        return keyword_relationships

    global_max = max(all_values)
    if global_max == 0:
        return keyword_relationships

    normalized = defaultdict(lambda: defaultdict(float))
    for src, targets in keyword_relationships.items():
        for tgt, weight in targets.items():
            normalized[src][tgt] = weight / global_max

    return normalized


# =============================================================================
# ANALÝZA KĽÚČOVÝCH SLOV
# =============================================================================

def analyze_keywords(document_structure, valid_patterns):
    # Prehľadávanie viet v texte každého vzoru a detekcia vzťahov pomocou kľúčových slov
    pattern_titles = [p.lower() for p in valid_patterns]
    # Verzie názvov vzorov bez interpunkcie pre porovnávanie v texte
    pattern_titles_clean = {
        p: re.sub(r"[^\w\s]", " ", p).strip()
        for p in pattern_titles
    }
    relationships = defaultdict(lambda: defaultdict(float))

    for doc in document_structure:
        pattern_title = doc["title"].lower()
        if pattern_title not in pattern_titles:
            continue

        content = normalize_text(doc.get("content", ""))
        # Rozdelenie textu na vety podľa interpunkcie
        sentences = re.split(r'(?<=[.!?])\s+', content)

        for sentence in sentences:
            sentence_clean = re.sub(r"[^\w\s]", " ", sentence.strip())
            if not sentence_clean:
                continue

            for kw in KEYWORDS:
                kw_clean = re.sub(r"[^\w\s]", " ", kw).strip()
                if contains_phrase(kw_clean, sentence_clean):
                    for other_pattern, other_clean in pattern_titles_clean.items():
                        if (
                            re.search(rf"\b{re.escape(other_clean)}\b", sentence_clean)
                            and other_pattern != pattern_title
                        ):
                            # Jednotný príspevok za každý nájdený vzťah
                            relationships[pattern_title][other_pattern] += 1.0

    return relationships


# =============================================================================
# ZOSTAVENIE HRÁN GRAFU
# =============================================================================

def build_edges(keyword_relationships, mention_counts_norm):
    # Kombinovanie signálov z kľúčových slov a výskytov do jedného skóre na hranu
    all_pairs = defaultdict(lambda: {
        "keyword_weight": 0.0,
        "mention_score": 0.0,
    })

    for src, targets in keyword_relationships.items():
        for tgt, kw_weight in targets.items():
            all_pairs[(src, tgt)]["keyword_weight"] += kw_weight

    for src, targets in mention_counts_norm.items():
        for tgt, mention_score in targets.items():
            all_pairs[(src, tgt)]["mention_score"] += mention_score

    edges = []

    for (src, tgt), components in all_pairs.items():
        kw_weight = components["keyword_weight"]
        mention_score = components["mention_score"]
        # Výsledné skóre hrany ako vážený súčet oboch normalizovaných signálov
        combined_score = KEYWORD_COEF * kw_weight + MENTION_COEF * mention_score
        
        edge = {
            "source": src,
            "target": tgt,
            "keyword_weight": round(kw_weight, 6),
            "mention_score": round(mention_score, 6),
            "combined_score": round(combined_score, 6),
        }
        edges.append(edge)

    return edges


# =============================================================================
# EXPORT
# =============================================================================

def export_edges(edges, path=OUTPUT_PATH):
    # Uloženie hrán do JSON súboru
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"edges": edges}, f, indent=2)
    print(f"Uložených {len(edges)} hrán -> {path}")


# =============================================================================
# HLAVNÁ FUNKCIA
# =============================================================================

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    textual_descriptions = data["textual_descriptions"]
    pattern_names = [d["title"] for d in textual_descriptions]

    # Spustenie analýzy kľúčových slov a počítania výskytov
    keyword_relationships     = analyze_keywords(textual_descriptions, pattern_names)
    keyword_relationships_norm = normalize_keyword_weights(keyword_relationships)   # nové

    mention_counts_raw        = compute_mention_counts(textual_descriptions, pattern_names)
    mention_counts_norm       = normalize_mention_counts(mention_counts_raw)

    edges = build_edges(keyword_relationships_norm, mention_counts_norm)  # normalizované vstupy

    export_edges(edges)


if __name__ == "__main__":
    main()