neutral_keywords = [
    "see also pattern",
    "used with",
    "apply before",
    "apply after",
    "related to",
    "related pattern",
    "similar pattern",
    "alternative to",
    "alternatively used pattern"
]

negative_keywords = [
    "do not use",
    "never use",
    "avoid using",
    "not recommended",
    "not to be used"
]

positive_keywords = [
    "always use",
    "must use",
    "recommended",
    "best practice",
    "should use",
    "this pattern uses",
    "pattern is an alternative"
]

KEYWORD_WEIGHTS = {
    "negative": -1.0,
    "neutral": 0.5,
    "positive": 1.0
}

import json
import re
import networkx as nx
import matplotlib.pyplot as plt

# Load the patterns
with open('./patterns.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = data['patterns']
pattern_titles = [p['title'].lower() for p in patterns]
pattern_texts = [p['text'].lower() for p in patterns]    

relationships = {
    title: {} for title in pattern_titles
}

def contains_all_words(words, text):
    return all(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

def normalize_text(text):
    return " ".join(text.lower().split())

def set_weight(a, b, weight):
    relationships[a][b] = weight
    relationships[b][a] = weight

def update_weight(src, tgt, delta):
    relationships[src][tgt] = relationships[src].get(tgt, 0) + delta

for i, pattern_text in enumerate(pattern_texts):
    pattern_title = pattern_titles[i]

    for sentence in pattern_text.split('.'):
        sentence_norm = normalize_text(sentence)

        for kw in neutral_keywords:
            kw_words = kw.split()

            if contains_all_words(kw_words, sentence):
                print(f"DEBUG: Found words: {kw_words} in sentence: '{sentence.strip()}'")
                for other_pattern in pattern_titles:
                    print(f"---------- DEBUG: Checking other pattern: '{other_pattern} in {sentence} | State:  {other_pattern in sentence_norm}'")
                    if other_pattern in sentence_norm and other_pattern not in pattern_title:
                        print(f"+++---+++---+++---+++---DEBUG: Updating NEUTRAL weight between '{pattern_title}' and '{other_pattern}'")
                        update_weight(
                            pattern_title,
                            other_pattern,
                            KEYWORD_WEIGHTS["neutral"]
                        )


G = nx.DiGraph()

for src, targets in relationships.items():
    for tgt, weight in targets.items():
        if weight != 0:
            G.add_edge(src, tgt, weight=weight)

# --- VISUALIZATION ---
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42)

edges = G.edges(data=True)

edge_colors = []
edge_widths = []

for _, _, data in edges:
    w = data["weight"]
    if w < 0:
        edge_colors.append("red")
    elif w > 0:
        edge_colors.append("green")
    else:
        edge_colors.append("gray")
    edge_widths.append(abs(w) * 2)

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#E8E8E8")
nx.draw_networkx_labels(G, pos, font_size=9)
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    width=edge_widths,
    alpha=0.7,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=20,
    connectionstyle="arc3,rad=0.05"
)

# Draw edge weights as labels
edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

plt.title("Pattern Relationship Graph", fontsize=14)
plt.axis("off")
plt.show()

