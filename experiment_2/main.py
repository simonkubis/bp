import json
import re

with open('./patterns.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = data['patterns']

pattern_titles = [p['title'].lower() for p in patterns]
pattern_texts = [p['text'].lower() for p in patterns]

occurrences = {
    title: {} for title in pattern_titles
}

def normalize_text(text):
    return " ".join(text.lower().split())

for i, pattern_text in enumerate(pattern_texts):
    pattern_norm = normalize_text(pattern_text)
    current_title = pattern_titles[i]

    for other_title in pattern_titles:
        if other_title == current_title:
            continue 

        count = len(re.findall(rf'\b{re.escape(other_title)}\b', pattern_norm))

        if count > 0:
            occurrences[current_title][other_title] = count

# ---- PRINT RESULTS ----

for pattern, refs in occurrences.items():
    print(f"\nPattern: {pattern}")
    if not refs:
        print("  (no other patterns mentioned)")
    else:
        for mentioned, count in refs.items():
            print(f"  mentions '{mentioned}': {count}")
