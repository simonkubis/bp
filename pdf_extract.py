import fitz
import json

# ---- Konfigurácia ----
PDF_PATH    = "catalog.pdf"
OUTPUT_PATH = "parsed_catalog.json"


# Zozbiera všetky veľkosti písma v dokumente
def collect_font_sizes(doc):
    sizes = set()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes.add(round(span["size"], 1))
    return sorted(sizes, reverse=True)

def extract_hierarchical_text(pdf_path):
    doc = fitz.open(pdf_path)
    sizes = collect_font_sizes(doc)

    if len(sizes) < 3:
        raise RuntimeError("Dokument nemá dostatok úrovní písma")

    # Tri hlavné úrovne: názov vzoru, číslovanie, nadpis sekcie
    L1, L2, L3 = sizes[0], sizes[1], sizes[2]

    document = []
    current = None

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
                if not text:
                    continue

                size = max(round(s["size"], 1) for s in spans)

                if size == L1:
                    # Nadpis vzoru – prípadne viacriadkový, tak sa zlúči
                    if current and current.get("_open"):
                        current["title"] += " " + text
                    else:
                        current = {"title": text, "content": "", "_open": True}
                        document.append(current)

                elif size in (L2, L3):
                    if current:
                        current["content"] += " " + text

                else:
                    # Bežný text obsahu
                    if current:
                        current["content"] += " " + text

        # Uzavrie otvorený nadpis na konci strany
        if current and current.get("_open"):
            current.pop("_open")

    return {
        "textual_descriptions": document
    }


if __name__ == "__main__":
    output = extract_hierarchical_text(PDF_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Extrakcia dokončená → {OUTPUT_PATH}")