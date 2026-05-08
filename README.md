# Objavovanie spoločne použiteľných bezpečnostných vzorov

Aplikácia na zakladanie sekvencií a zlúčenín vzorov, ktoré sú očakávané byť použité spoločne, využívajúc bezpečnostné vzory, s interaktívnym rozhraním cez Streamlit a lokálne spustená aplikácia.

## Čo aplikácia robí
- načíta spracovaný katalóg vzorov
- identifikuje vzťahy medzi vzormi
- vytvorí výstupný súbor `pattern_edges.json`
- zobrazí výsledky v Streamlit aplikácii

## Požiadavky
- Python 3.8 alebo novší
- `requirements.txt`

## Inštalácia

1. Stiahni Python zo stránky:
   `https://www.python.org/downloads/`

2. V priečinku projektu vytvor virtuálne prostredie:
   ```bash
   python -m venv .venv
   ```

3. Aktivuj `.venv`:
   - PowerShell:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - CMD:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - Bash:
     ```bash
     source .venv/Scripts/activate
     ```

4. Nainštaluj závislosti:
   ```bash
   pip install -r requirements.txt
   ```

5. Spusti aplikáciu:
   ```bash
   streamlit run app.py
   ```

6. Otvor prehliadač na adrese:
   `http://localhost:8501`

## Voliteľné
Ak potrebujete vytvoriť `parsed_catalog.json` z PDF, použite:
```bash
python pdf_extract.py
```

## Základné súbory
- `app.py` – hlavný Streamlit skript, klasifikácia vzťahov, pravdepodobnostný model, Markovovský reťazec, generovanie sekvencií a zlúčenín
- `analyze_patterns.py` – analýza vzťahov medzi vzormi
- `pdf_extract.py` – extrakcia textu z PDF do JSON
- `catalog.pdf` - katalóg bezpečnostných vzorov vo formáte PDF
- `parsed_catalog.json` – spracovaný vstupný katalóg vzorov vo formáte JSON
- `pattern_edges.json` – súbor všetkých identifikovaných vzťahov
- `requirements.txt` – zoznam závislostí
- `README.md` - tento súbor
