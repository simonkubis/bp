import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load the patterns
with open('./patterns.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = data['patterns']
pattern_titles = [p['title'] for p in patterns]
pattern_texts = [p['text'] for p in patterns]

print("="*80)
print("PATTERN ANALYSIS USING TF-IDF")
print("="*80)
print(f"\nTotal patterns loaded: {len(patterns)}")
print(f"Pattern titles: {pattern_titles}\n")

# ============================================================================
# 1. BASIC TEXT STATISTICS
# ============================================================================
print("\n" + "="*80)
print("1. BASIC TEXT STATISTICS")
print("="*80)

for i, (title, text) in enumerate(zip(pattern_titles, pattern_texts)):
    word_count = len(text.split())
    char_count = len(text)
    print(f"\n{i+1}. {title}")
    print(f"   Words: {word_count:,}")
    print(f"   Characters: {char_count:,}")
    print(f"   Avg word length: {char_count/word_count:.2f}")

# ============================================================================
# 2. TF-IDF VECTORIZATION WITH DIFFERENT PARAMETERS
# ============================================================================
print("\n\n" + "="*80)
print("2. TF-IDF VECTORIZATION")
print("="*80)

# Create vectorizer with comprehensive parameters
vectorizer = TfidfVectorizer(
    max_features=200,  # Top 200 terms
    min_df=1,          # Minimum document frequency
    max_df=0.8,        # Maximum document frequency (ignore very common terms)
    ngram_range=(1, 3), # Unigrams, bigrams, and trigrams
    stop_words='english',
    lowercase=True
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(pattern_texts)
feature_names = vectorizer.get_feature_names_out()

print(f"\nTF-IDF Matrix Shape: {tfidf_matrix.shape}")
print(f"(Documents: {tfidf_matrix.shape[0]}, Features: {tfidf_matrix.shape[1]})")
print(f"\nTotal unique terms extracted: {len(feature_names)}")

# ============================================================================
# 3. TOP TERMS PER PATTERN
# ============================================================================
print("\n\n" + "="*80)
print("3. TOP TF-IDF TERMS PER PATTERN")
print("="*80)

# Convert to dense array for easier manipulation
tfidf_dense = tfidf_matrix.toarray()

for idx, title in enumerate(pattern_titles):
    print(f"\n{title}:")
    print("-" * 60)
    
    # Get TF-IDF scores for this pattern
    pattern_tfidf = tfidf_dense[idx]
    
    # Get indices of top 15 terms
    top_indices = pattern_tfidf.argsort()[-15:][::-1]
    
    print(f"{'Rank':<6} {'Term':<35} {'TF-IDF Score':<15}")
    print("-" * 60)
    for rank, term_idx in enumerate(top_indices, 1):
        if pattern_tfidf[term_idx] > 0:
            term = feature_names[term_idx]
            score = pattern_tfidf[term_idx]
            print(f"{rank:<6} {term:<35} {score:.6f}")

# ============================================================================
# 4. COSINE SIMILARITY MATRIX
# ============================================================================
print("\n\n" + "="*80)
print("4. PATTERN SIMILARITY ANALYSIS (COSINE SIMILARITY)")
print("="*80)

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Create DataFrame for better visualization
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=pattern_titles,
    columns=pattern_titles
)

print("\nSimilarity Matrix:")
print(similarity_df.to_string())

# Find most similar pattern pairs
print("\n\nMost Similar Pattern Pairs (excluding self-similarity):")
print("-" * 80)
pairs = []
for i in range(len(pattern_titles)):
    for j in range(i+1, len(pattern_titles)):
        pairs.append((
            pattern_titles[i],
            pattern_titles[j],
            similarity_matrix[i][j]
        ))

pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
print(f"{'Pattern 1':<40} {'Pattern 2':<40} {'Similarity':<10}")
print("-" * 80)
for p1, p2, sim in pairs_sorted:
    print(f"{p1:<40} {p2:<40} {sim:.6f}")

# ============================================================================
# 5. SHARED IMPORTANT TERMS BETWEEN PATTERNS
# ============================================================================
print("\n\n" + "="*80)
print("5. SHARED IMPORTANT TERMS ANALYSIS")
print("="*80)

# For each pair, find shared terms with high TF-IDF
threshold = 0.05  # Minimum TF-IDF score to consider

for p1, p2, sim in pairs_sorted[:3]:  # Top 3 similar pairs
    idx1 = pattern_titles.index(p1)
    idx2 = pattern_titles.index(p2)
    
    print(f"\n{p1} <-> {p2}")
    print(f"Similarity: {sim:.6f}")
    print("-" * 60)
    
    # Find shared important terms
    shared_terms = []
    for term_idx, term in enumerate(feature_names):
        score1 = tfidf_dense[idx1][term_idx]
        score2 = tfidf_dense[idx2][term_idx]
        
        if score1 > threshold and score2 > threshold:
            shared_terms.append((term, score1, score2, (score1 + score2) / 2))
    
    shared_terms_sorted = sorted(shared_terms, key=lambda x: x[3], reverse=True)
    
    if shared_terms_sorted:
        print(f"{'Term':<30} {'P1 Score':<12} {'P2 Score':<12} {'Avg Score':<12}")
        print("-" * 60)
        for term, s1, s2, avg in shared_terms_sorted[:10]:
            print(f"{term:<30} {s1:.6f}     {s2:.6f}     {avg:.6f}")
    else:
        print("No shared terms above threshold")

# ============================================================================
# 6. TERM DISTRIBUTION ANALYSIS
# ============================================================================
print("\n\n" + "="*80)
print("6. TERM DISTRIBUTION ACROSS PATTERNS")
print("="*80)

# Count how many patterns each term appears in
term_distribution = {}
for term_idx, term in enumerate(feature_names):
    count = sum(1 for doc_vec in tfidf_dense if doc_vec[term_idx] > 0)
    term_distribution[term] = count

# Terms appearing in all patterns (common concepts)
universal_terms = [term for term, count in term_distribution.items() if count == len(patterns)]
print(f"\nTerms appearing in ALL patterns ({len(universal_terms)}):")
print(universal_terms[:20] if len(universal_terms) > 20 else universal_terms)

# Terms unique to single patterns (distinctive concepts)
unique_terms = [term for term, count in term_distribution.items() if count == 1]
print(f"\nTerms unique to SINGLE patterns ({len(unique_terms)}):")
for idx, title in enumerate(pattern_titles):
    pattern_unique = [term for term_idx, term in enumerate(feature_names)
                      if tfidf_dense[idx][term_idx] > 0 and term_distribution[term] == 1]
    if pattern_unique:
        print(f"\n{title}:")
        print(f"  {pattern_unique[:10]}")

# ============================================================================
# 7. PATTERN RELATIONSHIPS & RECOMMENDATIONS
# ============================================================================
print("\n\n" + "="*80)
print("7. PATTERN USAGE RECOMMENDATIONS")
print("="*80)

for idx, title in enumerate(pattern_titles):
    print(f"\n{title}")
    print("-" * 60)
    
    # Find related patterns (sorted by similarity, excluding self)
    similarities = [(pattern_titles[i], similarity_matrix[idx][i]) 
                   for i in range(len(pattern_titles)) if i != idx]
    similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    print("Related patterns (use together):")
    for related_title, sim_score in similarities_sorted:
        if sim_score > 0.1:  # Threshold for "related"
            print(f"  • {related_title} (similarity: {sim_score:.4f})")
    
    # Complementary patterns (low similarity but might be useful)
    print("\nComplementary patterns (different focus):")
    for related_title, sim_score in similarities_sorted[-2:]:
        print(f"  • {related_title} (similarity: {sim_score:.4f})")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n\n" + "="*80)
print("8. GENERATING VISUALIZATIONS")
print("="*80)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Heatmap of similarity matrix
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(similarity_df, annot=True, fmt='.3f', cmap='YlOrRd', 
            square=True, ax=ax1, cbar_kws={'label': 'Cosine Similarity'})
ax1.set_title('Pattern Similarity Heatmap (TF-IDF)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 2. Term distribution histogram
ax2 = plt.subplot(2, 2, 2)
distribution_counts = Counter(term_distribution.values())
plt.bar(distribution_counts.keys(), distribution_counts.values(), color='steelblue')
ax2.set_xlabel('Number of Patterns Term Appears In', fontsize=12)
ax2.set_ylabel('Number of Terms', fontsize=12)
ax2.set_title('Term Distribution Across Patterns', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Top terms by average TF-IDF
ax3 = plt.subplot(2, 2, 3)
avg_tfidf = tfidf_dense.mean(axis=0)
top_20_indices = avg_tfidf.argsort()[-20:][::-1]
top_terms = [feature_names[i] for i in top_20_indices]
top_scores = [avg_tfidf[i] for i in top_20_indices]

plt.barh(range(len(top_terms)), top_scores, color='coral')
plt.yticks(range(len(top_terms)), top_terms)
ax3.set_xlabel('Average TF-IDF Score', fontsize=12)
ax3.set_title('Top 20 Terms by Average TF-IDF Score', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# 4. Pattern complexity (word count vs unique terms)
ax4 = plt.subplot(2, 2, 4)
word_counts = [len(text.split()) for text in pattern_texts]
unique_term_counts = [(tfidf_dense[i] > 0).sum() for i in range(len(patterns))]

plt.scatter(word_counts, unique_term_counts, s=200, alpha=0.6, color='green')
for i, title in enumerate(pattern_titles):
    plt.annotate(title, (word_counts[i], unique_term_counts[i]), 
                fontsize=8, ha='center')
ax4.set_xlabel('Total Word Count', fontsize=12)
ax4.set_ylabel('Number of Unique TF-IDF Terms', fontsize=12)
ax4.set_title('Pattern Complexity Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tfidf_pattern_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualizations saved as 'tfidf_pattern_analysis.png'")

# ============================================================================
# 9. IMPLEMENTATION ORDER RECOMMENDATION
# ============================================================================
print("\n\n" + "="*80)
print("9. SUGGESTED IMPLEMENTATION ORDER")
print("="*80)

print("""
Based on TF-IDF analysis and pattern dependencies:

TIER 1 - Foundation Patterns (Implement First):
  These patterns are referenced by others and provide basic security primitives.
  
TIER 2 - Building Block Patterns (Implement Second):
  These patterns build on foundation patterns and can be used independently.
  
TIER 3 - Complex Integration Patterns (Implement Last):
  These patterns combine multiple lower-tier patterns.
""")

# Simple heuristic: patterns with lower avg similarity might be more fundamental
avg_similarities = similarity_matrix.mean(axis=1)
independence_order = sorted(enumerate(pattern_titles), key=lambda x: avg_similarities[x[0]])

print("\nPattern Independence Ranking (lower similarity = more fundamental):")
print(f"{'Rank':<6} {'Pattern':<45} {'Avg Similarity':<15}")
print("-" * 70)
for rank, (idx, title) in enumerate(independence_order, 1):
    print(f"{rank:<6} {title:<45} {avg_similarities[idx]:.6f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)