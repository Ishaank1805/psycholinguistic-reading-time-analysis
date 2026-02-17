"""
Word Processing Assignment - Part III: Frequency Ordered Bin Search (FOBS)
==========================================================================
1. Build a FOBS model using lemmatized forms and surface forms
2. Hypothesis 1: Root frequency predicts RT better than surface frequency
3. Hypothesis 2: Pseudo-affixed words take longer than regularly affixed words

Requirements: pip install nltk statsmodels
              python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import re
import os
import warnings
warnings.filterwarnings('ignore')

# NLTK imports for lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# ============================================================
# Configuration
# ============================================================
MERGED_FILE = "part2_outputs/merged_data_part2.csv"
FREQ_FILE = "freqs-1.tsv"
OUTPUT_DIR = "part3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
df = pd.read_csv(MERGED_FILE)
print(f"Loaded {len(df)} word positions from Part 2\n")

# Load unigram frequencies for lemma frequency lookup
freq_df = pd.read_csv(FREQ_FILE, sep='\t', header=None,
                       names=['token_id', 'ngram_order', 'token', 'frequency', 'context_freq'])
freq_df[['story_id', 'word_id', 'part_id']] = freq_df['token_id'].str.split('.', expand=True)
# Build a word -> frequency lookup from all tokens in the corpus
# Use the "word" rows for frequency
word_freq_lookup = freq_df[freq_df['part_id'] == 'word'].groupby(
    freq_df[freq_df['part_id'] == 'word']['token'].str.lower()
)['frequency'].max().to_dict()

print(f"Frequency lookup: {len(word_freq_lookup)} unique words\n")

# ============================================================
# Helper: Get POS tag for WordNet
# ============================================================
def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN,
                'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# ============================================================
# Lemmatization
# ============================================================
print("=" * 50)
print("Lemmatizing words")
print("=" * 50)

lemmatizer = WordNetLemmatizer()

def clean_word(w):
    """Strip punctuation from word for lemmatization."""
    return re.sub(r'[^a-zA-Z]', '', str(w)).lower()

def lemmatize_word(w):
    """Lemmatize a word using WordNet with POS tagging."""
    cleaned = clean_word(w)
    if not cleaned:
        return cleaned
    pos = get_wordnet_pos(cleaned)
    return lemmatizer.lemmatize(cleaned, pos)

# Apply lemmatization
df['word_clean'] = df['word'].apply(clean_word)
df['lemma'] = df['word'].apply(lemmatize_word)
df['lemma_length'] = df['lemma'].apply(len)

# Show some examples
print("\nLemmatization examples:")
examples = df[df['word_clean'] != df['lemma']][['word', 'word_clean', 'lemma']].drop_duplicates().head(20)
for _, row in examples.iterrows():
    print(f"  {row['word']:20s} -> {row['word_clean']:20s} -> {row['lemma']}")

# ============================================================
# Compute Lemma Frequencies
# ============================================================
print("\n" + "=" * 50)
print("Computing lemma frequencies")
print("=" * 50)

# For each lemma, look up its frequency from the corpus frequency data
# Use the maximum frequency across all surface forms that share the same lemma
# First, build lemma -> frequency from our frequency lookup
lemma_freq_map = {}
for surface, freq in word_freq_lookup.items():
    cleaned = re.sub(r'[^a-zA-Z]', '', surface).lower()
    if not cleaned:
        continue
    try:
        pos = get_wordnet_pos(cleaned)
        lem = lemmatizer.lemmatize(cleaned, pos)
    except:
        lem = cleaned
    if lem not in lemma_freq_map:
        lemma_freq_map[lem] = 0
    lemma_freq_map[lem] += freq  # Sum frequencies of all surface forms

# Map lemma frequency to each word in the dataframe
df['lemma_freq'] = df['lemma'].map(lemma_freq_map).fillna(0)
df['log_lemma_freq'] = np.log10(df['lemma_freq'].replace(0, 1))

# Summary
has_lemma_freq = (df['lemma_freq'] > 0).sum()
print(f"  Words with lemma frequency: {has_lemma_freq}/{len(df)}")
print(f"  Log lemma freq range: {df['log_lemma_freq'].min():.2f} - {df['log_lemma_freq'].max():.2f}")

# Filter out words with zero lemma frequency
df_fobs = df[df['lemma_freq'] > 0].copy()
df_fobs = df_fobs[df_fobs['lemma_length'] > 0].copy()
print(f"  Clean FOBS dataset: {len(df_fobs)} words\n")

# ============================================================
# FOBS Model Construction
# ============================================================
print("=" * 50)
print("Building FOBS Model")
print("=" * 50)

# The FOBS model (Frequency Ordered Bin Search) organizes the mental lexicon
# by frequency. Words are stored with their root forms, and lookup proceeds
# by searching frequency-ordered bins.
#
# We build the model by:
# 1. Getting unique lemmas and their frequencies
# 2. Sorting by frequency (most frequent first)
# 3. Assigning bin positions (rank)
# 4. Each surface form inherits its lemma's bin position

# Get unique lemmas sorted by frequency
lemma_table = df_fobs.groupby('lemma').agg(
    lemma_freq=('lemma_freq', 'first'),
    n_surface_forms=('word_clean', 'nunique'),
    surface_forms=('word_clean', lambda x: ', '.join(sorted(x.unique())[:5]))
).reset_index()
lemma_table = lemma_table.sort_values('lemma_freq', ascending=False).reset_index(drop=True)
lemma_table['fobs_rank'] = lemma_table.index + 1  # 1-indexed rank
lemma_table['fobs_bin'] = pd.qcut(lemma_table['fobs_rank'], q=10, labels=False) + 1

print(f"  Unique lemmas: {len(lemma_table)}")
print(f"  Top 10 lemmas by frequency:")
for _, row in lemma_table.head(10).iterrows():
    print(f"    Rank {row['fobs_rank']:4.0f}: {row['lemma']:15s} "
          f"(freq={row['lemma_freq']:.0f}, forms: {row['surface_forms']})")

# Map FOBS rank and bin back to main dataframe
rank_map = lemma_table.set_index('lemma')['fobs_rank'].to_dict()
bin_map = lemma_table.set_index('lemma')['fobs_bin'].to_dict()
df_fobs['fobs_rank'] = df_fobs['lemma'].map(rank_map)
df_fobs['fobs_bin'] = df_fobs['lemma'].map(bin_map)

# FOBS prediction: words in lower-rank (higher frequency) bins should be read faster
fobs_bin_rt = df_fobs.groupby('fobs_bin').agg(
    mean_RT=('mean_RT', 'mean'),
    se_RT=('mean_RT', lambda x: x.std() / np.sqrt(len(x))),
    count=('mean_RT', 'count'),
    mean_lemma_freq=('log_lemma_freq', 'mean')
).reset_index()

print(f"\n  FOBS Bin Summary (mean RT by frequency bin):")
print(f"  {'Bin':>5} {'Mean RT':>10} {'SE':>8} {'N':>8} {'Mean Log Freq':>15}")
for _, row in fobs_bin_rt.iterrows():
    print(f"  {row['fobs_bin']:>5.0f} {row['mean_RT']:>10.2f} {row['se_RT']:>8.2f} "
          f"{row['count']:>8.0f} {row['mean_lemma_freq']:>15.2f}")

# FOBS visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].errorbar(fobs_bin_rt['fobs_bin'], fobs_bin_rt['mean_RT'],
                 yerr=fobs_bin_rt['se_RT'], fmt='o-', color='purple',
                 capsize=3, markersize=6)
axes[0].set_xlabel('FOBS Bin (1=most frequent)')
axes[0].set_ylabel('Mean RT (ms)')
axes[0].set_title('FOBS Model: Mean RT by Frequency Bin')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df_fobs['fobs_rank'], df_fobs['mean_RT'], alpha=0.1, s=10, color='purple')
axes[1].set_xlabel('FOBS Rank (1=most frequent)')
axes[1].set_ylabel('Mean RT (ms)')
axes[1].set_title('FOBS Model: RT vs Frequency Rank')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fobs_model.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fobs_model.png")

# Save FOBS table
lemma_table.to_csv(os.path.join(OUTPUT_DIR, "fobs_lemma_table.csv"), index=False)

# ============================================================
# HYPOTHESIS 1: Root Frequency vs Surface Frequency
# ============================================================
print("\n" + "=" * 60)
print("HYPOTHESIS 1: Root frequency vs surface frequency")
print("=" * 60)

def run_regression(data, formula, model_name):
    """Fit OLS model and print summary stats."""
    model = ols(formula, data=data).fit()
    print(f"\n--- {model_name} ---")
    print(f"  Formula:    {formula}")
    print(f"  N:          {model.nobs:.0f}")
    print(f"  R²:         {model.rsquared:.4f}")
    print(f"  Adj R²:     {model.rsquared_adj:.4f}")
    print(f"  AIC:        {model.aic:.1f}")
    print(f"  BIC:        {model.bic:.1f}")
    print(f"  F-stat:     {model.fvalue:.2f} (p={model.f_pvalue:.2e})")
    print(f"  Coefficients:")
    for var in model.params.index:
        coef = model.params[var]
        pval = model.pvalues[var]
        print(f"    {var:25s}: B={coef:10.4f}, p={pval:.2e}")
    return model

model_h1_1 = run_regression(df_fobs, 'mean_RT ~ log_freq + word_length',
                             'Model 1: RT ~ Surface Freq + Word Length')
model_h1_2 = run_regression(df_fobs, 'mean_RT ~ log_lemma_freq + lemma_length',
                             'Model 2: RT ~ Lemma Freq + Lemma Length')

print("\n--- Hypothesis 1: Model Comparison ---")
print(f"  {'Metric':<15} {'M1 (Surface)':>18} {'M2 (Lemma)':>18}")
print(f"  {'R²':<15} {model_h1_1.rsquared:>18.4f} {model_h1_2.rsquared:>18.4f}")
print(f"  {'Adj R²':<15} {model_h1_1.rsquared_adj:>18.4f} {model_h1_2.rsquared_adj:>18.4f}")
print(f"  {'AIC':<15} {model_h1_1.aic:>18.1f} {model_h1_2.aic:>18.1f}")
print(f"  {'BIC':<15} {model_h1_1.bic:>18.1f} {model_h1_2.bic:>18.1f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df_fobs['log_freq'], df_fobs['mean_RT'], alpha=0.15, s=10, color='steelblue')
z = np.polyfit(df_fobs['log_freq'], df_fobs['mean_RT'], 1)
x_r = np.linspace(df_fobs['log_freq'].min(), df_fobs['log_freq'].max(), 100)
axes[0].plot(x_r, np.poly1d(z)(x_r), 'r-', linewidth=2)
axes[0].set_xlabel('Log10(Surface Frequency)')
axes[0].set_ylabel('Mean RT (ms)')
axes[0].set_title(f'Surface Freq vs RT (R²={model_h1_1.rsquared:.4f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df_fobs['log_lemma_freq'], df_fobs['mean_RT'], alpha=0.15, s=10, color='green')
z = np.polyfit(df_fobs['log_lemma_freq'], df_fobs['mean_RT'], 1)
x_r = np.linspace(df_fobs['log_lemma_freq'].min(), df_fobs['log_lemma_freq'].max(), 100)
axes[1].plot(x_r, np.poly1d(z)(x_r), 'r-', linewidth=2)
axes[1].set_xlabel('Log10(Lemma Frequency)')
axes[1].set_ylabel('Mean RT (ms)')
axes[1].set_title(f'Lemma Freq vs RT (R²={model_h1_2.rsquared:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h1_surface_vs_lemma.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: h1_surface_vs_lemma.png")

# R² bar chart
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Model 1\n(Surface Freq + Length)', 'Model 2\n(Lemma Freq + Lemma Length)']
r2s = [model_h1_1.rsquared, model_h1_2.rsquared]
colors = ['steelblue', 'green']
bars = ax.bar(labels, r2s, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'R²={val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('R²')
ax.set_title('Hypothesis 1: Surface vs Lemma Frequency')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h1_r2_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h1_r2_comparison.png")

# Residual comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, model, title, color in [
    (axes[0], model_h1_1, 'M1: Surface Freq + Length', 'steelblue'),
    (axes[1], model_h1_2, 'M2: Lemma Freq + Lemma Length', 'green'),
]:
    ax.scatter(model.fittedvalues, model.resid, alpha=0.15, s=10, color=color)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{title}\nR²={model.rsquared:.4f}, AIC={model.aic:.0f}')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h1_residuals.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h1_residuals.png")

# ============================================================
# HYPOTHESIS 2: Pseudo-Affixed vs Regularly Affixed Words
# ============================================================
print("\n" + "=" * 60)
print("HYPOTHESIS 2: Pseudo-affixed vs regularly affixed words")
print("=" * 60)

# The FOBS model predicts that pseudo-affixed words (e.g., "finger" which
# looks like "fing" + "-er" but isn't) take longer to process because:
# - The parser attempts morphological decomposition
# - Fails to find the pseudo-root ("fing"), causing extra lookup time
# - Regularly affixed words (e.g., "driver" = "drive" + "-er") decompose
#   successfully and benefit from root frequency

# Select 5 pairs of similar-length, similar-frequency words:
# pseudo-affixed (root doesn't exist) vs regularly affixed (root exists)

# We'll search for these words in our dataset
# Select 5 matched pairs: all length 6, similar frequency range (~6.5-7.5)
# Pseudo-affixed: surface form contains "-er" but root is not a real
# related word (morphological decomposition fails)
pseudo_affixed_candidates = {
    'finger': ('fing', '-er', 'no root "fing"'),
    'corner': ('corn', '-er', '"corn" is unrelated'),
    'master': ('mast', '-er', '"mast" is unrelated'),
    'silver': ('silv', '-er', 'no root "silv"'),
    'sister': ('sist', '-er', 'no root "sist"'),
}

# Regular-affixed: surface form contains "-er" and root IS a real word
# (morphological decomposition succeeds)
regular_affixed_candidates = {
    'closer': ('close', '-er', 'comparative of close'),
    'higher': ('high', '-er', 'comparative of high'),
    'larger': ('large', '-er', 'comparative of large'),
    'longer': ('long', '-er', 'comparative of long'),
    'lesser': ('less', '-er', 'comparative of less'),
}

# Find which of these words appear in our dataset
print("\nSearching for pseudo-affixed and regularly affixed words in corpus...\n")

def find_word_in_data(word, data):
    """Find a word in the dataset (case-insensitive, strip punctuation)."""
    matches = data[data['word_clean'] == word.lower()]
    if len(matches) == 0:
        return None
    return {
        'word': word,
        'mean_RT': matches['mean_RT'].mean(),
        'word_freq': matches['word_freq'].mean(),
        'log_freq': matches['log_freq'].mean(),
        'word_length': len(word),
        'n_occurrences': len(matches),
    }

pseudo_found = []
for word, (root, affix, explanation) in pseudo_affixed_candidates.items():
    result = find_word_in_data(word, df_fobs)
    if result:
        result['type'] = 'Pseudo-affixed'
        result['root'] = root
        result['affix'] = affix
        result['explanation'] = explanation
        pseudo_found.append(result)

regular_found = []
for word, (root, affix, explanation) in regular_affixed_candidates.items():
    result = find_word_in_data(word, df_fobs)
    if result:
        result['type'] = 'Regular-affixed'
        result['root'] = root
        result['affix'] = affix
        result['explanation'] = explanation
        regular_found.append(result)

print(f"Pseudo-affixed found in corpus: {len(pseudo_found)}")
for w in pseudo_found:
    print(f"  {w['word']:12s} len={w['word_length']:2d} log_freq={w['log_freq']:.2f} "
          f"RT={w['mean_RT']:.1f}ms (n={w['n_occurrences']}) [{w['explanation']}]")

print(f"\nRegular-affixed found in corpus: {len(regular_found)}")
for w in regular_found:
    print(f"  {w['word']:12s} len={w['word_length']:2d} log_freq={w['log_freq']:.2f} "
          f"RT={w['mean_RT']:.1f}ms (n={w['n_occurrences']}) [{w['explanation']}]")

# Select 5 best-matched pairs by length and frequency
# Combine and select matched words
pseudo_df = pd.DataFrame(pseudo_found)
regular_df = pd.DataFrame(regular_found)

# Select top 5 from each based on availability
n_select = min(5, len(pseudo_df), len(regular_df))

if n_select > 0:
    pseudo_select = pseudo_df.head(n_select)
    regular_select = regular_df.head(n_select)

    # Combine for analysis
    h2_data = pd.concat([pseudo_select, regular_select], ignore_index=True)

    print(f"\n--- Selected Words for Comparison (n={n_select} each) ---")
    print(f"\n  {'Word':<12} {'Type':<18} {'Length':>6} {'Log Freq':>10} {'Mean RT':>10}")
    print("  " + "-" * 60)
    for _, row in h2_data.iterrows():
        print(f"  {row['word']:<12} {row['type']:<18} {row['word_length']:>6} "
              f"{row['log_freq']:>10.2f} {row['mean_RT']:>10.1f}")

    # Descriptive comparison
    print(f"\n--- Group Means ---")
    for grp in ['Pseudo-affixed', 'Regular-affixed']:
        subset = h2_data[h2_data['type'] == grp]
        print(f"  {grp}:")
        print(f"    Mean RT:     {subset['mean_RT'].mean():.2f} ms")
        print(f"    Mean Length:  {subset['word_length'].mean():.2f}")
        print(f"    Mean Log Freq: {subset['log_freq'].mean():.2f}")

    # T-test between pseudo and regular affixed RTs
    pseudo_rts = h2_data[h2_data['type'] == 'Pseudo-affixed']['mean_RT']
    regular_rts = h2_data[h2_data['type'] == 'Regular-affixed']['mean_RT']

    t_stat, p_val = stats.ttest_ind(pseudo_rts, regular_rts)
    print(f"\n--- Independent t-test ---")
    print(f"  t = {t_stat:.4f}, p = {p_val:.4f}")

    # Also do Mann-Whitney U test (non-parametric, good for small N)
    u_stat, u_pval = stats.mannwhitneyu(pseudo_rts, regular_rts, alternative='greater')
    print(f"\n--- Mann-Whitney U test (pseudo > regular) ---")
    print(f"  U = {u_stat:.1f}, p = {u_pval:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of individual words
    colors_h2 = ['firebrick' if t == 'Pseudo-affixed' else 'forestgreen'
                  for t in h2_data['type']]
    bars = axes[0].bar(range(len(h2_data)), h2_data['mean_RT'], color=colors_h2,
                        edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(h2_data)))
    axes[0].set_xticklabels(h2_data['word'], rotation=45, ha='right')
    axes[0].set_ylabel('Mean RT (ms)')
    axes[0].set_title('Reading Times: Pseudo vs Regular Affixed Words')
    axes[0].grid(True, alpha=0.3, axis='y')
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='firebrick', label='Pseudo-affixed'),
                       Patch(facecolor='forestgreen', label='Regular-affixed')]
    axes[0].legend(handles=legend_elements)

    # Group comparison
    group_means = h2_data.groupby('type')['mean_RT'].agg(['mean', 'std', 'count'])
    group_means['se'] = group_means['std'] / np.sqrt(group_means['count'])
    grp_colors = {'Pseudo-affixed': 'firebrick', 'Regular-affixed': 'forestgreen'}
    x_pos = [0, 1]
    for i, (grp, row) in enumerate(group_means.iterrows()):
        axes[1].bar(x_pos[i], row['mean'], yerr=row['se'],
                     color=grp_colors[grp], edgecolor='black',
                     capsize=5, linewidth=0.5)
        axes[1].text(x_pos[i], row['mean'] + row['se'] + 5,
                      f'{row["mean"]:.1f}', ha='center', fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Pseudo-affixed', 'Regular-affixed'])
    axes[1].set_ylabel('Mean RT (ms)')
    axes[1].set_title(f'Group Comparison\nt={t_stat:.2f}, p={p_val:.4f}')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "h2_affix_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: h2_affix_comparison.png")

    # Also look at this in the broader dataset:
    # Tag all words ending in common suffixes as potentially affixed
    print("\n--- Broader Corpus Analysis: -er suffix words ---")
    er_words = df_fobs[df_fobs['word_clean'].str.endswith('er') &
                        (df_fobs['word_length'] >= 4)].copy()

    # Check if stripping -er gives a real word (exists in frequency lookup)
    def has_real_root(word):
        stem = word[:-2]  # strip -er
        if stem in word_freq_lookup:
            return True
        # Also check adding 'e' back (e.g., driver -> drive)
        if (stem + 'e') in word_freq_lookup:
            return True
        return False

    er_words['has_real_root'] = er_words['word_clean'].apply(has_real_root)
    er_words['affix_type'] = er_words['has_real_root'].map(
        {True: 'Transparent (-er)', False: 'Opaque (-er)'})

    er_summary = er_words.groupby('affix_type').agg(
        mean_RT=('mean_RT', 'mean'),
        se_RT=('mean_RT', lambda x: x.std() / np.sqrt(len(x))),
        mean_freq=('log_freq', 'mean'),
        mean_length=('word_length', 'mean'),
        count=('mean_RT', 'count')
    )
    print(er_summary.to_string())

    if len(er_words[er_words['has_real_root']]) > 0 and len(er_words[~er_words['has_real_root']]) > 0:
        t2, p2 = stats.ttest_ind(
            er_words[~er_words['has_real_root']]['mean_RT'],
            er_words[er_words['has_real_root']]['mean_RT']
        )
        print(f"\n  t-test (Opaque vs Transparent -er words): t={t2:.4f}, p={p2:.4f}")

else:
    print("\nInsufficient words found in corpus for Hypothesis 2 analysis.")
    print("You may need to supplement with external frequency data.")

# ============================================================
# Save data
# ============================================================
df_fobs.to_csv(os.path.join(OUTPUT_DIR, "fobs_data.csv"), index=False)
print(f"\nFOBS data saved to {OUTPUT_DIR}/fobs_data.csv")

print("\nPart III complete! Outputs in:", OUTPUT_DIR)