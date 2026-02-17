"""
Word Processing Assignment - Part I: Preliminary Data Analysis
==============================================================
Preprocessing follows Futrell et al. (2021):
  - Exclude participant-story passes with < 5 correct comprehension answers
  - Exclude raw RTs < 100 ms or > 3000 ms
  - Then compute mean RT per word
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
RT_FILE = "processed_RTs.tsv"
FREQ_FILE = "freqs-1.tsv"
OUTPUT_DIR = "part1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
rt_df = pd.read_csv(RT_FILE, sep='\t')
print(f"Raw Reading Times: {rt_df.shape[0]} rows, {rt_df.shape[1]} columns")
print(f"Columns: {list(rt_df.columns)}\n")

freq_df = pd.read_csv(FREQ_FILE, sep='\t', header=None,
                       names=['token_id', 'ngram_order', 'token', 'frequency', 'context_freq'])
print(f"Frequency Data: {freq_df.shape[0]} rows\n")

# ============================================================
# Preprocessing (following Futrell et al. 2021)
# ============================================================
print("=" * 50)
print("Preprocessing (following Futrell et al. 2021)")
print("=" * 50)

n_raw = len(rt_df)

# Step 1: Exclude participant-story passes with < 5 correct answers
# 'correct' column = number of correct comprehension questions per pass
rt_df = rt_df[rt_df['correct'] >= 5].copy()
n_after_correct = len(rt_df)
print(f"  After excluding correct < 5: {n_after_correct} rows "
      f"(removed {n_raw - n_after_correct}, {100*(n_raw - n_after_correct)/n_raw:.1f}%)")

# Step 2: Exclude RTs < 100 ms or > 3000 ms
rt_df = rt_df[(rt_df['RT'] >= 100) & (rt_df['RT'] <= 3000)].copy()
n_after_rt = len(rt_df)
print(f"  After excluding RT < 100 or > 3000: {n_after_rt} rows "
      f"(removed {n_after_correct - n_after_rt}, {100*(n_after_correct - n_after_rt)/n_after_correct:.1f}%)")

# ============================================================
# Preprocess Frequency Data
# ============================================================
freq_df[['story_id', 'word_id', 'part_id']] = freq_df['token_id'].str.split('.', expand=True)
freq_df['story_id'] = freq_df['story_id'].astype(int)
freq_df['word_id'] = freq_df['word_id'].astype(int)

# Use "word" token type -> one frequency per word position
freq_word = freq_df[freq_df['part_id'] == 'word'][['story_id', 'word_id', 'token', 'frequency']].copy()
freq_word = freq_word.rename(columns={'token': 'freq_token', 'frequency': 'word_freq'})

# ============================================================
# Q1: Compute Mean RT per Word
# ============================================================
print("\n" + "=" * 50)
print("Q1: Mean RT per word (averaged across subjects)")
print("=" * 50)

mean_rt = rt_df.groupby(['item', 'zone']).agg(
    mean_RT=('RT', 'mean'),
    word=('word', 'first'),
    n_subjects=('RT', 'count')
).reset_index()

print(f"Unique word positions: {len(mean_rt)}")
print(f"Sample:\n{mean_rt.head(10)}\n")

# ============================================================
# Merge RT with Frequency Data
# ============================================================
merged = mean_rt.merge(freq_word, left_on=['item', 'zone'],
                       right_on=['story_id', 'word_id'], how='inner')

merged['word_length'] = merged['word'].astype(str).apply(len)
merged['log_freq'] = np.log10(merged['word_freq'].replace(0, 1))

print(f"Merged dataset: {len(merged)} word positions\n")

# Save for later parts
merged.to_csv(os.path.join(OUTPUT_DIR, "mean_rt_per_word.csv"), index=False)

# ============================================================
# Q2: Plot Word Length vs Mean RT
# ============================================================
print("=" * 50)
print("Q2: Word Length vs Mean RT")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(merged['word_length'], merged['mean_RT'],
                alpha=0.15, s=10, color='steelblue')
axes[0].set_xlabel('Word Length (characters)')
axes[0].set_ylabel('Mean RT (ms)')
axes[0].set_title('Word Length vs Mean RT (Scatter)')
axes[0].grid(True, alpha=0.3)

length_agg = merged.groupby('word_length').agg(
    avg_RT=('mean_RT', 'mean'),
    se_RT=('mean_RT', lambda x: x.std() / np.sqrt(len(x))),
    count=('mean_RT', 'count')
).reset_index()
length_agg = length_agg[length_agg['count'] >= 5]

axes[1].errorbar(length_agg['word_length'], length_agg['avg_RT'],
                 yerr=length_agg['se_RT'], fmt='o-', color='steelblue',
                 capsize=3, markersize=5)
axes[1].set_xlabel('Word Length (characters)')
axes[1].set_ylabel('Mean RT (ms)')
axes[1].set_title('Word Length vs Mean RT (Aggregated ± SE)')
axes[1].grid(True, alpha=0.3)

for _, row in length_agg.iterrows():
    axes[1].annotate(f'n={int(row["count"])}',
                     (row['word_length'], row['avg_RT']),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q2_word_length_vs_rt.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: q2_word_length_vs_rt.png\n")

# ============================================================
# Q3: Plot Word Frequency vs Mean RT
# ============================================================
print("=" * 50)
print("Q3: Word Frequency vs Mean RT")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(merged['log_freq'], merged['mean_RT'],
                alpha=0.15, s=10, color='darkorange')
axes[0].set_xlabel('Log10(Word Frequency)')
axes[0].set_ylabel('Mean RT (ms)')
axes[0].set_title('Word Frequency vs Mean RT (Scatter)')
axes[0].grid(True, alpha=0.3)

merged['freq_bin'] = pd.cut(merged['log_freq'], bins=15)
freq_agg = merged.groupby('freq_bin', observed=True).agg(
    avg_RT=('mean_RT', 'mean'),
    se_RT=('mean_RT', lambda x: x.std() / np.sqrt(len(x))),
    avg_freq=('log_freq', 'mean'),
    count=('mean_RT', 'count')
).reset_index()

axes[1].errorbar(freq_agg['avg_freq'], freq_agg['avg_RT'],
                 yerr=freq_agg['se_RT'], fmt='o-', color='darkorange',
                 capsize=3, markersize=5)
axes[1].set_xlabel('Log10(Word Frequency)')
axes[1].set_ylabel('Mean RT (ms)')
axes[1].set_title('Word Frequency vs Mean RT (Binned ± SE)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q3_word_freq_vs_rt.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: q3_word_freq_vs_rt.png\n")

# ============================================================
# Q4: Pearson Correlation - Word Length vs Frequency
# ============================================================
print("=" * 50)
print("Q4: Pearson Correlation - Word Length vs Log Frequency")
print("=" * 50)
r, p = stats.pearsonr(merged['word_length'], merged['log_freq'])
print(f"  r = {r:.4f}, p = {p:.2e}\n")

# ============================================================
# Q5: Pearson Correlation - Word Length vs Mean RT
# ============================================================
print("=" * 50)
print("Q5: Pearson Correlation - Word Length vs Mean RT")
print("=" * 50)
r, p = stats.pearsonr(merged['word_length'], merged['mean_RT'])
print(f"  r = {r:.4f}, p = {p:.2e}\n")

# ============================================================
# Q6: Pearson Correlation - Word Frequency vs Mean RT
# ============================================================
print("=" * 50)
print("Q6: Pearson Correlation - Log Frequency vs Mean RT")
print("=" * 50)
r, p = stats.pearsonr(merged['log_freq'], merged['mean_RT'])
print(f"  r = {r:.4f}, p = {p:.2e}\n")

r_raw, p_raw = stats.pearsonr(merged['word_freq'], merged['mean_RT'])
print(f"  (raw freq) r = {r_raw:.4f}, p = {p_raw:.2e}\n")

# ============================================================
# Correlation Summary Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

r1, p1 = stats.pearsonr(merged['word_length'], merged['log_freq'])
axes[0].scatter(merged['word_length'], merged['log_freq'], alpha=0.15, s=10, color='green')
axes[0].set_xlabel('Word Length')
axes[0].set_ylabel('Log10(Frequency)')
axes[0].set_title(f'Length vs Frequency\nr={r1:.4f}, p={p1:.2e}')
axes[0].grid(True, alpha=0.3)

r2, p2 = stats.pearsonr(merged['word_length'], merged['mean_RT'])
axes[1].scatter(merged['word_length'], merged['mean_RT'], alpha=0.15, s=10, color='steelblue')
axes[1].set_xlabel('Word Length')
axes[1].set_ylabel('Mean RT (ms)')
axes[1].set_title(f'Length vs Mean RT\nr={r2:.4f}, p={p2:.2e}')
axes[1].grid(True, alpha=0.3)

r3, p3 = stats.pearsonr(merged['log_freq'], merged['mean_RT'])
axes[2].scatter(merged['log_freq'], merged['mean_RT'], alpha=0.15, s=10, color='darkorange')
axes[2].set_xlabel('Log10(Frequency)')
axes[2].set_ylabel('Mean RT (ms)')
axes[2].set_title(f'Frequency vs Mean RT\nr={r3:.4f}, p={p3:.2e}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_summary.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_summary.png")

print("\nPart I complete! Outputs in:", OUTPUT_DIR)
