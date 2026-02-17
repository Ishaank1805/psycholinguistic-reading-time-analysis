"""
Word Processing Assignment - Part II: Hypothesis Testing
=========================================================
Preprocessing follows Futrell et al. (2021):
  - Uses mean RT per word computed in Part 1 (already filtered)
  - GPT-3 surprisal from all_stories_gpt3.csv (davinci logprobs)
  - Surprisal = -logprob / ln(2)  (convert natural log to bits)

Hypothesis 1: LM probabilities are better predictors of RT than word frequency
Hypothesis 2: Content words are processed differently than function words
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
MEAN_RT_FILE = "part1_outputs/mean_rt_per_word.csv"
GPT3_FILE = "all_stories_gpt3.csv"
OUTPUT_DIR = "part2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
df = pd.read_csv(MEAN_RT_FILE)
gpt = pd.read_csv(GPT3_FILE)
print(f"Mean RT data: {len(df)} word positions")
print(f"GPT-3 data: {len(gpt)} tokens across {gpt['story'].nunique()} stories\n")

# ============================================================
# Align GPT-3 Tokens to RT Words
# ============================================================
# GPT-3 (davinci) tokenizes punctuation separately (e.g., "England" + ",")
# but RT has them merged (e.g., "England,").
# Strategy: walk through both token lists per story, accumulating GPT tokens
# until they reconstruct the RT word, then sum logprobs.

print("=" * 50)
print("Aligning GPT-3 tokens to RT words")
print("=" * 50)

def align_story(gpt_tokens, gpt_logprobs, rt_words):
    """
    Align GPT-3 tokens to RT words for a single story.
    Returns a list of (word_index, summed_logprob) pairs.
    """
    results = []
    g_idx = 0

    for w_idx, rt_word in enumerate(rt_words):
        accumulated = ""
        summed_logprob = 0.0
        has_valid_logprob = False

        while g_idx < len(gpt_tokens):
            token = gpt_tokens[g_idx].strip()
            logprob = gpt_logprobs[g_idx]

            accumulated += token
            if not np.isnan(logprob):
                summed_logprob += logprob
                has_valid_logprob = True
            g_idx += 1

            if accumulated == rt_word:
                if has_valid_logprob:
                    results.append((w_idx, summed_logprob))
                else:
                    results.append((w_idx, np.nan))
                break
        else:
            if accumulated:
                results.append((w_idx, np.nan))

    return results

aligned_records = []
total_matched = 0
total_words = 0

for story_gpt in sorted(gpt['story'].unique()):
    story_rt = story_gpt + 1  # GPT is 0-indexed, RT is 1-indexed

    gpt_story = gpt[gpt['story'] == story_gpt].reset_index(drop=True)
    gpt_tokens = gpt_story['token'].tolist()
    gpt_logprobs = gpt_story['logprob'].tolist()

    rt_story = df[df['item'] == story_rt].sort_values('zone').reset_index(drop=True)
    rt_words = rt_story['word'].astype(str).tolist()
    rt_zones = rt_story['zone'].tolist()

    total_words += len(rt_words)

    alignments = align_story(gpt_tokens, gpt_logprobs, rt_words)

    for w_idx, logprob in alignments:
        if w_idx < len(rt_zones):
            aligned_records.append({
                'item': story_rt,
                'zone': rt_zones[w_idx],
                'gpt3_logprob': logprob
            })
            if not np.isnan(logprob):
                total_matched += 1

    print(f"  Story {story_gpt}/{story_rt}: {len(alignments)}/{len(rt_words)} words aligned")

align_df = pd.DataFrame(aligned_records)
print(f"\nTotal aligned: {total_matched}/{total_words} words with valid logprobs")

# Merge GPT-3 logprobs into main dataframe
df = df.merge(align_df, on=['item', 'zone'], how='inner')

# Compute GPT-3 surprisal: -log2(P) = -logprob / ln(2)
# logprob from OpenAI is in natural log; convert to bits
df['gpt3_surprisal'] = -df['gpt3_logprob'] / np.log(2)

# Drop rows with NaN surprisal (first word of each story has no logprob)
df = df.dropna(subset=['gpt3_surprisal'])
df = df[df['gpt3_surprisal'] > 0].copy()
print(f"Final dataset with GPT-3 surprisal: {len(df)} word positions\n")

print(f"GPT-3 Surprisal stats:")
print(f"  Mean: {df['gpt3_surprisal'].mean():.2f} bits")
print(f"  Std:  {df['gpt3_surprisal'].std():.2f}")
print(f"  Min:  {df['gpt3_surprisal'].min():.2f}")
print(f"  Max:  {df['gpt3_surprisal'].max():.2f}")

# ============================================================
# Content vs Function Word Classification
# ============================================================
print("\n" + "=" * 50)
print("Classifying Content vs Function Words")
print("=" * 50)

FUNCTION_WORDS = set([
    # Determiners / Articles
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your',
    'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every',
    'each', 'all', 'both', 'few', 'more', 'most', 'other', 'another',
    'such', 'what', 'which', 'whose',
    # Pronouns
    'i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'it', 'they',
    'them', 'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'themselves', 'who', 'whom', 'whoever', 'whomever',
    'anyone', 'everyone', 'someone', 'nobody', 'everybody', 'somebody',
    'anything', 'everything', 'something', 'nothing', 'one', 'ones',
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about',
    'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'over', 'of', 'up', 'down', 'out',
    'off', 'against', 'along', 'around', 'among', 'without', 'within',
    'upon', 'toward', 'towards', 'across', 'behind', 'beyond', 'near',
    'until', 'since', 'beside', 'besides', 'except', 'throughout',
    # Conjunctions
    'and', 'but', 'or', 'nor', 'so', 'yet', 'for', 'because',
    'although', 'though', 'while', 'if', 'unless', 'than', 'whether',
    'either', 'neither', 'when', 'where', 'how', 'that',
    # Auxiliary / Modal Verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can',
    'could', 'must', 'need', 'dare', 'ought',
    # Other function words
    'not', "n't", 'very', 'too', 'also', 'just', 'only', 'even',
    'still', 'already', 'ever', 'never', 'always', 'often',
    'here', 'there', 'then', 'now', 'well', 'quite', 'rather',
])

df['word_lower'] = df['word'].astype(str).str.lower().str.strip()
df['is_function'] = df['word_lower'].isin(FUNCTION_WORDS)
df['word_type'] = df['is_function'].map({True: 'Function', False: 'Content'})

n_func = df['is_function'].sum()
n_cont = (~df['is_function']).sum()
print(f"  Function words: {n_func} ({100*n_func/len(df):.1f}%)")
print(f"  Content words:  {n_cont} ({100*n_cont/len(df):.1f}%)")

# ============================================================
# Helper: Run and Report OLS Regression
# ============================================================
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

# ============================================================
# HYPOTHESIS 1: GPT-3 Surprisal vs Word Frequency
# ============================================================
print("\n" + "=" * 60)
print("HYPOTHESIS 1: GPT-3 probabilities vs word frequency")
print("=" * 60)

model1 = run_regression(df, 'mean_RT ~ log_freq + word_length',
                        'Model 1 (Frequency + Length)')
model2 = run_regression(df, 'mean_RT ~ gpt3_surprisal + word_length',
                        'Model 2 (GPT-3 Surprisal + Length)')

print("\n--- Model Comparison (Hypothesis 1) ---")
print(f"  {'Metric':<15} {'Model 1 (Freq)':>18} {'Model 2 (GPT-3)':>18}")
print(f"  {'R²':<15} {model1.rsquared:>18.4f} {model2.rsquared:>18.4f}")
print(f"  {'Adj R²':<15} {model1.rsquared_adj:>18.4f} {model2.rsquared_adj:>18.4f}")
print(f"  {'AIC':<15} {model1.aic:>18.1f} {model2.aic:>18.1f}")
print(f"  {'BIC':<15} {model1.bic:>18.1f} {model2.bic:>18.1f}")

# ============================================================
# Hypothesis 1: Visualizations
# ============================================================

# Residual & Actual vs Predicted plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(model1.fittedvalues, model1.resid, alpha=0.15, s=10, color='steelblue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title(f'Model 1 Residuals (Freq + Length)\nR²={model1.rsquared:.4f}')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(model2.fittedvalues, model2.resid, alpha=0.15, s=10, color='darkorange')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Fitted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title(f'Model 2 Residuals (GPT-3 Surprisal + Length)\nR²={model2.rsquared:.4f}')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(df['mean_RT'], model1.fittedvalues, alpha=0.15, s=10, color='steelblue')
axes[1, 0].plot([df['mean_RT'].min(), df['mean_RT'].max()],
                [df['mean_RT'].min(), df['mean_RT'].max()], 'r--', linewidth=1)
axes[1, 0].set_xlabel('Actual Mean RT')
axes[1, 0].set_ylabel('Predicted Mean RT')
axes[1, 0].set_title('Model 1: Actual vs Predicted')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(df['mean_RT'], model2.fittedvalues, alpha=0.15, s=10, color='darkorange')
axes[1, 1].plot([df['mean_RT'].min(), df['mean_RT'].max()],
                [df['mean_RT'].min(), df['mean_RT'].max()], 'r--', linewidth=1)
axes[1, 1].set_xlabel('Actual Mean RT')
axes[1, 1].set_ylabel('Predicted Mean RT')
axes[1, 1].set_title('Model 2: Actual vs Predicted')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Hypothesis 1: Model Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h1_model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: h1_model_comparison.png")

# R² bar chart
fig, ax = plt.subplots(figsize=(8, 5))
models_h1 = ['Model 1\n(Freq + Length)', 'Model 2\n(GPT-3 Surprisal + Length)']
r2_vals = [model1.rsquared, model2.rsquared]
colors = ['steelblue', 'darkorange']

bars = ax.bar(models_h1, r2_vals, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'R²={val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('R²')
ax.set_title('Hypothesis 1: Model Fit Comparison (R²)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h1_r2_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h1_r2_comparison.png")

# Predictor scatter plots with regression lines
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['log_freq'], df['mean_RT'], alpha=0.15, s=10, color='steelblue')
z = np.polyfit(df['log_freq'], df['mean_RT'], 1)
x_range = np.linspace(df['log_freq'].min(), df['log_freq'].max(), 100)
axes[0].plot(x_range, np.poly1d(z)(x_range), 'r-', linewidth=2)
axes[0].set_xlabel('Log10(Word Frequency)')
axes[0].set_ylabel('Mean RT (ms)')
axes[0].set_title('Word Frequency vs Mean RT')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df['gpt3_surprisal'], df['mean_RT'], alpha=0.15, s=10, color='darkorange')
z = np.polyfit(df['gpt3_surprisal'], df['mean_RT'], 1)
x_range = np.linspace(df['gpt3_surprisal'].min(), df['gpt3_surprisal'].quantile(0.99), 100)
axes[1].plot(x_range, np.poly1d(z)(x_range), 'r-', linewidth=2)
axes[1].set_xlabel('GPT-3 Surprisal (bits)')
axes[1].set_ylabel('Mean RT (ms)')
axes[1].set_title('GPT-3 Surprisal vs Mean RT')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h1_predictors_vs_rt.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h1_predictors_vs_rt.png")

# ============================================================
# HYPOTHESIS 2: Content vs Function Words
# ============================================================
print("\n" + "=" * 60)
print("HYPOTHESIS 2: Content vs Function Words")
print("=" * 60)

df_content = df[df['word_type'] == 'Content'].copy()
df_function = df[df['word_type'] == 'Function'].copy()

print(f"\nContent words: {len(df_content)}")
print(f"Function words: {len(df_function)}")

# Descriptive stats
print(f"\n--- Descriptive Statistics ---")
for label, subset in [('Content', df_content), ('Function', df_function)]:
    print(f"\n  {label} words:")
    print(f"    Mean RT:         {subset['mean_RT'].mean():.2f} +/- {subset['mean_RT'].std():.2f} ms")
    print(f"    Mean length:     {subset['word_length'].mean():.2f} chars")
    print(f"    Mean log freq:   {subset['log_freq'].mean():.2f}")
    print(f"    Mean surprisal:  {subset['gpt3_surprisal'].mean():.2f} bits")

# Run all 4 models
print("\n--- Content Word Models ---")
model_c1 = run_regression(df_content, 'mean_RT ~ log_freq + word_length',
                          'Model 1: Content ~ Freq + Length')
model_c2 = run_regression(df_content, 'mean_RT ~ gpt3_surprisal + word_length',
                          'Model 2: Content ~ GPT-3 Surprisal + Length')

print("\n--- Function Word Models ---")
model_f1 = run_regression(df_function, 'mean_RT ~ log_freq + word_length',
                          'Model 3: Function ~ Freq + Length')
model_f2 = run_regression(df_function, 'mean_RT ~ gpt3_surprisal + word_length',
                          'Model 4: Function ~ GPT-3 Surprisal + Length')

# Comparison table
print("\n\n--- Hypothesis 2: Full Model Comparison ---")
header = f"  {'Model':<45} {'R²':>8} {'Adj R²':>8} {'AIC':>12} {'BIC':>12}"
print(header)
print("  " + "-" * 85)
for name, m in [('M1: Content ~ Freq + Length', model_c1),
                ('M2: Content ~ GPT-3 Surprisal + Length', model_c2),
                ('M3: Function ~ Freq + Length', model_f1),
                ('M4: Function ~ GPT-3 Surprisal + Length', model_f2)]:
    print(f"  {name:<45} {m.rsquared:>8.4f} {m.rsquared_adj:>8.4f} {m.aic:>12.1f} {m.bic:>12.1f}")

# ============================================================
# Hypothesis 2: Visualizations
# ============================================================

# R² grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2)
width = 0.3

r2_content = [model_c1.rsquared, model_c2.rsquared]
r2_function = [model_f1.rsquared, model_f2.rsquared]

bars1 = ax.bar(x - width/2, r2_content, width, label='Content Words',
               color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, r2_function, width, label='Function Words',
               color='darkorange', edgecolor='black', linewidth=0.5)

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('R²')
ax.set_title('Hypothesis 2: Model Fit by Word Type')
ax.set_xticks(x)
ax.set_xticklabels(['Freq + Length', 'GPT-3 Surprisal + Length'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h2_r2_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: h2_r2_comparison.png")

# Distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df_content['mean_RT'], bins=50, alpha=0.6, color='steelblue',
             label='Content', density=True)
axes[0].hist(df_function['mean_RT'], bins=50, alpha=0.6, color='darkorange',
             label='Function', density=True)
axes[0].set_xlabel('Mean RT (ms)')
axes[0].set_ylabel('Density')
axes[0].set_title('RT Distribution by Word Type')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(df_content['log_freq'], bins=50, alpha=0.6, color='steelblue',
             label='Content', density=True)
axes[1].hist(df_function['log_freq'], bins=50, alpha=0.6, color='darkorange',
             label='Function', density=True)
axes[1].set_xlabel('Log10(Frequency)')
axes[1].set_ylabel('Density')
axes[1].set_title('Frequency Distribution by Word Type')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(df_content['gpt3_surprisal'], bins=50, alpha=0.6, color='steelblue',
             label='Content', density=True)
axes[2].hist(df_function['gpt3_surprisal'], bins=50, alpha=0.6, color='darkorange',
             label='Function', density=True)
axes[2].set_xlabel('GPT-3 Surprisal (bits)')
axes[2].set_ylabel('Density')
axes[2].set_title('Surprisal Distribution by Word Type')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h2_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h2_distributions.png")

# Scatter by word type
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(df_content['log_freq'], df_content['mean_RT'],
                    alpha=0.15, s=10, color='steelblue')
axes[0, 0].set_xlabel('Log10(Frequency)')
axes[0, 0].set_ylabel('Mean RT (ms)')
axes[0, 0].set_title(f'Content: Freq vs RT (R²={model_c1.rsquared:.4f})')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(df_content['gpt3_surprisal'], df_content['mean_RT'],
                    alpha=0.15, s=10, color='steelblue')
axes[0, 1].set_xlabel('GPT-3 Surprisal (bits)')
axes[0, 1].set_ylabel('Mean RT (ms)')
axes[0, 1].set_title(f'Content: Surprisal vs RT (R²={model_c2.rsquared:.4f})')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(df_function['log_freq'], df_function['mean_RT'],
                    alpha=0.15, s=10, color='darkorange')
axes[1, 0].set_xlabel('Log10(Frequency)')
axes[1, 0].set_ylabel('Mean RT (ms)')
axes[1, 0].set_title(f'Function: Freq vs RT (R²={model_f1.rsquared:.4f})')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(df_function['gpt3_surprisal'], df_function['mean_RT'],
                    alpha=0.15, s=10, color='darkorange')
axes[1, 1].set_xlabel('GPT-3 Surprisal (bits)')
axes[1, 1].set_ylabel('Mean RT (ms)')
axes[1, 1].set_title(f'Function: Surprisal vs RT (R²={model_f2.rsquared:.4f})')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Hypothesis 2: Predictors vs RT by Word Type',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h2_scatter_by_type.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h2_scatter_by_type.png")

# Residuals for all 4 models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, model, title, color in [
    (axes[0, 0], model_c1, 'M1: Content ~ Freq+Len', 'steelblue'),
    (axes[0, 1], model_c2, 'M2: Content ~ Surp+Len', 'teal'),
    (axes[1, 0], model_f1, 'M3: Function ~ Freq+Len', 'darkorange'),
    (axes[1, 1], model_f2, 'M4: Function ~ Surp+Len', 'firebrick'),
]:
    ax.scatter(model.fittedvalues, model.resid, alpha=0.15, s=10, color=color)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{title}\nR²={model.rsquared:.4f}, AIC={model.aic:.0f}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Hypothesis 2: Residual Plots', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "h2_residuals.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: h2_residuals.png")

# ============================================================
# Save merged data for Part 3
# ============================================================
df.to_csv(os.path.join(OUTPUT_DIR, "merged_data_part2.csv"), index=False)
print(f"\nMerged data saved to {OUTPUT_DIR}/merged_data_part2.csv")

print("\nPart II complete! Outputs in:", OUTPUT_DIR)
