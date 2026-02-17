# Natural Stories Reading Time Analysis

Psycholinguistic analysis of the [Natural Stories Corpus](https://github.com/languageMIT/naturalstories) (Futrell et al., 2021), examining the relationships between word frequency, word length, language model surprisal, and self-paced reading times.

## Overview

This project investigates three main questions using reading time data from 181 native English speakers:

1. **How do word length and frequency relate to reading time?**
2. **Are language model (GPT-3) probabilities better predictors of reading time than word frequency? Do content and function words differ in processing?**
3. **Does the Frequency Ordered Bin Search (FOBS) model of lexical access explain reading time patterns?**

## Repository Structure

```
├── README.md
├── report.pdf                  # Final PDF report
├── report.tex                  # LaTeX source for the report
├── part1_analysis.py           # Part I: Preliminary data analysis
├── part2_analysis.py           # Part II: Hypothesis testing (GPT-3 surprisal, content/function words)
├── part3_analysis.py           # Part III: FOBS model and morphological analysis
├── data/                       # Data files (not included — see below)
│   ├── processed_RTs.tsv
│   ├── freqs-1.tsv
│   ├── freqs-2.tsv
│   ├── freqs-3.tsv
│   ├── freqs-4.tsv
│   └── all_stories_gpt3.csv
├── part1_outputs/              # Part I plots and intermediate data
├── part2_outputs/              # Part II plots and intermediate data
└── part3_outputs/              # Part III plots and intermediate data
```

## Data

The data files are from the Natural Stories Corpus and are **not included** in this repository. Download them from:

- **Corpus:** https://github.com/languageMIT/naturalstories
  - `processed_RTs.tsv` — Self-paced reading times
  - `freqs-1.tsv` through `freqs-4.tsv` — N-gram frequency data from Google Books
  - `all_stories_gpt3.csv` — GPT-3 (davinci) token-level log-probabilities

## Requirements

```bash
pip install pandas numpy matplotlib scipy statsmodels nltk
```

NLTK data (for Part III):
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')"
```

## Usage

Run the scripts sequentially — each part depends on the output of the previous:

```bash
# Part I: Correlations and plots
python part1_analysis.py

# Part II: Regression models (GPT-3 surprisal, content/function words)
python part2_analysis.py

# Part III: FOBS model, lemma frequency, pseudo-affix analysis
python part3_analysis.py
```

## Key Findings

### Part I
| Variable Pair | Pearson r |
|---|---|
| Word Length vs Log Frequency | −0.7202 |
| Word Length vs Mean RT | +0.2462 |
| Log Frequency vs Mean RT | −0.2282 |

Shorter words are more frequent, longer words are read slower, and more frequent words are read faster — replicating standard psycholinguistic effects.

### Part II
- **GPT-3 surprisal** (R² = 0.0775) outperforms **word frequency** (R² = 0.0658) as a predictor of reading time, supporting the hypothesis that contextual predictability matters beyond raw frequency.
- **Content words** show much stronger frequency/surprisal effects (R² ≈ 0.10) than **function words** (R² ≈ 0.02–0.03), confirming different processing mechanisms for the two word classes.

### Part III
- The **FOBS model** shows a clear monotonic increase in RT across frequency bins (325 ms → 371 ms).
- **Surface frequency** (R² = 0.0576) slightly outperforms **lemma frequency** (R² = 0.0483), suggesting words are accessed primarily via surface forms.
- **Pseudo-affixed** words show numerically higher RT than regularly affixed words (349.4 vs 341.3 ms), but the difference is not significant (p = 0.794) due to limited sample size.

## Reference

Futrell, R., Gibson, E., Tily, H. J., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, E. (2021). The Natural Stories corpus: a reading-time corpus of English texts containing rare syntactic constructions. *Language Resources and Evaluation*, 55, 63–77. https://doi.org/10.1007/s10579-020-09503-7
