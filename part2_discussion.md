# Part II: Hypothesis Testing — Discussion

## Hypothesis 1: LM Probabilities vs Word Frequency

| Metric | Model 1 (Freq + Length) | Model 2 (GPT-3 Surprisal + Length) |
|--------|------------------------|------------------------------------|
| R²     | 0.0658                 | 0.0775                             |

Model 2 (GPT-3 surprisal + word length) provides a better fit to reading time data than Model 1 (word frequency + word length), with an R² of 0.0775 compared to 0.0658. This supports the hypothesis that language model probabilities are better predictors of reading time than word frequency alone.

The advantage of GPT-3 surprisal likely arises because it captures contextual predictability — how expected a word is given the preceding sentence context — rather than just how common a word is in isolation. A word like "bank" may be frequent overall, but highly surprising after "the pilot landed the plane on the river ___." GPT-3 surprisal reflects this contextual information, whereas unigram frequency does not.

That said, both models explain a relatively small proportion of the total variance in reading times (< 8%), indicating that many other factors (e.g., syntactic complexity, semantic plausibility, individual differences) also contribute to reading time variation. The residual plots for both models show similar patterns of heteroscedasticity, with larger residuals at higher fitted values.

## Hypothesis 2: Content Words vs Function Words

| Model | Word Type | Predictor | R² | AIC |
|-------|-----------|-----------|-----|-----|
| M1 | Content  | Freq + Length     | 0.1028 | 40783 |
| M2 | Content  | Surprisal + Length | 0.0991 | 40799 |
| M3 | Function | Freq + Length     | 0.0230 | 40557 |
| M4 | Function | Surprisal + Length | 0.0306 | 40525 |

**Content words are processed differently than function words.** The key findings are:

1. **Content words show much higher predictability from both frequency and surprisal** (R² ≈ 0.10) compared to function words (R² ≈ 0.02–0.03). This is a substantial difference, indicating that lexical properties like frequency and contextual predictability have a much stronger influence on reading times for content words (nouns, verbs, adjectives, adverbs) than for function words (determiners, prepositions, conjunctions, auxiliaries).

2. **For content words, frequency slightly outperforms surprisal** (R² = 0.1028 vs 0.0991, AIC 40783 vs 40799). This suggests that for open-class words, raw lexical frequency is a marginally better predictor than contextual probability.

3. **For function words, surprisal slightly outperforms frequency** (R² = 0.0306 vs 0.0230, AIC 40525 vs 40557). Since function words are mostly high-frequency and short, there is less variance in their frequency to explain RT differences. GPT-3 surprisal captures contextual factors that create the small RT variations observed among function words.

4. **The distribution plots confirm fundamental differences** between the two classes: function words are concentrated at high frequencies (log₁₀ freq > 7) and low surprisal (< 5 bits), while content words are spread across a much wider range of both frequency and surprisal.

These results are consistent with dual-route models of lexical access, where function words may be processed via a faster, more automatized route that is less sensitive to frequency variation, while content words rely more on full lexical retrieval where frequency plays a larger role.
