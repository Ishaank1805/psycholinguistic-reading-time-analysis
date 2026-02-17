# Part III: FOBS Model — Summary Note

## FOBS Model

The Frequency Ordered Bin Search model organizes the mental lexicon by sorting lemmas (root forms) into frequency-ordered bins, with the most frequent words in Bin 1 and the least frequent in Bin 10. The model predicts that lexical access time increases with bin number, as the search proceeds sequentially through the frequency-ordered structure.

Our FOBS bin plot confirms this prediction clearly: mean RT increases monotonically from approximately 325 ms in Bin 1 (most frequent lemmas) to 371 ms in Bin 10 (least frequent lemmas). This steady gradient supports the core FOBS assumption that word retrieval is faster for higher-frequency roots.

## Hypothesis 1: Root Frequency vs Surface Frequency

| Metric | Model 1 (Surface Freq + Length) | Model 2 (Lemma Freq + Lemma Length) |
|--------|-------------------------------|-------------------------------------|
| R²     | 0.0576                        | 0.0483                              |
| AIC    | 81530                         | 81608                               |

Contrary to the hypothesis, surface (word-form) frequency is a slightly better predictor of reading time than root (lemma) frequency. Model 1 explains 5.76% of RT variance compared to 4.83% for Model 2, and has a lower AIC (81530 vs 81608), indicating better fit.

This suggests that readers access words primarily via their surface forms rather than through morphological decomposition to root forms. While the FOBS model's frequency-ordered structure is supported by the bin analysis, the specific prediction that root frequency should outperform surface frequency is not borne out. This is consistent with a "full-listing" view of the mental lexicon, where each inflected form has its own entry and frequency count, rather than being accessed exclusively through its root.

However, the difference between the two models is modest (~1 percentage point in R²), indicating that both surface and root frequency capture similar variance — which makes sense given the high correlation between a word's frequency and its lemma's frequency.

## Hypothesis 2: Pseudo-Affixed vs Regularly Affixed Words

| Group | Words | Mean RT | 
|-------|-------|---------|
| Pseudo-affixed | finger, corner, master | 349.4 ms |
| Regular-affixed | longer, higher, older | 341.3 ms |

The t-test comparing the two groups was not significant (t = 0.28, p = 0.7937). While pseudo-affixed words showed a numerically higher mean RT (349.4 ms vs 341.3 ms), the difference of ~8 ms is not statistically reliable.

Several factors limit this analysis. First, only 3 pseudo-affixed and 3 regular-affixed words were found in the corpus, giving very low statistical power. Second, the words are not perfectly matched on frequency and length. Third, self-paced reading may not be sensitive enough to detect the small morphological decomposition cost predicted by the FOBS model; primed lexical decision tasks have traditionally been used to study this effect.

In summary, we find a trend in the predicted direction — pseudo-affixed words are read slightly slower — but the evidence is inconclusive due to the limited sample size available in the Natural Stories corpus.
