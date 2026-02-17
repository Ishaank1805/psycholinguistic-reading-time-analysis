# Part I: Summary Note

## Relationship Between Word Length, Frequency, and Mean Reading Time

Our analysis of the Natural Stories corpus reveals three significant relationships between word length, word frequency, and mean reading time (RT).

**Word Length and Frequency (r = −0.7202, p ≈ 0):**
A strong negative correlation exists between word length and log-transformed word frequency. Shorter words tend to be far more frequent in the language, while longer words are comparatively rare. This is consistent with Zipf's Law of Abbreviation, which states that more frequent words tend to be shorter.

**Word Length and Mean RT (r = 0.2462, p = 2.61e−141):**
A moderate positive correlation indicates that longer words take more time to read. The aggregated plot shows a steady increase in mean RT from approximately 280 ms for 1–2 character words to around 420 ms for 13+ character words. This likely reflects the additional time required for visual processing and lexical access of longer orthographic forms.

**Word Frequency and Mean RT (r = −0.2282, p = 4.23e−121):**
A moderate negative correlation shows that more frequent words are read faster. The binned plot demonstrates a clear downward trend: words with the lowest frequencies (log₁₀ frequency ≈ 0) have mean RTs around 550 ms, while the most frequent words (log₁₀ frequency ≈ 10) are read in approximately 320 ms. This aligns with the well-established word frequency effect in psycholinguistics, where frequently encountered words have lower activation thresholds in the mental lexicon and are thus recognized more quickly.

**Confound Between Predictors:**
The strong negative correlation between word length and frequency (r = −0.72) means these two predictors are heavily confounded. Part of the length effect on RT may be driven by the fact that longer words are less frequent, and vice versa. Disentangling these contributions requires regression models that include both predictors simultaneously, which is addressed in Part II.
