# Methods

**Corpus:** 16 short texts (8 poetic, 8 report-like), each with `id, type, author, year, text`.

**Processing**
- Tokenization: lowercase + punctuation removal; whitespace split
- Metrics: token count, type-token ratio (unique/total), naive sentiment (small custom lexicon)
- Keywords: top non-stopword tokens per class (poetry vs report)
- Co-occurrence network: type-agnostic token co-occurrence across documents

**Visualization**
- All charts produced with matplotlib (no seaborn), one figure per plot
- Network drawn with NetworkX spring layout
