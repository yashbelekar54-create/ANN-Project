# Data

**Dataset:** 20 Newsgroups ("bydate" version), loaded through
`sklearn.datasets.fetch_20newsgroups`
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html).
Original source: http://qwone.com/~jason/20Newsgroups/ (about 18,846 posts after the bydate split: 11,314 train + 7,532 test, 20 categories).

The dataset is **not bundled** in this ZIP: it must be downloaded through the official scikit-learn loader (about 14 MB download), which this project does for you:

```
python -m src.data_loader
```

This command

1. downloads the dataset into `data/raw/` (once; later runs reuse the cache),
2. verifies there are exactly 20 categories (taken from the dataset's own `target_names`),
3. builds `data/processed/newsgroups_processed.csv` with columns
   `document_id, source_file, original_split, dataset_split, category, category_id, document_text, text_no_metadata, cleaned_text, is_empty_after_cleaning, is_duplicate_cleaned_text`,
4. writes `results/metrics/dataset_summary.json` (counts per category and split, missing / empty / duplicate analysis, and the cleaning decisions).

Splits: the original train/test separation is kept. 15% of the original training portion is held out as
validation (stratified, seed 42). The test split is never used for fitting or tuning.

No document is removed; empty or duplicate documents are only flagged.
