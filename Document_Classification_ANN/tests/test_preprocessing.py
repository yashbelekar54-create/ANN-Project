import numpy as np
import pandas as pd
from scipy import sparse

from src.data_loader import add_splits, quality_report
from src.preprocessing import (TextPreprocessor, build_vectorizer, clean_text, fit_vectorizer, load_vectorizer,
                               remove_stopwords, save_vectorizer, strip_metadata, tokenize)

TOY_CFG = {"max_features": 100, "ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "sublinear_tf": True,
           "stop_words": "english"}
CORPUS = ["the space shuttle launch was great", "hockey game tonight in the arena", "encryption keys and security"]


def test_clean_text_normalises():
    assert clean_text("  Hello,   WORLD!!\n\tTest. ") == "hello world test"


def test_clean_text_removes_urls_and_emails():
    out = clean_text("Mail john@example.com or visit https://example.org/page now")
    assert "example" not in out and "john" not in out and out == "mail or visit now"


def test_clean_text_non_string_and_empty():
    assert clean_text(None) == "" and clean_text("") == "" and clean_text("!!!") == ""


def test_tokenize_and_stopwords():
    tokens = tokenize(clean_text("The rocket is a fast machine"))
    assert tokens == ["the", "rocket", "is", "fast", "machine"]
    assert remove_stopwords(tokens) == ["rocket", "fast", "machine"]


def test_strip_metadata_removes_headers_quotes_signature():
    post = ("From: a@b.com\nSubject: Re: engines\nNewsgroups: rec.autos\n\n"
            "John wrote:\n> old quoted text\nThe engine needs new spark plugs.\n-- \nJohn Smith\nSome University")
    out = strip_metadata(post)
    assert "Newsgroups" not in out and "quoted" not in out and "John Smith" not in out
    assert "spark plugs" in out


def test_strip_metadata_keeps_ordinary_documents():
    doc = "Annual report\n\nRevenue grew by ten percent during the year."
    assert "Revenue grew" in strip_metadata(doc)


def test_preprocessor_falls_back_when_everything_is_stripped():
    assert TextPreprocessor(strip_meta=True)("> only quoted text here") == "only quoted text here"


def test_vectorizer_is_sparse_float32_with_bigrams():
    vec = fit_vectorizer([clean_text(t) for t in CORPUS], TOY_CFG)
    X = vec.transform([clean_text(CORPUS[0])])
    assert sparse.issparse(X) and X.dtype == np.float32
    assert any(" " in term for term in vec.get_feature_names_out())  # bigram present
    assert "the" not in vec.vocabulary_  # stopword removed


def test_vectorizer_joblib_roundtrip(tmp_path):
    vec = fit_vectorizer([clean_text(t) for t in CORPUS], TOY_CFG)
    save_vectorizer(vec, tmp_path / "vec.joblib")
    loaded = load_vectorizer(tmp_path / "vec.joblib")
    a = vec.transform(["space shuttle"]).toarray()
    b = loaded.transform(["space shuttle"]).toarray()
    assert np.allclose(a, b)


def test_default_config_values():
    from src.config import load_config
    v = build_vectorizer(load_config()["tfidf"])
    assert v.max_features == 15000 and v.ngram_range == (1, 2) and v.min_df == 2 and v.max_df == 0.95
    assert v.sublinear_tf is True and v.dtype == np.float32


def _toy_frame(n_per_class=20):
    rows = []
    for split in ("train", "test"):
        for cid, name in enumerate(["a", "b", "c"]):
            for i in range(n_per_class):
                rows.append({"original_split": split, "category": name, "category_id": cid,
                             "document_text": f"doc {name} {i} " + ("dup" if i == 0 else str(i)),
                             "text_no_metadata": f"doc {name} {i}", "cleaned_text": f"doc {name} {i % 5}"})
    rows.append({"original_split": "train", "category": "a", "category_id": 0, "document_text": "",
                 "text_no_metadata": "", "cleaned_text": ""})
    return pd.DataFrame(rows)


def test_split_is_stratified_and_keeps_test_untouched():
    df = add_splits(_toy_frame(), val_size=0.25, seed=42)
    assert (df.loc[df["original_split"] == "test", "dataset_split"] == "test").all()
    assert set(df.loc[df["original_split"] == "train", "dataset_split"]) == {"train", "validation"}
    val = df[df["dataset_split"] == "validation"]
    assert set(val["category_id"]) == {0, 1, 2}


def test_quality_report_flags_without_removing():
    df = add_splits(_toy_frame(), 0.25, 42)
    rep = quality_report(df)
    assert rep["total_documents"] == len(df)
    assert rep["empty_after_cleaning"] == 1
    assert rep["duplicate_cleaned_documents_within_dataset"] > 0
    assert rep["num_categories"] == 3
