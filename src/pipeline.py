from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    RANDOM_STATE,
    USE_STRATIFIED_SAMPLE,
    SAMPLE_SIZE,
    MIN_TOKEN_LEN,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    SBERT_MODEL_NAME,
    TOPIC_K_CANDIDATES,
    COHERENCE_METRIC,
    TOP_N_WORDS,
)

from .preprocessing import basic_clean, spacy_lemmatize
from .vectorization import build_tfidf, build_sbert_embeddings
from .topic_models import run_nmf, run_lda, run_bertopic, scan_k_nmf, scan_k_lda


def stratified_sample(df: pd.DataFrame, group_col: str, n: int, random_state: int) -> pd.DataFrame:
    if n >= len(df):
        return df

    group_sizes = df[group_col].value_counts(normalize=True)
    alloc = (group_sizes * n).round().astype(int)
    alloc[alloc < 1] = 1

    sampled_parts = []
    for group, k in alloc.items():
        part = df[df[group_col] == group]
        k = min(k, len(part))
        sampled_parts.append(part.sample(n=k, random_state=random_state))

    out = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state)
    if len(out) > n:
        out = out.sample(n=n, random_state=random_state)
    return out


def save_prevalence_plot(prevalence_df: pd.DataFrame, title: str, outpath: Path, top_k: int = 10):
    top = prevalence_df.head(top_k).copy()
    plt.figure()
    plt.bar(top["topic"].astype(str), top["share"])
    plt.title(title)
    plt.xlabel("Topic")
    plt.ylabel("Share of documents")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_topic_overview(
    topics_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
    outpath: Path,
) -> pd.DataFrame:
    """
    Erzeugt eine übersichtliche Tabelle je Modell:
    - Topic-ID
    - Top-Keywords (top_terms)
    - Prävalenz (n_docs, share)
    Sortierung nach share (absteigend) + Rank.
    """
    df = prevalence_df.merge(topics_df, on="topic", how="left").copy()

    # defensiv: falls "share" fehlt
    if "share" not in df.columns and "n_docs" in df.columns:
        df["share"] = df["n_docs"] / df["n_docs"].sum()

    df = df.sort_values("share", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    cols = ["rank", "topic"]
    for c in ["share", "n_docs", "top_terms"]:
        if c in df.columns:
            cols.append(c)

    df = df[cols]
    df.to_csv(outpath, index=False)
    return df


def main(input_csv: str, outdir: str):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = pd.read_csv(input_csv)

    # 2) Relevante Spalten behalten + Zeilen ohne "narrative" entfernen + Duplikate entfernen
    df = df[["product", "narrative"]].copy()
    df = df.dropna(subset=["narrative"])
    df = df.drop_duplicates(subset=["product", "narrative"])

    # 3) Stichprobe ziehen (Random = 42)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        if USE_STRATIFIED_SAMPLE:
            df = stratified_sample(df, group_col="product", n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        else:
            df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # 4) Basic cleaning
    df["clean_basic"] = df["narrative"].astype(str).apply(basic_clean)

    # 5) Lemmatization / stopword filtering
    df["clean_text"] = spacy_lemmatize(df["clean_basic"].tolist(), min_token_len=MIN_TOKEN_LEN)

    # Remove empty clean texts
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    # Save cleaned sample
    df.to_csv(outdir_path / "cleaned_sample.csv", index=False)

    texts = df["clean_text"].tolist()
    tokenized_texts = [t.split() for t in texts]

    # 6) Vectorization A: TF-IDF
    X_tfidf, tfidf_vec = build_tfidf(
        texts,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
    )

    # 7) K-Auswahl via Coherence
    nmf_k_df = scan_k_nmf(
        X_tfidf,
        tfidf_vec,
        tokenized_texts,
        k_values=TOPIC_K_CANDIDATES,
        random_state=RANDOM_STATE,
        top_n=TOP_N_WORDS,
        coherence=COHERENCE_METRIC,
    )

    lda_k_df = scan_k_lda(
        X_tfidf,
        tfidf_vec,
        tokenized_texts,
        k_values=TOPIC_K_CANDIDATES,
        random_state=RANDOM_STATE,
        top_n=TOP_N_WORDS,
        coherence=COHERENCE_METRIC,
    )

    coh_df = pd.concat([nmf_k_df, lda_k_df], ignore_index=True)
    coh_df.to_csv(outdir_path / "coherence_scan.csv", index=False)

    best_k_nmf = int(nmf_k_df.sort_values("coherence", ascending=False).iloc[0]["k"])
    best_k_lda = int(lda_k_df.sort_values("coherence", ascending=False).iloc[0]["k"])

    # Coherence-Plot
    plt.figure()
    for model_name, part in coh_df.groupby("model"):
        plt.plot(part["k"], part["coherence"], marker="o", label=model_name)
    plt.xlabel("K (n_topics)")
    plt.ylabel(f"Coherence ({COHERENCE_METRIC})")
    plt.title("Topic Count Selection via Coherence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir_path / "coherence_vs_k.png", dpi=200)
    plt.close()

    # 8) Finales Training mit best_k (TF-IDF Modelle)
    nmf_res = run_nmf(X_tfidf, tfidf_vec, n_topics=best_k_nmf, random_state=RANDOM_STATE)
    lda_res = run_lda(X_tfidf, tfidf_vec, n_topics=best_k_lda, random_state=RANDOM_STATE)

    nmf_res.topics_df.to_csv(outdir_path / "topics_nmf.csv", index=False)
    lda_res.topics_df.to_csv(outdir_path / "topics_lda.csv", index=False)
    nmf_res.prevalence_df.to_csv(outdir_path / "prevalence_nmf.csv", index=False)
    lda_res.prevalence_df.to_csv(outdir_path / "prevalence_lda.csv", index=False)

    # Tutor-Feedback: Übersicht als eigene Datei (Keywords + Prävalenz)
    save_topic_overview(nmf_res.topics_df, nmf_res.prevalence_df, outdir_path / "topic_overview_nmf.csv")
    save_topic_overview(lda_res.topics_df, lda_res.prevalence_df, outdir_path / "topic_overview_lda.csv")

    # Prevalence-Plots
    plot_top_k = 10
    save_prevalence_plot(
        nmf_res.prevalence_df,
        f"NMF Topic Prevalence (Top {plot_top_k})",
        outdir_path / "prevalence_nmf.png",
        top_k=plot_top_k,
    )
    save_prevalence_plot(
        lda_res.prevalence_df,
        f"LDA Topic Prevalence (Top {plot_top_k})",
        outdir_path / "prevalence_lda.png",
        top_k=plot_top_k,
    )

    # 9) Vectorization B: Sentence-BERT Embeddings
    embeddings = build_sbert_embeddings(texts, model_name=SBERT_MODEL_NAME)

    # 10) BERTopic
    bert_res = run_bertopic(texts, embeddings, random_state=RANDOM_STATE)
    bert_res.topics_df.to_csv(outdir_path / "topics_bertopic.csv", index=False)
    bert_res.prevalence_df.to_csv(outdir_path / "prevalence_bertopic.csv", index=False)

    # Tutor-Feedback: Übersicht für BERTopic
    save_topic_overview(bert_res.topics_df, bert_res.prevalence_df, outdir_path / "topic_overview_bertopic.csv")

    save_prevalence_plot(
        bert_res.prevalence_df,
        f"BERTopic Prevalence (Top {plot_top_k})",
        outdir_path / "prevalence_bertopic.png",
        top_k=plot_top_k,
    )

    # 11) Vergleich TF-IDF vs SBERT (kurze Run-Zusammenfassung)
    summary = {
        "n_documents_used": len(df),
        "tfidf_shape": [int(X_tfidf.shape[0]), int(X_tfidf.shape[1])],
        "sbert_embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "best_k_nmf": best_k_nmf,
        "best_k_lda": best_k_lda,
        "note": "TF-IDF sparse & interpretable; SBERT dense & semantisch (Synonyme/Paraphrasen).",
    }
    (outdir_path / "run_summary.txt").write_text(str(summary), encoding="utf-8")

    print("Done. Outputs written to:", outdir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Bank_complaints.csv")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.outdir)