import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Acceptable scrape-status values for keeping rows.
SCRAPE_OK = {"scraped", "success", "successful", "true", "1", "yes"}


def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["latin-1", "cp1252", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", 
                               on_bad_lines="skip")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read CSV {path}: {last_err}")


def normalize_text(value: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_html_text(path: str) -> str:
    try:
        content = Path(path).read_text(errors="ignore")
        return content
    except Exception:
        return ""


def build_text(
    row: pd.Series,
    text_columns: Iterable[str],
    include_html_file: bool,
    html_column: str,
) -> str:
    # Combine relevant columns into a single training text feature.
    parts: List[str] = []
    for col in text_columns:
        parts.append(normalize_text(row.get(col)))
    if include_html_file:
        html_path = normalize_text(row.get(html_column))
        if html_path:
            parts.append(load_html_text(html_path))
    return " ".join(p for p in parts if p)


def prepare_dataset(
    df: pd.DataFrame,
    text_columns: List[str],
    include_html_file: bool,
    html_column: str,
    min_class_count: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["scrape_status"] = df["scrape_status"].astype(str).str.strip().str.lower()
    df = df[df["scrape_status"].isin(SCRAPE_OK)]

    df["violation_name"] = df["violation_name"].astype(str).str.strip()
    df = df[df["violation_name"].notna() & (df["violation_name"] != "")]

    df["text"] = df.apply(
        build_text,
        axis=1,
        text_columns=text_columns,
        include_html_file=include_html_file,
        html_column=html_column,
    )
    df = df[df["text"].str.len() > 0]

    if min_class_count > 1:
        # Drop rare classes to make stratified splitting possible.
        counts = df["violation_name"].value_counts()
        keep = counts[counts >= min_class_count].index
        df = df[df["violation_name"].isin(keep)]

    return df, df["violation_name"]


def identify_high_violations(df: pd.DataFrame) -> pd.DataFrame:
    # Flag high-severity by impact or numeric score threshold.
    impact = df["violation_impact"].astype(str).str.strip().str.lower()
    score = pd.to_numeric(df["violation_score"], errors="coerce")
    high_mask = impact.isin({"critical", "serious"}) | (score >= 4)
    high_df = df[high_mask]
    return (
        high_df["violation_name"]
        .value_counts()
        .rename_axis("violation_name")
        .reset_index(name="count")
    )


def train_and_evaluate(
    df: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, str]:
    # Train/test split with stratification and TF-IDF + linear SVM.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, zero_division=0)
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a violation-type classifier from HTML/styling text."
    )
    parser.add_argument(
        "--data",
        default="../data/Access_to_Tech_Dataset.csv",
        help="Path to AccessGuru CSV.",
    )
    parser.add_argument(
        "--text-columns",
        default="affected_html_elements,supplementary_information,violation_description",
        help="Comma-separated list of columns to use as text features.",
    )
    parser.add_argument(
        "--include-html-files",
        action="store_true",
        help="Include raw HTML from html_file_path if files are present.",
    )
    parser.add_argument(
        "--html-column",
        default="html_file_path",
        help="Column containing HTML file paths.",
    )
    parser.add_argument(
        "--model-out",
        default="violation_type_model.joblib",
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=3,
        help="Minimum samples per violation type to keep for training.",
    )
    parser.add_argument(
        "--high-out",
        help="Write high-severity violation names (JSON list) to a file.",
    )
    parser.add_argument(
        "--report-out",
        help="Write the classification report to a file.",
    )
    parser.add_argument(
        "--training-report-out",
        default="training_report.txt",
        help="Write the full training summary to a file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = read_csv_robust(Path(args.data))
    text_columns = [c.strip() for c in args.text_columns.split(",") if c.strip()]

    df, labels = prepare_dataset(
        df,
        text_columns=text_columns,
        include_html_file=args.include_html_files,
        html_column=args.html_column,
        min_class_count=args.min_class_count,
    )

    training_rows_line = f"Training rows: {len(df)}"
    unique_types_line = f"Unique violation types: {labels.nunique()}"
    print(training_rows_line)
    print(unique_types_line)

    high = identify_high_violations(df)
    high_title = "Top high-severity violations:"
    high_table = high.head(15).to_string(index=False)
    print("\n" + high_title)
    print(high_table)

    model, report = train_and_evaluate(
        df,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("\n" + report)
    joblib.dump(model, args.model_out)
    print(f"\nSaved model to {args.model_out}")

    if args.report_out:
        Path(args.report_out).write_text(report)

    if args.training_report_out:
        training_report = "\n".join(
            [
                training_rows_line,
                unique_types_line,
                "",
                high_title,
                high_table,
                "",
                report,
            ]
        )
        Path(args.training_report_out).write_text(training_report)

    if args.high_out:
        high_names = high["violation_name"].astype(str).dropna().unique().tolist()
        Path(args.high_out).write_text(json.dumps(high_names, indent=2))


if __name__ == "__main__":
    main()
