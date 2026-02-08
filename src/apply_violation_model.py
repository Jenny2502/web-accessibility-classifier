import argparse
import datetime as dt
import html as html_lib
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


class HTMLFeatureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        # Capture tag names and accessibility-relevant attributes.
        self.parts.append(f"tag={tag}")
        for key, value in attrs:
            if key in {
                "role",
                "alt",
                "title",
                "name",
                "id",
                "class",
                "href",
                "type",
                "for",
                "style",
            } or key.startswith("aria-"):
                if value is None:
                    self.parts.append(f"{key}=")
                else:
                    self.parts.append(f"{key}={value}")

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def extract_features_from_html(html: str, max_chars: int) -> str:
    # Combine raw HTML with parsed tokens to mirror training signals.
    if len(html) > max_chars:
        html = html[:max_chars]
    parser = HTMLFeatureParser()
    parser.feed(html)
    extracted = " ".join(parser.parts)
    return " ".join([html, extracted]).strip()


def fetch_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="ignore")


def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["latin-1", "cp1252", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read CSV {path}: {last_err}")


def load_high_violation_set(csv_path: Optional[Path], json_path: Optional[Path]) -> set:
    # Load high-severity labels from CSV and/or precomputed JSON list.
    high_set = set()
    if csv_path:
        df = read_csv_robust(csv_path)
        impact = df["violation_impact"].astype(str).str.strip().str.lower()
        score = pd.to_numeric(df["violation_score"], errors="coerce")
        high_mask = impact.isin({"critical", "serious"}) | (score >= 4)
        high_set.update(
            df.loc[high_mask, "violation_name"]
            .astype(str)
            .str.strip()
            .dropna()
            .unique()
        )
    if json_path:
        data = json.loads(json_path.read_text())
        if isinstance(data, list):
            high_set.update(str(x).strip() for x in data if str(x).strip())
    return high_set


def trim_snippet(snippet: str, max_len: int) -> str:
    snippet = " ".join(snippet.split())
    if len(snippet) > max_len:
        return snippet[: max_len - 3] + "..."
    return snippet


def _find_matches(html: str, pattern: str, max_snippets: int) -> List[str]:
    snippets: List[str] = []
    for match in re.finditer(pattern, html, flags=re.IGNORECASE | re.DOTALL):
        snippets.append(match.group(0))
        if len(snippets) >= max_snippets:
            break
    return snippets


def _find_img_missing_alt(html: str, max_snippets: int) -> List[str]:
    snippets: List[str] = []
    for match in re.finditer(r"<img\b[^>]*>", html, flags=re.IGNORECASE):
        tag = match.group(0)
        alt_match = re.search(r'alt\s*=\s*["\']([^"\']*)["\']', tag, re.IGNORECASE)
        if alt_match is None or alt_match.group(1).strip() == "":
            snippets.append(tag)
        if len(snippets) >= max_snippets:
            break
    return snippets


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _find_buttons_without_text(html: str, max_snippets: int) -> List[str]:
    snippets: List[str] = []
    for match in re.finditer(
        r"<button\b[^>]*>(.*?)</button>", html, flags=re.IGNORECASE | re.DOTALL
    ):
        tag = match.group(0)
        inner = _strip_tags(match.group(1)).strip()
        if not inner and "aria-label" not in tag.lower():
            snippets.append(tag)
        if len(snippets) >= max_snippets:
            break
    return snippets


def _find_links_without_text(html: str, max_snippets: int) -> List[str]:
    snippets: List[str] = []
    for match in re.finditer(
        r"<a\b[^>]*>(.*?)</a>", html, flags=re.IGNORECASE | re.DOTALL
    ):
        tag = match.group(0)
        inner = _strip_tags(match.group(1)).strip()
        if not inner and "aria-label" not in tag.lower():
            snippets.append(tag)
        if len(snippets) >= max_snippets:
            break
    return snippets


def _find_duplicate_ids(html: str, max_snippets: int) -> List[str]:
    ids: Dict[str, int] = {}
    for match in re.finditer(r"id\s*=\s*[\"']([^\"']+)[\"']", html, re.IGNORECASE):
        ids[match.group(1)] = ids.get(match.group(1), 0) + 1
    duplicates = {key for key, count in ids.items() if count > 1}
    if not duplicates:
        return []
    snippets: List[str] = []
    for dup in duplicates:
        pattern = rf"<[^>]+id\s*=\s*[\"']{re.escape(dup)}[\"'][^>]*>"
        snippets.extend(_find_matches(html, pattern, max_snippets))
        if len(snippets) >= max_snippets:
            break
    return snippets[:max_snippets]


def extract_snippets_for_label(
    html: str,
    label: str,
    max_snippets: int,
    max_len: int,
) -> List[str]:
    # Heuristic mapping between violation types and likely DOM patterns.
    patterns = {
        "color-contrast": [r"style\s*=\s*[\"'][^\"']*(color|background-color)[^\"']*[\"']"],
        "color-contrast-enhanced": [r"style\s*=\s*[\"'][^\"']*(color|background-color)[^\"']*[\"']"],
        "region": [r"role\s*=\s*[\"'](region|main|navigation|banner|contentinfo|complementary)[\"']"],
        "landmark-one-main": [r"<main\b[^>]*>", r"role\s*=\s*[\"']main[\"']"],
        "landmark-unique": [r"role\s*=\s*[\"'](main|navigation|banner|contentinfo|complementary)[\"']"],
        "page-has-heading-one": [r"<h1\b[^>]*>.*?</h1>"],
        "heading-order": [r"<h[1-6]\b[^>]*>.*?</h[1-6]>"],
        "image-alt": [r"<img\b[^>]*>"],
        "button-name": [r"<button\b[^>]*>.*?</button>", r"<input\b[^>]*type\s*=\s*[\"']?(button|submit|reset)[\"']?[^>]*>"],
        "link-name": [r"<a\b[^>]*>.*?</a>"],
        "meta-viewport": [r"<meta\b[^>]*name\s*=\s*[\"']viewport[\"'][^>]*>"],
        "html-has-lang": [r"<html\b[^>]*>"],
        "lang-mismatch": [r"lang\s*=\s*[\"'][^\"']+[\"']"],
        "frame-title": [r"<iframe\b[^>]*>"],
        "list": [r"<ul\b[^>]*>.*?</ul>", r"<ol\b[^>]*>.*?</ol>"],
        "listitem": [r"<li\b[^>]*>.*?</li>"],
        "tabindex": [r"tabindex\s*=\s*[\"']-?\d+[\"']"],
        "skip-link": [r"<a\b[^>]*href\s*=\s*[\"']#"],
        "nested-interactive": [r"<button\b[^>]*>.*?<a\b", r"<a\b[^>]*>.*?<button\b"],
    }

    snippets: List[str] = []
    if label == "image-alt":
        snippets = _find_img_missing_alt(html, max_snippets)
    elif label == "button-name":
        snippets = _find_buttons_without_text(html, max_snippets)
    elif label == "link-name":
        snippets = _find_links_without_text(html, max_snippets)
    elif label.startswith("duplicate-id"):
        snippets = _find_duplicate_ids(html, max_snippets)
    elif label.startswith("aria-"):
        snippets = _find_matches(html, r"aria-[a-zA-Z-]+\s*=\s*[\"'][^\"']*[\"']", max_snippets)
    elif label in patterns:
        for pattern in patterns[label]:
            snippets.extend(_find_matches(html, pattern, max_snippets))
            if len(snippets) >= max_snippets:
                break

    # Fallback: return a few tag snippets if no match.
    if not snippets:
        snippets = _find_matches(html, r"<[^>]+>", max_snippets)

    return [trim_snippet(s, max_len) for s in snippets[:max_snippets]]


def predict_from_text(
    model,
    text: str,
    top_k: int,
) -> List[Dict[str, float]]:
    # Use decision_function for LinearSVC; fallback to predict_proba if available.
    clf = model.named_steps.get("clf")
    classes = clf.classes_
    if hasattr(model, "decision_function"):
        scores = model.decision_function([text])[0]
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba([text])[0]
    else:
        label = model.predict([text])[0]
        return [{"label": label, "score": 0.0}]

    scores = np.array(scores, dtype=float)
    idx = np.argsort(scores)[-top_k:][::-1]
    return [{"label": classes[i], "score": float(scores[i])} for i in idx]


def build_recommendations(labels: List[str]) -> List[str]:
    # Provide lightweight remediation hints for the top predictions.
    recs: List[str] = []

    def add(text: str) -> None:
        if text not in recs:
            recs.append(text)

    for label in labels:
        if "color-contrast" in label:
            add("Check text/background contrast ratios and update colors.")
        if "image-alt" in label or label in {"role-img-alt", "svg-img-alt"}:
            add("Provide meaningful alt text for informative images.")
        if "link-name" in label:
            add("Ensure links have accessible names (text or aria-label).")
        if "button-name" in label:
            add("Ensure buttons have accessible names.")
        if "label" in label and "label-mismatch" not in label:
            add("Associate form inputs with visible labels using for/id.")
        if "heading" in label:
            add("Use a single H1 and a logical heading order.")
        if label.startswith("landmark") or label == "region":
            add("Add unique ARIA landmarks for page regions.")
        if "duplicate-id" in label:
            add("Ensure all element IDs are unique.")
        if label.startswith("aria-") or "aria-" in label:
            add("Validate ARIA roles/attributes and required structure.")
        if "meta-viewport" in label:
            add("Set a correct viewport meta tag for zoom and responsive layouts.")
        if "tabindex" in label:
            add("Avoid positive tabindex and keep a logical focus order.")
        if "frame-title" in label:
            add("Add unique, descriptive titles to frames/iframes.")
        if label in {"list", "listitem"}:
            add("Use proper list markup with ul/ol and li elements.")
    return recs


def build_report(
    preds: List[Dict[str, float]],
    high_set: set,
    source: Dict[str, str],
    model_path: str,
    max_chars: int,
    raw_html: str,
    snippets_per_violation: int,
    snippet_max_len: int,
) -> Dict[str, object]:
    # Risk level is derived from how many top predictions are high-severity.
    labels = [p["label"] for p in preds]
    high_count = sum(1 for p in preds if p["label"] in high_set)
    if high_count >= max(2, len(preds) // 3):
        risk_level = "high"
    elif high_count >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "source": source,
        "model": model_path,
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "max_chars": max_chars,
        "risk_level": risk_level,
        "top_predictions": [
            {
                "violation_name": p["label"],
                "score": p["score"],
                "high_severity": p["label"] in high_set,
                "snippets": extract_snippets_for_label(
                    raw_html,
                    p["label"],
                    max_snippets=snippets_per_violation,
                    max_len=snippet_max_len,
                ),
            }
            for p in preds
        ],
        "analysis": {
            "high_severity_count": high_count,
            "recommendations": build_recommendations(labels),
            "notes": [
                "Scores are model margins, not probabilities.",
                "Snippets are heuristic matches, not confirmed violations.",
            ],
        },
    }


def format_report_text(report: Dict[str, object]) -> str:
    lines = []
    source = report["source"]
    lines.append("AccessGuru Violation Prediction Report")
    lines.append(f"Source: {source['type']}={source['value']}")
    lines.append(f"Risk level: {report['risk_level']}")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("Top predicted violation types:")
    for entry in report["top_predictions"]:
        flag = "HIGH" if entry["high_severity"] else "normal"
        snippet = ""
        if entry["snippets"]:
            snippet = f" snippet={entry['snippets'][0]}"
        lines.append(
            f"- {entry['violation_name']} score={entry['score']:.4f} severity={flag}{snippet}"
        )
    lines.append("")
    lines.append("Analysis:")
    lines.append(
        f"- High-severity in top list: {report['analysis']['high_severity_count']}"
    )
    for rec in report["analysis"]["recommendations"]:
        lines.append(f"- {rec}")
    for note in report["analysis"]["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def format_report_json(report: Dict[str, object]) -> str:
    return json.dumps(report, indent=2)


def format_report_html(report: Dict[str, object]) -> str:
    source = report["source"]
    rows = []
    for entry in report["top_predictions"]:
        cls = "high" if entry["high_severity"] else "normal"
        snippet_html = ""
        if entry["snippets"]:
            joined = "\n".join(entry["snippets"])
            snippet_html = f"<pre>{html_lib.escape(joined)}</pre>"
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(entry['violation_name'])}</td>"
            f"<td>{entry['score']:.4f}</td>"
            f"<td class=\"{cls}\">{'HIGH' if entry['high_severity'] else 'normal'}</td>"
            f"<td>{snippet_html}</td>"
            "</tr>"
        )

    recs = "".join(
        f"<li>{html_lib.escape(text)}</li>"
        for text in report["analysis"]["recommendations"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AccessGuru Violation Report</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --text: #1b1b1b;
      --muted: #5b5b5b;
      --border: #dad4c7;
      --accent: #1f7a5e;
      --high: #b23a48;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: "Georgia", "Times New Roman", serif;
      margin: 32px;
      line-height: 1.5;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 24px;
    }}
    .card {{
      background: white;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      margin: 16px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 8px;
      vertical-align: top;
    }}
    .high {{
      color: var(--high);
      font-weight: bold;
    }}
    .risk {{
      font-weight: bold;
      color: var(--accent);
    }}
    pre {{
      white-space: pre-wrap;
      font-size: 12px;
      background: #f4f2ec;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 8px;
      margin: 0;
    }}
  </style>
</head>
<body>
  <h1>AccessGuru Violation Prediction Report</h1>
  <div class="meta">
    Source: {html_lib.escape(source['type'])}={html_lib.escape(source['value'])} |
    Generated: {html_lib.escape(report['generated_at'])} |
    Model: {html_lib.escape(report['model'])}
  </div>

  <div class="card">
    <div>Risk level: <span class="risk">{html_lib.escape(report['risk_level'])}</span></div>
    <div>High-severity predictions: {report['analysis']['high_severity_count']}</div>
    <div>Max chars analyzed: {report['max_chars']}</div>
  </div>

  <div class="card">
    <h2>Top Predicted Violation Types</h2>
    <table>
      <thead>
        <tr>
          <th>Violation Type</th>
          <th>Score</th>
          <th>Severity</th>
          <th>Detected Snippets</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Analysis</h2>
    <ul>
      {''.join(f'<li>{html_lib.escape(note)}</li>' for note in report['analysis']['notes'])}
    </ul>
    <h3>Recommended Focus Areas</h3>
    <ul>
      {recs if recs else '<li>No specific recommendations from the top predictions.</li>'}
    </ul>
  </div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a violation-type classifier to HTML, text, or a URL."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--url", help="URL to fetch and inspect.")
    inputs.add_argument("--file", help="Local HTML/text file to inspect.")
    inputs.add_argument("--text", help="Raw HTML/text input to inspect.")
    parser.add_argument(
        "--model",
        default="violation_type_model.joblib",
        help="Model path for prediction.",
    )
    parser.add_argument(
        "--high-from-csv",
        help="CSV path to derive high-severity violation set.",
    )
    parser.add_argument(
        "--high-from-json",
        help="JSON path with high-severity violation list.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top predicted violation types to show.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500000,
        help="Max characters of HTML to analyze.",
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "json", "html"],
        default="text",
        help="Prediction report format.",
    )
    parser.add_argument(
        "--report-out",
        help="Write report output to a file.",
    )
    parser.add_argument(
        "--snippets-per-violation",
        type=int,
        default=2,
        help="Number of code snippets to include per violation type.",
    )
    parser.add_argument(
        "--snippet-max-len",
        type=int,
        default=240,
        help="Max length for each snippet in reports.",
    )
    args = parser.parse_args()

    if args.report_format == "html" and not args.report_out:
        args.report_out = "violation_report.html"

    if args.url:
        raw = fetch_url(args.url)
        source = {"type": "url", "value": args.url}
    elif args.file:
        raw = Path(args.file).read_text(errors="ignore")
        source = {"type": "file", "value": args.file}
    else:
        raw = args.text or ""
        source = {"type": "text", "value": "inline"}

    model = joblib.load(args.model)
    features = extract_features_from_html(raw, max_chars=args.max_chars)

    high_set = load_high_violation_set(
        Path(args.high_from_csv) if args.high_from_csv else None,
        Path(args.high_from_json) if args.high_from_json else None,
    )
    preds = predict_from_text(model, features, top_k=args.top_k)
    report = build_report(
        preds=preds,
        high_set=high_set,
        source=source,
        model_path=args.model,
        max_chars=args.max_chars,
        raw_html=raw,
        snippets_per_violation=args.snippets_per_violation,
        snippet_max_len=args.snippet_max_len,
    )

    if args.report_format == "json":
        output = format_report_json(report)
    elif args.report_format == "html":
        output = format_report_html(report)
    else:
        output = format_report_text(report)

    if args.report_out:
        Path(args.report_out).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
