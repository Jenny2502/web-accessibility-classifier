# AccessGuru ML Project Report

## Problem Statement

We need to understand why accessibility violations happen, identify the dominant patterns that create exclusion, and build a model that predicts violation types from HTML/styling content. The goal is to triage risk and suggest remediation priorities.

## Dataset Preliminary Review

Dataset: `Access_to_Tech_Dataset.csv`

Key columns:

- `web_URL_id`, `web_URL`, `domain_category`
- `scrape_status`
- `html_file_name`, `html_file_path`
- `violation_name`, `violation_description`, `violation_description_url`
- `affected_html_elements`
- `violation_category`, `violation_impact`, `violation_score`
- `wcag_reference`, `supplementary_information`

Initial observations from the dataset:

- Multiple violation types per page (multi-label at the page level).
- Many records are syntactic issues with serious/critical impacts.
- Some categories have very few samples, which complicates stratified splits.

## Data Cleaning Methodology

1. Read CSV with a robust encoding fallback (`latin-1`, `cp1252`, `utf-8`), and skip malformed rows.
2. Filter only successfully scraped rows (`scrape_status` in `scraped`, `success`, `successful`, etc.).
3. Normalize whitespace and drop empty `violation_name`.
4. Build a training text feature by concatenating:
   - `affected_html_elements`
   - `supplementary_information`
   - `violation_description`
5. Optionally include raw HTML content from `html_file_path` if available.
6. Drop classes with fewer than `min_class_count` samples to enable stratified train/test splits.
7. Define high-severity violations as:
   - `violation_impact` in `{critical, serious}`, or
   - `violation_score >= 4`.

## Thinking Flow and Solution Rationale

1. **Goal:** Predict violation type from HTML-like evidence.
2. **Representation:** The dataset already includes HTML snippets and attributes in `affected_html_elements` and `supplementary_information`, so we treated them as text tokens.
3. **Model choice:** TF-IDF + LinearSVC is a strong baseline for sparse text features and avoids heavy compute.
4. **Evaluation:** Use a stratified train/test split to keep class proportions aligned.
5. **Application:** Parse new HTML and extract tags + accessibility-relevant attributes to produce a similar text feature at inference time.
6. **Reporting:** Provide top predicted violation types, mark those that are high-severity, and include human-readable remediation hints.

## Extraction Mechanism (HTML/Styling)

For inference, the application:

- Keeps the raw HTML text as part of the feature input.
- Parses tags and attributes:
  - `aria-*`, `role`, `alt`, `title`, `name`, `id`, `class`, `href`, `type`, `for`, `style`
- Adds visible text nodes to the feature string.

This mirrors the dataset’s violation context and provides signal for ARIA/landmark/label patterns.

For reporting, the application also extracts small HTML snippets that match each predicted violation label
so users can see concrete code fragments to investigate. These snippets are heuristic matches, not confirmed violations.

## Model Evaluation (On the Dataset)

We trained the model with:

```bash
python train_violation_model.py --data Access_to_Tech_Dataset.csv
```

Results (hold-out split, `min_class_count=3`):

- Accuracy: 0.87
- Macro F1: 0.85
- Weighted F1: 0.86

Top high-severity types (by frequency):

- `color-contrast-enhanced`
- `color-contrast`
- `link-name`
- `image-alt`
- `button-name`
- `duplicate-id-aria`
- `duplicate-id-active`
- `meta-viewport`
- `aria-allowed-attr`
- `html-has-lang`

## Application Flow (How We Use the Model)

1. Fetch or load HTML.
2. Extract features (tags, key attributes, text nodes).
3. Predict top-K violation types.
4. Flag predictions that are high-severity.
5. Attach heuristic code snippets from the input that match each predicted violation.
6. Produce a report (text, JSON, or HTML).

## Sample Test Cases

Training:

```bash
python train_violation_model.py \
  --data Access_to_Tech_Dataset.csv \
  --model-out violation_type_model.joblib \
  --high-out high_violations.json
```

Apply to a URL:

```bash
python apply_violation_model.py \
  --url https://example.com \
  --model violation_type_model.joblib \
  --high-from-json high_violations.json \
  --report-format html \
  --report-out report.html
```

Apply to a local HTML file:

```bash
python apply_violation_model.py \
  --file ./page.html \
  --model violation_type_model.joblib \
  --report-format text
```

Apply to raw HTML:

```bash
python apply_violation_model.py \
  --text "<button></button><img src='x'>" \
  --model violation_type_model.joblib \
  --report-format json
```

## Improvement Ideas

- Train on raw HTML only (drop violation description fields) to reduce label leakage.
- Add DOM-level features: heading depth, landmark counts, form label coverage, image alt coverage.
- Compute contrast ratios from CSS to directly flag contrast issues.
- Use multi-label learning at the page level instead of per-violation row.
- Evaluate with site-level splits to measure generalization to unseen websites.
