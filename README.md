# AccessGuru Violation Type Predictor

This tool trains a model to predict accessibility violation types from HTML and related content, and applies the model to inspect new pages or text.

**Project Links:**
- 📂 [GitHub Repository](https://github.com/kthudoan/accesstotech)
- 📄 [Full Documentation](https://docs.google.com/document/d/1OCUD5zRZ-SbzOF7GjGtmmDr7n5IG9jLFHCaUi4cgiCE/edit?usp=sharing)

---

## Files

* `train_violation_model.py`: Model training + evaluation.
* `apply_violation_model.py`: Apply the model to a URL, file, or raw text.
* `Access_to_Tech_Dataset.csv`: Source dataset.
* `violation_type_model.joblib`: Trained model output (created after training).

---

## Setup

### Required Python packages:

* pandas
* numpy
* scikit-learn
* joblib

### Optional:

* None. URL fetching uses Python standard library (`urllib`).

### Install with:

```bash
pip install pandas numpy scikit-learn joblib
```

Or use the pinned dependency list:

```bash
pip install -r requirements.txt
```

---

## Feature Extraction

Training text features are built from these dataset columns by default:

* `affected_html_elements`
* `supplementary_information`
* `violation_description`

Optional: add raw HTML via `--include-html-files` to read from `html_file_path`.

For prediction, the tool:

1. Uses the raw HTML/text input.
2. Parses tags and key attributes (`aria-*`, `role`, `alt`, `title`, `name`, `id`, `class`, `href`, `type`, `for`, `style`).
3. Appends visible text nodes.

This creates a single text blob passed to the model.

---

## Model Training + Evaluation

Train the model and print a classification report:

```bash
python train_violation_model.py --data Access_to_Tech_Dataset.csv
```

### Key options:

* `--include-html-files`: Adds raw HTML from `html_file_path` into training text.
* `--min-class-count 3`: Drops rare violation types with fewer than N samples.
* `--model-out violation_type_model.joblib`: Output model path.
* `--test-size 0.2`: Train/test split ratio.
* `--high-out high_violations.json`: Save a JSON list of high-severity types.
* `--training-report-out training_report.txt`: Save a full training summary (rows, high-severity table, and classification report).

### Example with high-severity export:

```bash
python train_violation_model.py \
  --data Access_to_Tech_Dataset.csv \
  --model-out violation_type_model.joblib \
  --high-out high_violations.json
```

### Notes:

* Some violation types appear only once; use `--min-class-count` to keep stratified splits valid.
* The model is a linear SVM over TF-IDF (uni/bi-grams).

---

## High-Severity Identification

The tool flags high-severity violations as:

* `violation_impact` in `{critical, serious}`, or
* `violation_score >= 4`.

When training, the script prints the most frequent high-severity violation types. When predicting, `--high-from-csv` or `--high-from-json` is optional and only used to mark which predictions are high-severity.

---

## Apply the Model (Predict)

### General usage:

```bash
python apply_violation_model.py --model violation_type_model.joblib --top-k 10 --report-format html
```

### Control output:

* `--top-k 10`: Number of predicted violation types.
* `--report-format json`: JSON report instead of plain text.
* `--report-format html --report-out report.html`: HTML report output with analysis.
* If `--report-format html` is set without `--report-out`, the report is written to `violation_report.html`.
* `--report-out report.json`: Write report to a file.
* `--snippets-per-violation 2`: Include up to N detected code snippets per violation.
* `--snippet-max-len 240`: Limit snippet length in reports.

---

## Apply to a Website (URL)

```bash
python apply_violation_model.py \
  --url https://example.com \
  --model violation_type_model.joblib \
  --report-format html --report-out report.html
```

---

## Apply to Local HTML/Text

### Inspect a local HTML file:

```bash
python apply_violation_model.py \
  --file ./page.html \
  --model violation_type_model.joblib
```

### Inspect raw text or HTML:

```bash
python apply_violation_model.py \
  --text "<button></button><img src='x'>" \
  --model violation_type_model.joblib
```

### Example JSON report:

```json
{
  "top_predictions": [
    {
      "violation_name": "image-alt",
      "score": 2.5134,
      "high_severity": true
    }
  ]
}
```

Scores are LinearSVC margins, not calibrated probabilities.

---

## Application Usage Notes

* Predictions are best used for triage. Confirm issues with a full scanner ([axe](https://www.deque.com/axe/), [pa11y](https://pa11y.org/)).
* For best accuracy, retrain with `--include-html-files` if you have local HTML files.
* URL inspection requires network access.
* Snippets in reports are heuristic matches, not confirmed violations.

---

## Resources

- 📂 [GitHub Repository](https://github.com/kthudoan/accesstotech)
- 📄 [Full Documentation](https://docs.google.com/document/d/1OCUD5zRZ-SbzOF7GjGtmmDr7n5IG9jLFHCaUi4cgiCE/edit?usp=sharing)
- 🔧 [Web Content Accessibility Guidelines (WCAG)](https://www.w3.org/WAI/WCAG21/quickref/)

---

## License

See LICENSE file for details.
