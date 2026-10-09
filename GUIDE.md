# GROUP 23 — Community Complaint Categorization: Step-by-Step Guide

Follow the parts in order. Each part tells you **what to do**, **what to hand in**, and **what to write** in your report.

**Your files**

| File | What it is |
|---|---|
| `test_records.csv` | The 100-record dataset (use this one in KNIME and Python) |
| `train_model.py` | Python: cleans data, trains Logistic Regression + Naive Bayes, compares them, saves the model |
| `app.py` | Streamlit interface |
| `GUIDE.md` | This guide |

**Generated when you run `train_model.py`:** `complaint_pipeline.joblib` (saved model), `model_comparison.csv` and `model_comparison.png` (comparison table/chart), `confusion_logistic_regression.png` and `confusion_naive_bayes.png`.

---

## 0. Setup (do once)

1. Install Python 3.9+ from python.org (tick **Add to PATH**).
2. Put `test_records.csv`, `train_model.py`, `app.py` in **one folder**.
3. Open a terminal in that folder and run:

```bash
pip install pandas scikit-learn matplotlib joblib streamlit
```

4. Train the model:

```bash
python train_model.py
```

5. Start the interface:

```bash
streamlit run app.py
```

A browser tab opens automatically (usually http://localhost:8501).

> If `app.py` says "Model file not found", you skipped step 4.

---

## Part 1 — Understand the dataset

**Scenario:** a community office wants to read written complaints, assign a category, and suggest which office handles it.

**The 7 columns and their roles**

| Column | Role | Why |
|---|---|---|
| `complaint_id` | **Reference only** | Just a label (CC-001…). Keep it, never feed it to the model. It would let the model "memorize" rows. |
| `complaint_text` | **Input (main signal)** | The words describe the problem. |
| `location` | Input (collected, not used by the model) | Zone A–J. |
| `submission_channel` | Input (collected, not used by the model) | Web form, Mobile app, Phone, Walk-in, Email. |
| `submission_date` | Input (collected, not used by the model) | Date received. |
| `complaint_category` | **Target label** | What we want to predict. |
| `routing_suggestion` | **Generated output** | Office chosen from the category (see table below). |

**Category → office mapping (the routing rule)**

| Category | Routing suggestion |
|---|---|
| Waste and sanitation | Sanitation Office |
| Road and sidewalk repair | Public Works Office |
| Drainage and flooding | Drainage Services |
| Water supply and leaks | Water Utility |
| Street lighting | Electrical Maintenance |

**Class activity answers (copy and adapt)**

- **Reference-only field:** `complaint_id`
- **Analytical inputs:** `complaint_text` (main), `location`, `submission_channel`, `submission_date`
- **Required target label:** `complaint_category`
- **Generated output:** `routing_suggestion` (plus the predicted category)
- **Missing information:** none in this file, but real submissions can have blank text, missing location, or invalid dates. The app handles these (Part 5).
- **Privacy / fairness / safety / accessibility concern (pick one):**
  - *Fairness:* location and channel could become proxies for *who* gets service. A model that learns "Zone J = slow service" would be unfair, so we rely on the complaint text.
  - *Safety:* urgent items (exposed wire, sinkhole, broken water main) should go to a human quickly, not wait on a model.
  - *Accessibility:* people who submit by phone or walk-in need staff to type the complaint correctly.
  - *Privacy:* real complaints may contain names or addresses. This dataset is synthetic.

**Dataset facts:** 100 records, 5 categories × 20 records each (balanced), 10 zones, 5 channels, dates 2026-01-01 to 2026-04-16.

---

## Part 2 — Prepare and check the dataset

Requirement: 30–100 records, 4–8 relevant input features. Cleaning checklist:

| Task | Done by | Result on this data |
|---|---|---|
| Clean text (lowercase, remove punctuation) | `clean_text()` in `train_model.py` | Done |
| Standardize categories (trim spaces, consistent capitalization) | `load_and_prepare()` | Done |
| Remove duplicates | `drop_duplicates(subset="complaint_text")` | 0 found |
| Check blank descriptions | `blanks` check | 0 found |
| Preserve reference IDs | `complaint_id` is kept in the table, excluded from features | Done |

Running `python train_model.py` prints these counts. Take a screenshot of that output for your report.

---

## Part 3 — Build the KNIME workflow

KNIME is drag-and-drop (download KNIME Analytics Platform from knime.com). Build this chain:

```
CSV Reader → String Manipulation → (Strings to Document) → TF-IDF → Partitioning → Learners → Predictors → Scorer
```

**Step by step**

1. **CSV Reader** — drag it in, double-click, choose `test_records.csv`. Execute (F7). Right-click → *File Table* to check 100 rows.
2. **String Manipulation** — clean the text. Add a new column `clean_text` with the expression:
   `lowerCase(regexReplace($complaint_text$, "[^a-zA-Z ]", ""))`
3. **Document creation** (needed by the text nodes). Install *KNIME Textprocessing* extension (File → Install KNIME Extensions → search "Textprocessing"). Use **Strings to Document**: Full text = `clean_text`, Title = `complaint_id`, Category = `complaint_category`.
4. **Preprocessing (optional but helpful)** — *Stop Word Filter* (English), then *Punctuation Erasure*.
5. **Bag of Words Creator** → **Document Vector** (this creates the TF-IDF-style numeric table; tick *Use TF-IDF / relative frequency* if offered. If your KNIME version has a dedicated **TF-IDF** node, use that instead.) Tick "as collection" off so the vectors become numeric columns.
6. **Category to Class** (or use *Document Data Extractor* / *Joiner*) so the table has the numeric vectors **and** the `complaint_category` column.
7. **Partitioning** — Relative 80%, **Stratified sampling** on `complaint_category`, fixed random seed (e.g. 42). Top port = training, bottom port = testing.
8. **Learners** — add two learners from the *same* training port:
   - **Logistic Regression Learner** (primary) — target column `complaint_category`.
   - **Naive Bayes Learner** (comparison).
9. **Predictors** — *Logistic Regression Predictor* and *Naive Bayes Predictor*. Connect the learner model to the first port and the test data to the second port.
10. **Scorer** — one *Scorer* after each predictor. Compare column `complaint_category` vs. the prediction column.
11. **Evaluation** — right-click each Scorer → *Confusion Matrix* and *Accuracy Statistics* (gives precision, recall, F1 per class and accuracy). Screenshot both.

**Tips if you get stuck**
- Red node = not configured; yellow = ready to execute; green = executed.
- If the Predictor complains about missing columns, the test data went through a different preprocessing branch. Make both partitions come from the same vector table.
- The Scorer's two columns must be the true label and the prediction.
- Save the workflow (*File → Export KNIME Workflow*) as a `.knwf` file. That is a required output.

---

## Part 4 — Recreate the process in Python

All of this is already coded in `train_model.py`. What it does, in plain steps:

1. **Load** the CSV with `pandas.read_csv`.
2. **Prepare** — standardize categories, check blanks and duplicates, clean text.
3. **Build X and y** — `X` = cleaned complaint text, `y` = `complaint_category`.
4. **Split** — 80% train / 20% test, stratified (each category is represented equally), `random_state=42`.
5. **Train two models**, each inside a scikit-learn `Pipeline` (TF-IDF → classifier):
   - `TfidfVectorizer` + `LogisticRegression` (**primary**)
   - `TfidfVectorizer` + `MultinomialNB` (**comparison**)
6. **Evaluate** — accuracy, precision, recall, F1 (weighted averages), confusion matrix, and 5-fold cross-validation accuracy.
7. **Save** the chosen pipeline with `joblib.dump` to `complaint_pipeline.joblib`.

The required imports are in the file:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

**Why the model uses text only (a design decision to explain in your report):** location, channel and date follow a repeating pattern in this dataset (Zone A to J, in the same order, inside every category). They have no real link to the category. In testing, including them did not give a dependable improvement, and the dataset notes warn against using them as a proxy for who receives service. The app still collects them, as the assignment requires, and stores them with the record, but the prediction uses the words of the complaint.

**Results from my run** (your numbers should match because the seed is fixed; if your scikit-learn version differs, tiny changes are possible):

| Model | Test accuracy | Precision | Recall | F1 | 5-fold CV accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.35 | 0.242 | 0.35 | 0.277 | 0.43 |
| Naive Bayes | 0.20 | 0.140 | 0.20 | 0.161 | 0.42 |

> **Read this honestly.** Accuracy is modest. The test set is only 20 records (4 per category), so a single wrong answer moves accuracy by 5 points. The 5-fold CV number (0.43 and 0.42) is the more trustworthy one, and it says the two models are close. Do not claim either model is "good"; claim what the numbers show (see Part 6).

**Why accuracy is limited (use this in your limitations paragraph):**
- Only 100 short records, 80 used for training.
- Every complaint uses different wording, so a word seen in the test set is often unseen in training.
- Categories share words: "water" appears in Drainage and Water supply, "market", "sidewalk", "school" appear in almost every category, and "flooding" appears in a Road repair complaint.

---

## Part 5 — Create the Streamlit interface

`app.py` is ready. Run `streamlit run app.py`.

**Inputs the interface collects (as required):** complaint text, location, submission channel, date.

**Validation (incomplete or invalid values)**

| Problem | What the user sees |
|---|---|
| Text empty or spaces only | "Complaint text is empty." |
| Fewer than 3 words | "Complaint text is too short (at least 3 words)." |
| Missing location | "Location is missing." |
| Invalid channel | "Please choose a valid submission channel." |
| Date missing or in the future | "Date is missing." / "Date cannot be in the future." |

**Output:** predicted **category**, **routing suggestion**, **confidence**, a bar chart of the probability for each category, and a short **explanation**.

**Uncertain results:** if the top confidence is below 30% or the top two categories are within 10 points of each other, the app shows a warning and asks for manual review instead of a confident route.

**Bonus tab:** upload a CSV (columns `complaint_text`, `location`, `submission_channel`) to categorize many complaints and download `predictions.csv`.

**Try these in the app and screenshot them (this is Part 6 evidence):**

1. `The streetlight on my corner is broken and dark at night.` → Street lighting (about 96%).
2. Leave the box empty → error message.
3. `Leak now` → "too short" message.
4. `I would like to ask about my business permit.` → uncertain warning.

---

## Part 6 — Test, explain, and reflect

**Test cases (put them in a table in your report)**

| Case type | Example input | Expected behavior | What happened |
|---|---|---|---|
| Normal | "The streetlight on my corner is broken and dark at night." | Street lighting → Electrical Maintenance | Correct, confidence about 96% |
| Missing | Empty text | Blocked with a clear message | Blocked |
| Boundary | "Leak" (one word) | Rejected in the app (too short). In `train_model.py` the model gives Water supply and leaks with only 32% | Handled by validation |
| Uncertain | "Water and garbage are all over the road after the flood." | Mixed topics, should be flagged | Model guessed Water supply (35%), close to other classes, so the app asks for review |
| Unrelated | "I would like to ask about my business permit." | No suitable category | Low confidence (23%), flagged for manual review |

**Comparison table or chart:** use `model_comparison.csv` (table) or `model_comparison.png` (bar chart). Add the confusion-matrix images as supporting evidence.

### Critical-thinking questions (suggested answers)

**1. Which input features should have the strongest influence on the assigned category?**
The **words in the complaint text**. Words like "garbage", "pothole", "drain", "pipe", "streetlight" directly describe the problem. Location, channel and date should have little or no influence, because they say where or how the complaint arrived, not what it is about. Using them could create unfair patterns.

**2. What should the system do when the available evidence is incomplete or contradictory?**
Do not guess confidently. The system should (a) reject empty or too-short text and ask for more detail, (b) show low confidence and flag the record for **manual review** when the top categories are close, and (c) never auto-route urgent or safety-related complaints without a human check. Example of contradictory evidence: "Flooding damaged the road". It touches both Drainage and Road repair.

**3. What could happen if the system produces an incorrect result?**
A complaint goes to the wrong office, so it is delayed, bounced around, or ignored. Residents lose trust. For safety issues (broken water main, exposed wire, sinkhole) a delay can cause harm. If the errors are not spread evenly (for example one area's wording is misclassified more), service could become unfair. This is why uncertain cases need human review.

### Interpretation (starter paragraph — edit with your own numbers)

> We built a text classifier that assigns a community complaint to one of five categories and suggests the responsible office. Logistic Regression with TF-IDF reached 43% five-fold cross-validated accuracy and Naive Bayes reached 42%, so the two performed about the same; on the single 80/20 split Logistic Regression scored 35% and Naive Bayes 20%, but that split has only 20 test records, so the difference is not reliable. The results are better than guessing (20% for five classes) but too low for automatic routing. The main limits are the very small dataset, varied wording with few repeated words, and categories that share vocabulary such as water, market and sidewalk. The system therefore flags low-confidence cases for manual review, relies on complaint text rather than location or channel to avoid unfair routing, and should be retrained with more real, de-identified complaints before any real use. All data here is synthetic.

---

## Final checklist — Required outputs

- [ ] 30–100 record dataset → `test_records.csv`
- [ ] Data dictionary → section in `test_records.md` / Part 1 table
- [ ] KNIME workflow → exported `.knwf` file + screenshot
- [ ] Python implementation → `train_model.py`
- [ ] Streamlit interface → `app.py` + screenshots
- [ ] Comparison table or chart → `model_comparison.csv` / `.png`
- [ ] Short interpretation and limitations → Part 6 paragraph

## Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run the `pip install` line from Part 0 again |
| `FileNotFoundError: test_records.csv` | Run the script from the folder that contains the CSV |
| App says "Model file not found" | Run `python train_model.py` first |
| `streamlit` not recognized | Use `python -m streamlit run app.py` |
| Different accuracy than this guide | Different scikit-learn version, or you changed the seed or the data |
