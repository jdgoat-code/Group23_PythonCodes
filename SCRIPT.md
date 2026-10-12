# Video Script: Community Complaint Categorization

**Estimated length:** 5–6 minutes  
**Format:** Screen recording with voice-over  
**Suggested screens:** `test_records.csv` → `KNIME.knwf` → `train_model.py` and results → Streamlit app

## 1. Introduction (0:00–0:35)

**On screen:** Show the project folder, then a few rows of `test_records.csv`.

**Narration:**

> Hello! In this video, we'll explain our community complaint categorization system in three parts: the KNIME workflow, the Python training process, and the Streamlit interface. The goal is to read a written complaint, predict its category, and suggest where it should go. Our dataset has 100 complaints, evenly divided among five categories: waste and sanitation, road and sidewalk repair, drainage and flooding, water supply and leaks, and street lighting. Each record also includes an urgency level, location, submission channel, and date.

## 2. KNIME workflow (0:35–2:00)

**On screen:** Open `KNIME.knwf`. Pan from the CSV Reader and text preparation nodes toward the learner, predictor, and scorer branches. Zoom in on each group as it is mentioned.

**Narration:**

> KNIME gives us a visual way to build and inspect the classification process. We begin with a CSV Reader, which loads the complaint records. A Column Filter keeps the fields needed for this part of the workflow. String Manipulation standardizes the complaint text, and Strings to Document prepares it for KNIME's text-processing nodes.
>
> Next, Punctuation Erasure and Stop Word Filter remove punctuation and common words. Bag of Words Creator breaks the documents into terms. The Document Vector nodes turn those terms into numeric columns that a machine-learning model can use. The workflow also includes term-frequency and inverse-document-frequency nodes for weighting words. A Joiner reconnects the processed text with the complaint's category, which is the correct answer the models learn from.
>
> The table is then partitioned into training and testing portions. Separate branches train Naive Bayes and Logistic Regression. Each predictor applies its trained model to test records, and a Scorer compares the predicted category with the actual category. This lets us inspect correct predictions and mistakes, rather than judging the model from a few examples. The exported workflow contains multiple experiment branches; the key pattern is always preparation, vectorization, partitioning, training, prediction, and scoring.

## 3. Python training (2:00–3:45)

**On screen:** Show `train_model.py`, focusing on `load_and_prepare`, `make_pipeline`, and `main`. Then show `model_comparison.csv`, `model_comparison.png`, and one confusion matrix.

**Narration:**

> We implement the same core idea in Python with `train_model.py`. The script reads `test_records.csv`, checks that required columns are present, removes blank or duplicate descriptions, standardizes labels, and cleans the text. The complaint ID stays as a reference. Existing routing suggestions and resolution dates in the dataset are also excluded from model training, because they are outputs rather than evidence for a new prediction.
>
> For the category model, the inputs are cleaned complaint text and urgency level. TF-IDF converts the text into weighted numeric features, while one-hot encoding converts urgency into numeric indicators. We compare two classifiers: Logistic Regression and Multinomial Naive Bayes. An 80/20 stratified split gives both models the same training and test setup, and five-fold cross-validation provides an additional check across different subsets of the data.
>
> In the saved comparison results, Logistic Regression reached 55 percent accuracy on the 20-record test set, while Naive Bayes reached 40 percent. Their five-fold cross-validation accuracies were 52 percent and 45 percent. These scores show that Logistic Regression performed better in this run, but the dataset is small and the accuracy is too limited to treat every prediction as certain. The confusion matrices show which categories the models confuse. Finally, the script trains the selected Logistic Regression pipeline on all cleaned records and saves it as `complaint_pipeline.joblib` for the app.

## 4. Streamlit app (3:45–5:20)

**On screen:** Run `streamlit run app.py` or `run_app.bat`. In the **Single complaint** tab, enter a complaint of at least three words, choose urgency, location, channel, and date, then click **Categorize**. Show the output chart. Briefly show the **Upload CSV** and **Channel analytics** tabs.

**Narration:**

> Streamlit turns the trained pipeline into an interface for users. In the Single complaint tab, we enter a description and select its urgency, location, submission channel, and date. The app checks the inputs first; for example, empty text and descriptions shorter than three words produce clear error messages.
>
> After we click Categorize, the saved model estimates a category and displays its confidence and the probability for each of the five categories. The category prediction uses the complaint text and urgency. The route is then built with separate rules: the category chooses the office, the location chooses Branch 1 or Branch 2, and high urgency selects the Rapid Response queue. The estimated resolution date is calculated from the submission date: one calendar day for high urgency, three for medium, and seven for low. The submission channel is recorded for analytics; it does not influence the category prediction.
>
> If confidence is below 30 percent, or the top two category probabilities are less than 10 percentage points apart, the app flags the result for manual review instead of presenting a route as certain. For multiple complaints, the Upload CSV tab validates the records, displays predictions, and lets us download a results file. The Channel analytics tab charts the submission channels from the latest valid upload.

## 5. Closing (5:20–5:45)

**On screen:** Show the KNIME workflow, Python comparison chart, and Streamlit result side by side or in quick succession.

**Narration:**

> Together, these three parts show the full path from data to a usable interface. KNIME makes the text-processing and model comparison visible, Python trains and saves the classifier, and Streamlit applies it to new complaints. Because the current model was trained on only 100 examples and still makes errors, its suggestions should support a human reviewer. More representative, carefully checked complaint data would be needed before relying on it for real service routing.

---

**Recording note:** The older examples and accuracy figures in `GUIDE.md` describe a previous version of the project. Use the current `model_comparison.csv` figures shown above, and rerun `python train_model.py` if the dataset changes before recording.
