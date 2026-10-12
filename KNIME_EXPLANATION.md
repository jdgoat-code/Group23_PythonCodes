CSV READER - Reads the CSV
Column Filter - Only retains the factors needed for the model learning - The input (complaint_text) and the output (complaint_category)
String Manipulation - turns complaint_text to lowercase for processing
Strings to Document - Converts complaint_text to document for further processing (string manipulation and tokenizing)
Punctuation Erasure - removes punctuations from complaint_text
Stop word filter - Erases common words from complaint_text such as articles, prepositions, etc.
Bag of Words Creator - Separates complaint_text's words into rows of category-word pairs
Document Vector - Creates a table wherein the columns consist of every word in the test records, then compiles the category-word pairs from the last node into rows of each record, mapping the occurence of the words from their complaint_text into the word columns.
Joiner - Joins the mapped rows with the output (complaint_category) that was lost from mapping.
Column Filter - Removes columns that are unnecessary for the model and human observation (documents and formatted text, which has already been tokenized)
Table Partitioner - Divides the records for either learning or testing

Learners (Naive Bayes and Logistic Regression) - Using the tokens and the desired output, learns how patterns affect the outcome.
Predictors - Using the learners, predicts the outcome of the testing records
Scorer - Tracks the accuracy of the predictor.