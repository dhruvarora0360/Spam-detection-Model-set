# Spam Email Detection Model

This project is a machine learning-based Spam Email Classification System. It uses Natural Language Processing (NLP) techniques and the Multinomial Naive Bayes algorithm to classify emails as either **Spam** or **Ham** (non-spam).

## 📊 Dataset

The dataset consists of email messages labeled as spam or ham.
- **Total Messages:** 33,716
- **Columns:**
  - `Message ID`: Unique identifier for each message.
  - `Subject`: The subject line of the email.
  - `Message`: The body content of the email.
  - `Spam/Ham`: The label ('spam' or 'ham').
  - `Date`: The date the email was sent.

The dataset is preprocessed to combine the `Subject` and `Message` columns into a single `text` column for analysis.

## 🛠️ Technologies & Libraries

The project is built using Python and the following libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Seaborn & Matplotlib**: For data visualization.
- **Scikit-Learn**: For machine learning models and evaluation metrics.

## ⚙️ Workflow

1.  **Data Loading**: Reading the CSV file.
2.  **Data Cleaning**: Handling missing values and dropping unnecessary columns.
3.  **Preprocessing**:
    - Combining `Subject` and `Message`.
    - Encoding labels (`ham` -> 0, `spam` -> 1).
4.  **Feature Extraction**: Converting text data into numerical vectors using **TF-IDF Vectorizer**.
5.  **Model Training**: Training a **Multinomial Naive Bayes** model.
6.  **Evaluation**: Checking accuracy, confusion matrix, and classification report.

## 📈 Model Performance

The model achieves high accuracy on the test set:

- **Accuracy**: ~99%
- **Precision**: 0.99
- **Recall**: 0.99
- **F1-Score**: 0.99

## 🚀 Usage

To use the model for prediction, you can use the trained model and vectorizer. Here is an example function:

```python
def predict_mail(text):
    vector = tfidf.transform([text])
    output = model.predict(vector)[0]
    return "SPAM" if output == 1 else "HAM"

# Example
print(predict_mail("Congratulations! You have won $1000."))
# Output: SPAM
```

## 📦 Dependencies

Install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
