# 📩 SMS Spam Classifier  

A Machine Learning web application that classifies text messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques.  
The app takes an SMS as input, processes it, and predicts whether it is spam.  

🔗 **Live App:** [SMS Spam Classifier](https://mw3kk2nefjrw3c79sbg4hy.streamlit.app/)  

---

## 📂 Project Files  

- **`LICENSE`** – License for the project (MIT).  
- **`README.md`** – Project documentation.  
- **`SMS_spam_classifier.ipynb`** – Jupyter notebook for data preprocessing, feature extraction, model training, and evaluation.  
- **`app.py`** – Streamlit web app script for deployment.  
- **`model.pkl`** – Trained machine learning model for classification.  
- **`requirements.txt`** – Python dependencies.  
- **`spam.csv`** – Dataset containing SMS messages labeled as spam or ham.  
- **`vectorizer.pkl`** – Saved TF-IDF vectorizer for text transformation.  

---

## ⚙️ Features  

- **Data Preprocessing:**  
  - Cleans text messages (removes punctuation, converts to lowercase, removes stopwords).  
  - Tokenization and stemming.  

- **Feature Extraction:**  
  - Uses **TF-IDF Vectorization** to convert text into numerical features.  

- **Model:**  
  - Trained using **Naive Bayes** (MultinomialNB) for spam detection.  

- **Web App:**  
  - Simple interface to enter an SMS and check prediction in real-time.  
  - Displays clear spam/ham classification output.  

---

## 📊 Dataset  

- **Source:** Kaggle SMS Spam Collection Dataset
- **Size:** ~5,000 SMS messages in total.  
- **Classes:**  
  - **ham** – Legitimate messages.  
  - **spam** – Unwanted, advertisement, or fraudulent messages.  

---

## 📦 Tech Stack  

- **Python** – Core programming language for development  
- **Pandas**, **NumPy** – Data handling and preprocessing  
- **Scikit-learn** – Machine Learning model building and evaluation  
- **NLTK** – Natural Language Processing for text cleaning and tokenization  
- **Streamlit** – Web app framework for UI and deployment  
- **Pickle** – Model & vectorizer serialization for saving/loading  

---

## 🚀 How to Run Locally  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier

   
2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

