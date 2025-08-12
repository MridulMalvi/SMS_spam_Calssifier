import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data only if not already present
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize stemmer
ps = PorterStemmer()

# Function to clean and preprocess text
def transform_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app interface
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it's **Spam** or **Not Spam**.")

# Input box
input_sms = st.text_area("✏️ Type your message here:")

if st.button('Predict'):
    if input_sms.strip() != "":
        # 1. Preprocess the input
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Show result
        if result == 1:
            st.error("🚫 Spam", icon="🚫")
        else:
            st.success("✅ Not Spam", icon="✅")
    else:
        st.warning("⚠️ Please enter a message before predicting.")
