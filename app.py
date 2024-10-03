import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Initialize the stemmer and download necessary NLTK resources
ps = PorterStemmer()
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)  # Ensure stopwords are downloaded

# Load the vectorizer and model
tf = pickle.load(open('vectorizer.pkl', 'rb'))
mnb1_model = pickle.load(open('mnb1_model.pkl', 'rb'))

# Function to transform input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Set the title and a header for the app
st.title("SMS/Email Spam Classification")
#st.markdown("<h2 style='text-align: center;'>Detect if your message is Spam or Not</h2>", unsafe_allow_html=True)

# Add an image
#st.image("
# Text area for user input
input = st.text_area("Enter or Paste your message here", height=150)

# Add a button to trigger prediction
if st.button('Predict', key='predict_button'):
    # Check for empty input
    if not input.strip():
        st.error("Please enter a valid message.")
    else:
        text_transform = transform_text(input)  # Text preprocessing
        # Vectorization
        text_vector = tf.transform([text_transform])
        
        # Predict
        try:
            result = mnb1_model.predict(text_vector)[0]
            if result == 1:
                st.header('**Result: Spam**')
                st.write("🚫 This message is classified as Spam. Please be cautious!")
            else:
                st.header('**Result: Not Spam**')
                st.write("✅ This message is classified as Not Spam. You're safe!")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Add a sidebar for additional options or information
st.sidebar.header("About this App")
st.sidebar.write("""
    This application uses machine learning to classify SMS and email messages as spam or not spam.
    Simply enter your message in the text area and click on 'Predict' to get the classification.
""")

# Footer with credits or links
st.markdown("---")
st.markdown("<footer style='text-align: center;'>Created with ❤️ by Nuthalapati Azad</footer>", unsafe_allow_html=True)





