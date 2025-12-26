import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title="Text Emotion Classifier", layout="centered")


# Download NLTK stopwords if missing
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Paths (expect these files to be in same folder as app)
MODEL_PATH = 'emotion_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'


def clean(text: str) -> str:
    stem = PorterStemmer()
    s = set(stopwords.words('english'))
    punc = string.punctuation

    text = text.lower()
    tokens = text.split()
    filtered = [stem.stem(t) for t in tokens if t not in s]
    joined = " ".join(filtered)
    out = ''.join(ch for ch in joined if ch not in punc)
    return out


def main():
    st.title("Text Emotion Classifier")
    st.write("Enter a sentence and the model will predict the emotion.")

    # Check files (model and tokenizer are required; label encoder optional)
    missing = []
    import os
    for p in (MODEL_PATH, TOKENIZER_PATH):
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        st.error(f"Missing required files: {', '.join(missing)}")
        st.info("Place `emotion_model.h5` and `tokenizer.pkl` in this folder. See the project's notebook for saving them from Colab.")
        return

    model = load_model(MODEL_PATH)
    t = load_pickle(TOKENIZER_PATH)

    # Try loading LabelEncoder; if absent, fall back to known class order from training notebook
    if os.path.exists(LABEL_ENCODER_PATH):
        l = load_pickle(LABEL_ENCODER_PATH)
        labels_fallback = None
    else:
        l = None
        labels_fallback = [
            'anger', 'empty', 'enthusiasm', 'fun', 'happiness',
            'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise'
        ]

    user_input = st.text_area("Your text:", height=120)

    if st.button("Predict"):
        if not user_input or user_input.strip() == "":
            st.warning("Please enter text to classify.")
        else:
            cleaned = clean(user_input)
            seq = t.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, padding='post', maxlen=50)
            preds = model.predict(padded)
            idx = int(np.argmax(preds, axis=1)[0])
            # Resolve predicted label: prefer LabelEncoder when present,
            # otherwise use the notebook fallback list.
            if l is not None:
                try:
                    label = l.inverse_transform([idx])[0]
                except Exception:
                    classes_attr = getattr(l, 'classes_', None)
                    if classes_attr is not None and len(classes_attr) > idx:
                        label = classes_attr[idx]
                    else:
                        # last-resort fallback
                        label = labels_fallback[idx] if labels_fallback and 0 <= idx < len(labels_fallback) else str(idx)
            else:
                label = labels_fallback[idx] if labels_fallback and 0 <= idx < len(labels_fallback) else str(idx)
            st.success(f"Predicted emotion: {label}")
            st.write("Confidence:", float(np.max(preds)))


if __name__ == '__main__':
    main()
