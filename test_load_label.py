
import pickle
import tensorflow as tf
import numpy as np
import string
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk


def ensure_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")


def clean(text):
    stem = PorterStemmer()
    s = set(stopwords.words("english"))
    tokens = text.lower().split()
    filtered = [stem.stem(t) for t in tokens if t not in s]
    joined = " ".join(filtered)
    return "".join(ch for ch in joined if ch not in string.punctuation)


def idx_to_label(i):
    # fallback label order used in the notebook
    labels_fallback = [
        'anger','empty','enthusiasm','fun','happiness',
        'hate','love','neutral','relief','sadness','surprise'
    ]
    return labels_fallback[i] if 0 <= i < len(labels_fallback) else str(i)


def main():
    ensure_stopwords()

    if not os.path.exists('tokenizer.pkl'):
        print('tokenizer.pkl not found in current folder.')
        return
    if not os.path.exists('emotion_model.h5'):
        print('emotion_model.h5 not found in current folder.')
        return

    t = pickle.load(open('tokenizer.pkl', 'rb'))
    model = tf.keras.models.load_model('emotion_model.h5')

    # try label encoder
    l = None
    if os.path.exists('label_encoder.pkl'):
        try:
            l = pickle.load(open('label_encoder.pkl', 'rb'))
        except Exception:
            l = None

    sample = "I am feeling very happy today!"
    cleaned = clean(sample)
    seq = t.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, padding='post', maxlen=50)
    preds = model.predict(padded)
    idx = int(np.argmax(preds, axis=1)[0])

    if l is not None:
        try:
            label = l.inverse_transform([idx])[0]
        except Exception:
            label = idx_to_label(idx)
    else:
        label = idx_to_label(idx)

    print('Sample:', sample)
    print('Predicted label:', label)
    print('Confidence:', float(np.max(preds)))


if __name__ == '__main__':
    main()
