# 😊 Human Emotion Detection using Bi-LSTM (NLP)

This project implements a **Human Emotion Detection system** using **Natural Language Processing (NLP)** and a **Bidirectional LSTM (Bi-LSTM)** neural network.  
The model classifies text into **10 different emotion categories**, capturing contextual information from both past and future words in a sentence.

---

## 🚀 Project Overview

Emotion detection from text is an important NLP task with applications in:
- Sentiment & emotion analysis
- Mental health monitoring
- Chatbots & virtual assistants
- Social media analysis
- Customer feedback analysis

This project uses a **Bidirectional LSTM**, which improves performance over traditional RNNs by learning context in **both forward and backward directions**.

---

## 🎯 Emotion Classes (10)

The model predicts one of the following emotions:

1. Joy  
2. Sadness  
3. Anger  
4. Fear  
5. Surprise  
6. Love  
7. Disgust  
8. Guilt  
9. Shame  
10. Neutral  

*(Emotion labels may vary depending on dataset used)*

---

## 🧠 Model Architecture

- Text preprocessing (cleaning & tokenization)
- Word Embedding layer
- **Bidirectional LSTM**
- Dropout for regularization
- Dense output layer with Softmax activation

---

## 🏗️ Tech Stack

- **Language**: Python  
- **NLP**: NLTK / Tokenizer  
- **Deep Learning**: TensorFlow / Keras  
- **Model**: Bidirectional LSTM (Bi-LSTM)  
- **Data Handling**: NumPy, Pandas  
- **Visualization**: Matplotlib / Seaborn  
- **Notebook**: Jupyter Notebook  

---

## 📂 Project Structure
├── nlp_project.ipynb # Main notebook (training & evaluation)
├── dataset/ # Emotion dataset
├── README.md # Project documentation
├── requirements.txt # Python dependencies


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/emotion-detection-bilstm.git
cd emotion-detection-bilstm

2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

3️⃣ Install dependencies
pip install -r requirements.txt


Run:

nlp_project.ipynb


The notebook includes:

Data preprocessing

Model building

Training

Evaluation

Predictions

📊 Model Evaluation

Accuracy

Loss curves

Confusion matrix

Classification report (Precision, Recall, F1-score)

🔍 Key Highlights

Captures long-term dependencies in text

Bidirectional context improves emotion understanding

Handles multi-class emotion classification

Easily extendable to more emotion classes or datasets
⚠️ Limitations

Performance depends on dataset quality

May struggle with sarcasm or ambiguous sentences

Requires sufficient labeled data for best results

🛣️ Future Improvements

Use pretrained embeddings (GloVe / Word2Vec / FastText)

Transformer-based models (BERT, RoBERTa)

Real-time emotion detection API

Multilingual emotion classification
