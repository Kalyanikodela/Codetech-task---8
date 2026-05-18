# Codetech-task---8
To deliver this as an actual notebook file (.ipynb), I have written a Python script below that automatically generates the entire Jupyter Notebook file for you.How to get your .ipynb file:Copy the code block below and run it in any Python environment on your computer.It will instantly create a file named sentiment_analysis_notebook.ipynb in your current folder.Open that file directly in Jupyter Lab, Jupyter Notebook, or upload it to Google Colab.pythonimport json

# Define the complete structural content for the Jupyter Notebook
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sentiment Analysis Showcase Notebook\n",
    "### Deliverable: End-to-End NLP Pipeline\n",
    "\n",
    "This notebook demonstrates a complete Natural Language Processing (NLP) workflow: \n",
    "1. **Data Preprocessing**: Text cleaning, tokenization, and stop-word removal.\n",
    "2. **Feature Engineering**: Transforming text to numerical matrices using TF-IDF.\n",
    "3. **Model Implementation**: Training a machine learning classifier.\n",
    "4. **Insights & Evaluation**: Evaluation metrics and visual diagnostic tools."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment Setup & Dependencies\n",
    "We ingest required libraries for data manipulation, text processing, machine learning, and visualization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
    "\n",
    "# Download NLTK resources\n",
    "nltk.download('stopwords')\n",
    "print(\"Setup Complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset Simulation\n",
    "We instantiate a robust synthetic dataset representing typical text data variations (positive, negative, and neutral) to keep the pipeline reproducible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "raw_data = {\n",
    "    'review_text': [\n",
    "        \"I absolutely love this product! It works perfectly and saved me hours.\",\n",
    "        \"Worst purchase ever. The item arrived broken and customer service was rude.\",\n",
    "        \"Great value for the money. Highly recommend to everyone.\",\n",
    "        \"Terrible quality. It stopped working after just two days.\",\n",
    "        \"It is okay, does the job but nothing special or amazing.\",\n",
    "        \"Excellent build quality and fast shipping. Very happy!\",\n",
    "        \"Disappointed. The product looks completely different from the pictures.\",\n",
    "        \"Waste of money. Avoid this seller at all costs.\",\n",
    "        \"Super functional, easy to use, and beautiful design.\",\n",
    "        \"Horrible experience. The app crashes every time I open it.\"\n",
    "    ],\n",
    "    'sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Neutral', \n",
    "                  'Positive', 'Negative', 'Negative', 'Positive', 'Negative']\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(raw_data)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "Raw text contains structural noise. This stage strips out HTML, strips punctuation, downcases text, and isolates contextually high-value terms by dropping stop words."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    # Remove potential HTML tags\n",
    "    text = re.sub(r'<[^>]*>', '', text)\n",
    "    # Isolate alphabetic characters\n",
    "    text = re.sub(r'[^a-zA-Z\\s]', '', text)\n",
    "    # Case normalization\n",
    "    text = text.lower()\n",
    "    # Tokenize and filter standard English stopwords\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    cleaned_tokens = [word for word in text.split() if word not in stop_words]\n",
    "    \n",
    "    return ' '.join(cleaned_tokens)\n",
    "\n",
    "# Apply textual preprocessing matrix\n",
    "df['cleaned_text'] = df['review_text'].apply(clean_text)\n",
    "df[['review_text', 'cleaned_text']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Implementation\n",
    "We separate features from targets, partition the data for strict validation, execute TF-IDF feature mapping, and fit a Linear Logistic Regression variant."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train/Test Partitioning\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    df['cleaned_text'], df['sentiment'], test_size=0.3, random_state=42\n",
    ")\n",
    "\n",
    "# Token Vectorization via TF-IDF\n",
    "tfidf = TfidfVectorizer(max_features=5000)\n",
    "X_train_tfidf = tfidf.fit_transform(X_train)\n",
    "X_test_tfidf = tfidf.transform(X_test)\n",
    "\n",
    "# Fit Statistical Classifier\n",
    "model = LogisticRegression()\n",
    "model.fit(X_train_tfidf, y_train)\n",
    "\n",
    "# Generate Predictions\n",
    "y_pred = model.predict(X_test_tfidf)\n",
    "print(\"Model training complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Insights & Evaluation Metrics\n",
    "We print precision metrics and graph out a matrix evaluation to expose model error rates visually."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Overall Classification Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\\n\")\n",
    "print(\"Detailed Metric Reporting:\")\n",
    "print(classification_report(y_test, y_pred, zero_division=0))\n",
    "\n",
    "# Construct Error Matrix Graphic\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(5, 4))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', \n",
    "            xticklabels=model.classes_, yticklabels=model.classes_)\n",
    "plt.title('Pipeline Confusion Matrix Performance')\n",
    "plt.ylabel('Ground Truth')\n",
    "plt.xlabel('Model Predictions')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Live Interface Inference Simulation\n",
    "Pass any raw string data directly through the pipeline mapping below to get real-time evaluations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def analyze_sentiment(custom_string):\n",
    "    processed = clean_text(custom_string)\n",
    "    vector = tfidf.transform([processed])\n",
    "    return model.predict(vector)[0]\n",
    "\n",
    "# Sample Live String Run\n",
    "sample_input = \"The UI looks bad, but the functional backend architecture is fantastic.\"\n",
    "print(f\"Input Text: '{sample_input}'\")\n",
    "print(f\"Evaluated Class Outcome: {analyze_sentiment(sample_input)}\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Write out the JSON structure as an operational .ipynb file
file_name = "sentiment_analysis_notebook.ipynb"
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print(f"Success! Notebook file generated perfectly as: '{file_name}'")
