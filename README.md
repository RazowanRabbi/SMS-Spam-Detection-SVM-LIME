# SMS Spam Detection using Calibrated SVM and LIME Explainability

## Overview

This project implements an SMS Spam Detection system using Machine Learning and Explainable Artificial Intelligence (XAI).

The system classifies SMS messages into two categories:

- Spam
- Ham (Normal Message)

The project uses TF-IDF feature extraction with a calibrated Linear Support Vector Machine (SVM) classifier. To improve transparency and interpretability, LIME (Local Interpretable Model-Agnostic Explanations) is integrated to explain why a particular SMS message is classified as spam or ham.

The objective of this project is not only to achieve high classification performance but also to make machine learning decisions understandable to users.

---

# Project Features

The system includes:

- SMS text preprocessing
- TF-IDF based feature extraction
- Class imbalance handling
- Calibrated Linear SVM classification
- Hyperparameter optimization using GridSearchCV
- Performance evaluation using multiple metrics
- Probability-based prediction confidence
- Explainable AI using LIME
- Word-level explanation of predictions

---

# Machine Learning Pipeline

The complete workflow of the system:

```
SMS Message
      |
      ↓
Text Preprocessing
      |
      ↓
TF-IDF Feature Extraction
      |
      ↓
Class Balanced Linear SVM
      |
      ↓
Probability Calibration
      |
      ↓
Spam/Ham Prediction
      |
      ↓
LIME Explanation
      |
      ↓
Important Word Identification
```

---

# Dataset

The project uses the SMS Spam Collection Dataset.

The dataset contains SMS messages labeled as:

- ham: Normal messages
- spam: Unwanted promotional or fraudulent messages

The dataset contains thousands of SMS messages collected for spam detection research.

Since the dataset contains an imbalance between spam and ham messages, class balancing was applied during model training.

---

# Data Preprocessing

The following preprocessing techniques were applied:

- Lowercase conversion
- URL removal
- Email removal
- Number handling
- Punctuation removal
- Stopword removal
- Lemmatization

Example:

Before preprocessing:

```
Congratulations!!! You won $1000. Click now.
```

After preprocessing:

```
congratulation won click
```

---

# Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text messages into numerical features that can be understood by machine learning algorithms.

The optimized TF-IDF configuration includes:

- Maximum features: 10000
- Unigram and bigram features
- Sublinear TF scaling

Using bigrams allows the model to learn important spam phrases such as:

```
free prize
claim now
cash reward
winner offer
```

---

# Machine Learning Model

The final classifier used in this project is:

## Calibrated Linear Support Vector Machine (SVM)

The model consists of:

- LinearSVC classifier
- Class balancing
- Probability calibration using CalibratedClassifierCV

Hyperparameter tuning was performed using GridSearchCV.

Best parameter:

```
C = 100
```

---

# Model Performance

The final model achieved the following results:

| Metric | Score |
|--------|-------|
| Accuracy | 98.16% |
| Precision | 95.16% |
| Recall | 90.08% |
| F1-score | 92.55% |

## Performance Interpretation

### Accuracy

The model correctly classifies approximately 98% of SMS messages.

### Precision

When the model predicts a message as spam, approximately 95% of those predictions are correct.

### Recall

The model successfully detects approximately 90% of actual spam messages.

### F1-score

The F1-score provides a balance between precision and recall, showing strong overall spam detection performance.

---

# Explainable AI using LIME

To improve transparency, LIME (Local Interpretable Model-Agnostic Explanations) was integrated into the system.

LIME explains individual predictions by identifying the words that contribute most toward a spam or ham classification.

Example:

Input:

```
Congratulations! You have won a free iPhone. Click here now.
```

Prediction:

```
Spam
```

LIME explanation:

Important spam-related words:

```
free
click
iPhone
prize
```

This allows users to understand why the model made a specific decision.

---

# Example Predictions

## Spam Example

Message:

```
URGENT! You have won a cash prize. Claim now.
```

Prediction:

```
Spam
```

Important words:

```
urgent
cash
prize
claim
```

---

## Ham Example

Message:

```
Hey, are you coming to class tomorrow?
```

Prediction:

```
Ham
```

Important words:

```
class
tomorrow
coming
```

---

# Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- WordCloud
- LIME

---

# Project Structure

```
SMS-Spam-Detection-SVM-LIME

│
├── SMS_Spam_Detection_SVM_LIME.ipynb
│
├── spam.csv
│
├── README.md
│
├── requirements.txt
│
└── images
    │
    ├── lime_spam_example.png
    ├── lime_ham_example.png
    └── confusion_matrix.png
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/RazowanRabbi/SMS-Spam-Detection-SVM-LIME.git
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Open the notebook:

```
SMS_Spam_Detection_SVM_LIME.ipynb
```

Run all cells in order.

The notebook performs:

1. Dataset loading
2. Data preprocessing
3. TF-IDF feature extraction
4. SVM model training
5. Hyperparameter optimization
6. Model evaluation
7. SMS prediction
8. LIME explanation generation

---

# Future Improvements

Possible future improvements include:

- Comparing multiple machine learning algorithms
- Deploying the model as a web application
- Testing on larger external spam datasets
- Using deep learning approaches such as LSTM or BERT
- Real-time SMS spam detection system

---

# Conclusion

This project demonstrates a complete SMS Spam Detection system using machine learning and explainable AI.

The calibrated SVM model achieves strong classification performance with an accuracy of 98.16%.

By integrating LIME, the system not only predicts whether an SMS is spam or ham but also explains the important words responsible for the prediction.

This combination of performance and interpretability makes the system more reliable, transparent, and suitable for real-world applications.
