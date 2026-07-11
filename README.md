<h1 align="center">🚨 AI-Powered Phishing Detection System 🚨</h1>

![image](https://github.com/user-attachments/assets/cdfb7afc-6eeb-4fdb-9b26-6bcedc247ff4)

## ⚙️ About the Project

An AI-powered phishing detection system that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify emails as **Phishing** or **Safe**.

Built with **Python**, **Streamlit**, **scikit-learn**, and **NLTK**, the system analyzes email content and URL-based features to detect phishing attacks in real time.

[![Phishing Detection](https://img.shields.io/badge/Phishing%20Detection-Active-brightgreen)](https://ai-powered-phishing-detection-system-ples7i6bq2tzkaguiykzzt.streamlit.app/)

---

# 🚀 Features

- 📧 Email Body Analysis using NLP
- 🔗 URL Feature Extraction and Analysis
- 🤖 Machine Learning-based Classification
- 📊 Real-time Prediction with Confidence Score
- 📈 Risk Level Visualization
- 🎨 Interactive Streamlit Interface
- 💾 Train the Model with Custom Datasets

---

# 🧠 Technologies Used

- Python 🐍
- Streamlit 🌐
- scikit-learn 🤖
- pandas 📊
- NumPy 🔢
- NLTK 🧠
- Joblib 💾

---

# 🎯 Machine Learning Pipeline

```
Email
   │
   ▼
Text Cleaning
(Lowercase, Remove Stopwords, Remove Special Characters)
   │
   ▼
CountVectorizer (Bag of Words)
   │
   ▼
URL Feature Extraction
   │
   ▼
Logistic Regression Classifier
   │
   ▼
Prediction
(Phishing / Safe)
```

---

# 📂 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Powered-Phishing-Detection-System.git

cd AI-Powered-Phishing-Detection-System
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
streamlit run phishing_detection.py
```

---

# 📊 Training the Model

Upload a CSV file containing the following columns:

| Column | Description |
|---------|-------------|
| email_body | Email content |
| label | 1 = Phishing, 0 = Safe |

After uploading the dataset, click **Train Model**.

---

# 🔍 How to Use

1. Upload your dataset (optional).
2. Train the model.
3. Paste an email body.
4. Click **Check Phishing**.
5. View:
   - Prediction
   - Confidence Score
   - Risk Level

---

# 📧 Example Emails

## 🚨 Phishing Email

```text
Subject: Urgent Account Verification

Dear Customer,

We detected suspicious activity on your account.

Verify immediately:

https://secure-bank-login-update.com

Failure to verify within 24 hours will result in account suspension.

Thank you,
Security Team
```

---

## ✅ Safe Email

```text
Subject: Team Meeting Reminder

Hi Team,

This is a reminder that our weekly meeting is scheduled for tomorrow at 10 AM.

Regards,
Project Manager
```

---

# 📊 Results

The application predicts whether an email is:

- 🚨 Phishing
- ✅ Safe

It also displays:

- Prediction Confidence
- Risk Level
- Recent Prediction History

---

# 📸 Preview

## Phishing Email

![test1](https://github.com/user-attachments/assets/d1bdae97-ee19-4ac1-bd3f-87b98bad89b8)

---

## Safe Email

![test2](https://github.com/user-attachments/assets/dcef2e5c-01d8-4fb3-9343-6e73d7cb3588)

---

# 🛠 Future Improvements

- TF-IDF Vectorizer
- BERT-based Email Classification
- URL Reputation APIs
- Email Attachment Analysis
- Explainable AI (SHAP/LIME)
- Multi-language Phishing Detection

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Open a Pull Request.

---

# 🙏 Acknowledgements

- Streamlit
- scikit-learn
- NLTK
- pandas
- NumPy
- Shields.io

---

# 🌐 Connect with Me

📧 **Email**

**lonkarrohit77@gmail.com**

💼 **LinkedIn**

https://www.linkedin.com/in/rohit-lonkar-746948274/

---
⭐ If you found this project useful, consider giving it a star on GitHub!