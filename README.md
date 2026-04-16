# AI-Powered-Resume-Screening-System-NLP-ML

An intelligent web application that automates resume classification, skill extraction, and candidate filtering using Machine Learning and NLP techniques.

---

## Features

- Upload CSV files containing multiple resumes
- Upload PDF resumes for real-time prediction
- Automatic resume classification into job categories
- Skill extraction from resume text
- Experience detection using NLP
- Search and filter candidates by:
  - Skills
  - Category
  - Experience
- Export processed results to Excel

---

## Technologies Used

- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas**
- **TF-IDF Vectorization**
- **HTML, CSS**

---

## Machine Learning Model and NLP model

- Algorithm:**Logistic Regresstion** and **LinearSVC (Support Vector Machine)**
- Text Processing:
  - Cleaning and normalization
  - TF-IDF feature extraction
- Accuracy: ~75%

---

## Project Structure
Resume-Analyzer/
│── app.py
│── clf.pkl
│── tfidf.pkl
│── label_encoder.pkl
│── Resume.csv
│
├── templates/
│ ├── index.html
│ ├── upload.html
│ ├── upload_pdf.html
│ ├── search.html
│ ├── result.html
│
├── static/
│ ├── style.css
│
└── README.md

## Install dependencies
pip install -r requirements.txt

## Run the application in terminal
python app.py

## Web Application link
http://127.0.0.1:5000/
