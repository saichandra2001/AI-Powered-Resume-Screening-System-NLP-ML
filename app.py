from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import re
import PyPDF2

app = Flask(__name__)

# ========================
# LOAD MODELS
# ========================
model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# GLOBAL DATA STORAGE
data = None

# ========================
# CLEAN TEXT
# ========================
def cleanResume(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========================
# PDF TEXT EXTRACTION
# ========================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ========================
# SKILLS EXTRACTION
# ========================
skills_list = [
    'python', 'java', 'sql', 'machine learning', 'flask', 'django',
    'excel', 'communication', 'teaching', 'recruitment',
    'html', 'css', 'javascript', 'data analysis'
]

def extract_skills(text):
    text = str(text).lower()
    found = [skill for skill in skills_list if skill in text]
    return ', '.join(found) if found else 'No skills found'

# ========================
# EXPERIENCE EXTRACTION
# ========================
def extract_experience(text):
    text = str(text).lower()
    match = re.search(r'(\d+)\+?\s+years', text)
    if match:
        return int(match.group(1))
    return 0

# ========================
# PREPARE DATAFRAME
# ========================
def prepare_dataframe(df):
    df = df.copy()

    if 'Resume_str' not in df.columns:
        return None

    df['cleaned_resume'] = df['Resume_str'].apply(cleanResume)

    X = tfidf.transform(df['cleaned_resume'])
    pred_ids = model.predict(X)
    df['predicted_category'] = le.inverse_transform(pred_ids)

    df['skills'] = df['cleaned_resume'].apply(extract_skills)
    df['experience'] = df['cleaned_resume'].apply(extract_experience)

    return df

# ========================
# HOME
# ========================
@app.route('/')
def index():
    return render_template('index.html')

# ========================
# UPLOAD CSV
# ========================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global data

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return "Please upload a CSV file."

        try:
            data = pd.read_csv(file)
        except Exception as e:
            return f"Error reading file: {e}"

        data = prepare_dataframe(data)

        if data is None:
            return "CSV must contain a 'Resume_str' column."

        display_data = data.copy()

        if 'Resume_str' in display_data.columns:
            display_data['Resume_str'] = display_data['Resume_str'].apply(
                lambda x: str(x)[:80] + "..."
            )

        show_cols = ['predicted_category', 'skills', 'experience', 'Resume_str']
        display_data = display_data[show_cols]

        if display_data.empty:
            return render_template(
                'result.html',
                tables=["<h3>No data found.</h3>"]
            )

        return render_template(
            'result.html',
            tables=[display_data.to_html(classes='data', index=False, escape=False)]
        )

    return render_template('upload.html')

# ========================
# UPLOAD PDF
# ========================
@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return "Please upload a PDF file."

        try:
            resume_text = extract_text_from_pdf(file)
            cleaned = cleanResume(resume_text)
            vector = tfidf.transform([cleaned])

            pred_id = model.predict(vector)[0]
            prediction = le.inverse_transform([pred_id])[0]

            skills = extract_skills(cleaned)
            experience = extract_experience(cleaned)

            result_html = f"""
            <h3>Predicted Category: {prediction}</h3>
            <h3>Skills: {skills}</h3>
            <h3>Experience: {experience} years</h3>
            """

            return render_template(
                'result.html',
                tables=[result_html]
            )

        except Exception as e:
            return f"Error reading PDF: {e}"

    return render_template('upload_pdf.html')

# ========================
# SEARCH FILTER SYSTEM
# ========================
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        skill = request.form.get('skill', '').strip()
        category = request.form.get('category', '').strip()
        experience = request.form.get('experience', '').strip()

        try:
            df = pd.read_csv("Resume.csv")
        except Exception as e:
            return f"Error loading Resume.csv: {e}"

        df = prepare_dataframe(df)

        if df is None:
            return "Resume.csv must contain a 'Resume_str' column."

        filtered = df.copy()

        if skill:
            filtered = filtered[
                filtered['skills'].str.contains(skill, case=False, na=False)
            ]

        if category:
            filtered = filtered[
                filtered['predicted_category'].str.lower() == category.lower()
            ]

        if experience:
            try:
                exp_value = int(experience)
                filtered = filtered[filtered['experience'] >= exp_value]
            except ValueError:
                return "Experience must be a number."

        display_data = filtered.copy()

        if 'Resume_str' in display_data.columns:
            display_data['Resume_str'] = display_data['Resume_str'].apply(
                lambda x: str(x)[:80] + "..."
            )

        show_cols = ['predicted_category', 'skills', 'experience', 'Resume_str']
        display_data = display_data[show_cols]

        if display_data.empty:
            return render_template(
                'result.html',
                tables=["<h3>No matching resumes found.</h3>"]
            )

        return render_template(
            'result.html',
            tables=[display_data.to_html(classes='data', index=False, escape=False)]
        )

    return render_template('search.html')

# ========================
# DOWNLOAD EXCEL
# ========================
@app.route('/download')
def download():
    global data

    if data is not None:
        output_path = "output.xlsx"
        data.to_excel(output_path, index=False)
        return send_file(output_path, as_attachment=True)
    else:
        return "No data available"

# ========================
# RUN
# ========================
if __name__ == '__main__':
    app.run(debug=True)