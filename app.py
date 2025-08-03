
import streamlit as st
import spacy
import fitz  # PyMuPDF
import docx2txt
import requests
import os
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import openai

# Load OpenAI key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from resume
def extract_resume_text(resume_file):
    if resume_file.name.endswith(".pdf"):
        reader = PdfReader(resume_file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif resume_file.name.endswith(".docx"):
        return docx2txt.process(resume_file)
    return ""

# Function to extract text from JD
def extract_jd_text(jd_file):
    if jd_file.name.endswith(".pdf"):
        reader = PdfReader(jd_file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif jd_file.name.endswith(".docx"):
        return docx2txt.process(jd_file)
    return ""

# GPT-generated summary
def generate_summary_with_gpt(resume_text, jd_text, match_score):
    prompt = f"""You are an expert resume and job match reviewer.
Given the resume below:
"""{resume_text}"""

And the job description below:
"""{jd_text}"""

Write two paragraphs:

1. A summary of how well this resume matches the job, considering skills, background, experience, and strengths.
2. A paragraph about potential weaknesses or gaps the candidate may need to improve for this job.

Mention if the resume strongly aligns or needs improvement based on the score: {match_score}%.
""" 
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# UI starts
st.title("📄 Resume Matcher & Analyzer")
resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
jd_file = st.file_uploader("Upload the Job Description (PDF or DOCX)", type=["pdf", "docx"])

# Show uploaded files
if resume_file:
    st.success(f"Uploaded Resume: {resume_file.name}")
if jd_file:
    st.success(f"Uploaded JD: {jd_file.name}")

# Process files
if resume_file and jd_file:
    resume_text = extract_resume_text(resume_file)
    jd_text = extract_jd_text(jd_file)

    # NLP based similarity match
    resume_doc = nlp(resume_text)
    jd_doc = nlp(jd_text)
    match_score = round(resume_doc.similarity(jd_doc) * 100, 2)

    # Display summary
    st.subheader("📊 Match Score")
    st.write(f"**{match_score}%**")

    st.subheader("🧠 AI-Generated Resume Fit Summary")
    summary_output = generate_summary_with_gpt(resume_text, jd_text, match_score)
    st.write(summary_output)

    # Display resume insights (name, role, education, experience)
    st.subheader("🧾 Resume Overview")
    name = ""
    role = ""
    education = []
    experience = []

    for sent in resume_doc.sents:
        if "name" in sent.text.lower() and not name:
            name = sent.text.strip()
        if "role" in sent.text.lower() and not role:
            role = sent.text.strip()
        if "education" in sent.text.lower():
            education.append(sent.text.strip())
        if "experience" in sent.text.lower():
            experience.append(sent.text.strip())

    st.write(f"**Name:** {name if name else 'N/A'}")
    st.write(f"**Role Applied:** {role if role else 'N/A'}")
    st.write(f"**Education:** {'; '.join(education) if education else 'N/A'}")
    st.write(f"**Experience:** {'; '.join(experience) if experience else 'N/A'}")
