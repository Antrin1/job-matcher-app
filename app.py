import streamlit as st
import re
import fitz  # PyMuPDF
import docx2txt
import requests
from io import BytesIO
from PIL import Image
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import base64

# Load spaCy model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# OpenAI Setup
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Extract image from resume
def extract_image_from_pdf(pdf_stream):
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            return Image.open(BytesIO(image_bytes))
    return None

# Extract text from uploaded file
def extract_resume_text(file):
    if file.name.endswith(".pdf"):
        return "\n".join([page.get_text() for page in fitz.open(stream=file.read(), filetype="pdf")])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return "\n".join([page.get_text() for page in fitz.open(stream=uploaded_file.read(), filetype="pdf")])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    return ""

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower()

def get_cosine_similarity(text1, text2):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score * 100, tfidf

def extract_section(text, header):
    pattern = rf"{header}.*?(?=\n[A-Z ]+?:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else ""

def extract_name(text):
    lines = text.split("\n")
    return lines[0].strip() if lines else "Unknown"

def extract_field(text, keywords):
    for line in text.split("\n"):
        for keyword in keywords:
            if keyword.lower() in line.lower():
                return line.strip()
    return "Not found"

def fetch_similar_jobs(query):
    serp_api_key = st.secrets["SERPAPI_KEY"]
    params = {
        "q": query,
        "location": "India",
        "api_key": serp_api_key,
        "engine": "google_jobs",
    }
    response = requests.get("https://serpapi.com/search", params=params)
    jobs = response.json().get("jobs_results", [])[:5]
    return jobs

def generate_resume_summary(resume, jd, score):
    prompt = f"""Analyze the following resume and job description and explain the strengths and drawbacks of the candidate applying for this job.
Resume: {resume}
Job Description: {jd}
Match Score: {score:.2f}%
Provide:
1. Summary of Strengths
2. Potential Gaps and Suggestions"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating summary: {e}"

# UI Start
st.title("🤖 AI Resume-JD Matcher v2.1")
st.markdown("Upload your resume and job description. Get insights, match score, and suggested jobs!")

resume_file = st.file_uploader("📄 Upload Resume (.pdf or .docx)", type=["pdf", "docx"])
jd_file = st.file_uploader("📑 Upload Job Description (.pdf or .txt)", type=["pdf", "txt"])

resume_text = ""
jd_text = ""

# if resume_file:
#     resume_bytes = resume_file.read()
#     resume_text = extract_resume_text(BytesIO(resume_bytes))
#     name = extract_name(resume_text)
#     role = extract_field(resume_text, ["role", "job", "applying for"])
#     experience = extract_section(resume_text, "experience")
#     education = extract_section(resume_text, "education")
#     image = extract_image_from_pdf(BytesIO(resume_bytes)) if resume_file.name.endswith(".pdf") else None

if resume_file:
    resume_bytes = resume_file.read()
    resume_file.seek(0)
    resume_text = extract_resume_text(resume_file)
    name = extract_name(resume_text)
    role = extract_field(resume_text, ["role", "job", "applying for"])
    experience = extract_section(resume_text, "experience")
    education = extract_section(resume_text, "education")
    image = extract_image_from_pdf(BytesIO(resume_bytes)) if resume_file.name.endswith(".pdf") else None


    st.subheader("👤 Resume Preview")
    if image:
        st.image(image, width=120)
    st.write(f"**Name:** {name}")
    st.write(f"**Role Applied:** {role}")
    st.write(f"**Experience:** {experience if experience else 'Not mentioned'}")
    st.write(f"**Education:** {education if education else 'Not mentioned'}")

if resume_text and jd_file:
    jd_text = extract_text(jd_file)
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    score, tfidf_vec = get_cosine_similarity(cleaned_resume, cleaned_jd)
    st.subheader("🎯 Match Score")
    st.progress(int(score))
    st.write(f"Your resume matches **{score:.2f}%** of the job description.")

    st.subheader("📝 Resume Summary & Fit")
    resume_fit = generate_resume_summary(resume_text, jd_text, score)
    st.markdown(resume_fit)

    st.subheader("💼 Similar Jobs You Might Like")
    query = extract_field(jd_text, ["title", "job", "role"])
    jobs = fetch_similar_jobs(query)
    for job in jobs:
        st.markdown(f"**{job.get('title')}** at *{job.get('company_name')}*")
        st.write(job.get("description", "")[:200] + "...")

