import streamlit as st
import re
import fitz  # PyMuPDF
import spacy
import docx2txt
import PyPDF2
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------- Load SpaCy Model -------
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model not found.")
        return None

nlp = load_nlp()

# ------- Extract Text --------
def extract_resume_text(file):
    if file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def extract_jd_text(file):
    return extract_resume_text(file)

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower()

# ------- Extract Candidate Info --------
def extract_name(text):
    match = re.search(r"(?:Name|NAME|name)[^\n:]*[:\-]?\s*(.+)", text)
    return match.group(1).strip() if match else "Not Found"

def extract_experience(text):
    match = re.search(r"(\d+)\+?\s+years", text.lower())
    return match.group(1) + " years" if match else "Not Found"

def extract_role(text):
    match = re.search(r"(?i)(position|role applied for|title)[^\n:]*[:\-]?\s*(.+)", text)
    return match.group(2).strip() if match else "Not Found"

# ------- Resume Health --------
def resume_health_check(text):
    sections = ["summary", "experience", "education", "skills", "projects"]
    results = []
    word_count = len(text.split())

    for sec in sections:
        if sec not in text.lower():
            results.append(f"❌ Missing section: {sec.capitalize()}")

    if word_count < 150:
        results.append("⚠️ Resume is too short.")
    elif word_count > 1200:
        results.append("⚠️ Resume is too long.")
    else:
        results.append("✅ Resume length is good.")

    return results

# ------- Similarity Score --------
def get_cosine_similarity(text1, text2):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score * 100, tfidf

# ------- Job Recommendations --------
def fetch_related_jobs(query, serp_api_key):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_jobs",
        "q": query,
        "api_key": serp_api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        jobs = response.json().get("jobs_results", [])
        return jobs[:5]
    return []

# ------- Summary Fit --------
def generate_summary_insight(resume, jd):
    doc1 = nlp(resume)
    doc2 = nlp(jd)
    common_words = [token.text for token in doc1 if token.text in doc2.text and not token.is_stop]
    summary = (
        f"Your resume shows strengths in {', '.join(common_words[:5])}. "
        "This job could help enhance your current skills, especially in professional environments."
    )
    return summary

# ------- Upload UI --------
st.title("🤖 AI Resume & JD Matcher")
st.markdown("Get resume insights, match score, tips, and job suggestions.")

resume_file = st.file_uploader("📄 Upload Resume (.pdf/.docx/.txt)", type=["pdf", "docx", "txt"])
jd_file = st.file_uploader("📑 Upload Job Description (.pdf/.docx/.txt)", type=["pdf", "docx", "txt"])
serpapi_key = "10c9f2331b5181c18c5dd1800db2b20a902ca7e4f27946a47f8a927c04efcca6"

if resume_file:
    resume_text = extract_resume_text(resume_file)
    st.subheader("🧾 Resume Overview")
    
    name = extract_name(resume_text)
    role = extract_role(resume_text)
    experience = extract_experience(resume_text)

    st.markdown(f"**👤 Name:** {name}")
    st.markdown(f"**🎯 Role Applied For:** {role}")
    st.markdown(f"**📅 Experience:** {experience}")

    # Optional Image Preview
    if resume_file.type.startswith("image/"):
        image = Image.open(resume_file)
        st.image(image, caption="Uploaded Resume Image", width=150)

    st.subheader("🩺 Resume Health Check")
    for tip in resume_health_check(resume_text):
        st.write(tip)

if resume_file and jd_file:
    jd_text = extract_jd_text(jd_file)
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    # Match Score
    score, tfidf = get_cosine_similarity(cleaned_resume, cleaned_jd)
    st.subheader("🎯 Match Score")
    st.progress(int(score))
    st.markdown(f"**Match Score:** {score:.2f}%")

    # Summary Fit
    st.subheader("📌 Resume Summary & Fit")
    st.markdown(generate_summary_insight(resume_text, jd_text))

    # Similar Jobs
    st.subheader("🧭 Similar Jobs You Might Like")
    jobs = fetch_related_jobs(role, serpapi_key)
    if jobs:
        for job in jobs:
            st.markdown(f"- **{job.get('title', '')}** at {job.get('company_name', '')}")
            st.markdown(f"[More Info]({job.get('job_highlights', {}).get('link', '')})")
    else:
        st.info("No similar jobs found.")
