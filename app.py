
import streamlit as st
import spacy
import fitz  # PyMuPDF
import os
import base64
import tempfile
import requests
from PIL import Image
from docx import Document
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# SERP API config
SERP_API_KEY = "10c9f2331b5181c18c5dd1800db2b20a902ca7e4f27946a47f8a927c04efcca6"

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    image = None
    for i, page in enumerate(doc):
        text += page.get_text()
        if image is None:
            img_list = page.get_images(full=True)
            if img_list:
                xref = img_list[0][0]
                base_image = doc.extract_image(xref)
                image = Image.open(BytesIO(base_image["image"]))
    return text, image

def extract_text_from_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def get_keywords(text):
    doc = nlp(text.lower())
    return list(set([token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop]))

def similarity_score(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return round(cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 2)

def get_related_jobs(query):
    url = "https://serpapi.com/search.json"
    params = {
        "q": f"{query} jobs",
        "location": "India",
        "api_key": SERP_API_KEY,
        "engine": "google_jobs"
    }
    response = requests.get(url, params=params)
    jobs = []
    if response.status_code == 200:
        results = response.json().get("jobs_results", [])
        for job in results[:3]:
            jobs.append({
                "title": job.get("title"),
                "company": job.get("company_name"),
                "link": job.get("via")
            })
    return jobs

st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("🤖 AI Resume & JD Matcher - Enhanced Edition")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
with col2:
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])

if resume_file and jd_file:
    # Resume parsing
    if resume_file.type == "application/pdf":
        resume_text, resume_img = extract_text_from_pdf(resume_file)
    else:
        resume_text = extract_text_from_docx(resume_file)
        resume_img = None

    # JD parsing
    if jd_file.type == "application/pdf":
        jd_text, _ = extract_text_from_pdf(jd_file)
    else:
        jd_text = extract_text_from_docx(jd_file)

    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        if resume_img:
            st.image(resume_img, caption="Profile Photo", use_container_width=True)
    with col2:
        st.subheader("Resume Overview")
        name = nlp(resume_text).ents[0].text if nlp(resume_text).ents else "Not Detected"
        st.markdown(f"**Name:** {name}")
        st.markdown("**Experience:** Extracting...")
        st.markdown("**Role Applied:** Extracting...")

    st.markdown("---")
    st.subheader("🔍 Match Analysis")
    score = similarity_score(resume_text, jd_text)
    st.metric("Match Score", f"{score}%")

    res_keywords = get_keywords(resume_text)
    jd_keywords = get_keywords(jd_text)
    common = list(set(res_keywords) & set(jd_keywords))
    st.markdown(f"**Matched Keywords:** {', '.join(common[:15])}")

    st.subheader("💡 Resume Insights")
    tips = []
    if len(resume_text) < 500:
        tips.append("Resume is too short; consider elaborating your responsibilities and achievements.")
    if "objective" not in resume_text.lower():
        tips.append("Consider adding an Objective/Summary section.")
    if "project" not in resume_text.lower():
        tips.append("Highlight at least one major project.")

    for tip in tips:
        st.info(f"💡 {tip}")

    st.subheader("🌐 Similar Jobs You Might Like")
    related_jobs = get_related_jobs(name)
    for job in related_jobs:
        st.markdown(f"- **{job['title']}** at *{job['company']}* via [{job['link']}]({job['link']})")
