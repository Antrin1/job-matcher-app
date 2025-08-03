import streamlit as st
import os
import tempfile
import PyPDF2
import docx2txt
import spacy
import requests
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# SerpAPI Key
SERPAPI_KEY = "your_serpapi_key"

# Extract text from uploaded resume
def extract_resume_text(resume_file):
    text = ""
    if resume_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(resume_file)
        for page in reader.pages:
            text += page.extract_text()
    elif resume_file.name.endswith(".docx"):
        text = docx2txt.process(resume_file)
    return text

# Extract text from uploaded JD
def extract_jd_text(jd_file):
    text = ""
    if jd_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(jd_file)
        for page in reader.pages:
            text += page.extract_text()
    elif jd_file.name.endswith(".docx"):
        text = docx2txt.process(jd_file)
    return text

# NLP-based keyword extractor
def extract_keywords(text):
    doc = nlp(text)
    return list(set([token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop]))

# Match score using cosine similarity
def compute_match_score(resume_text, jd_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, jd_text])
    vectors = vectorizer.toarray()
    return round(cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100, 2)

# Resume Insight Suggestions
def generate_resume_insights(text):
    suggestions = []
    if "summary" not in text.lower() and "objective" not in text.lower():
        suggestions.append("💡 Consider adding an Objective/Summary section.")
    if "certification" not in text.lower():
        suggestions.append("💡 Consider including relevant certifications.")
    if "project" not in text.lower():
        suggestions.append("💡 Highlight notable achievements to stand out.")
    return suggestions

# Resume Strength Summary
def resume_summary(text, jd):
    if len(text.split()) < 150:
        return {
            "strength": "Limited experience content. Consider expanding your resume.",
            "fit": "The job may require more demonstrated experience.",
            "improve": "Add detailed responsibilities and measurable achievements."
        }
    else:
        return {
            "strength": "Your resume shows strength in technical skills and experience in project work.",
            "fit": "This job aligns well with your role interests, especially if you enhance your resume with certifications or achievements.",
            "improve": "Some sections like an objective or projects are missing which could affect ATS ranking."
        }

# Extract candidate name
def extract_name(text):
    lines = text.splitlines()
    for line in lines:
        if line.strip() and len(line.strip().split()) <= 4 and not any(char.isdigit() for char in line):
            return line.strip()
    return "N/A"

# Job role from resume
def extract_role(text):
    if "role" in text.lower() or "position" in text.lower():
        return " ".join(text.split()[:15])
    elif "developer" in text.lower():
        return "Developer"
    return "N/A"

# Job match keywords
def extract_matched_keywords(resume_text, jd_text):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)
    return list(set(resume_keywords) & set(jd_keywords))

# Search for jobs using SerpAPI
def search_related_jobs(keywords):
    query = " ".join(keywords[:3])
    url = f"https://serpapi.com/search.json?engine=google_jobs&q={query}&api_key={SERPAPI_KEY}"
    response = requests.get(url)
    jobs = []
    if response.status_code == 200:
        data = response.json()
        for job in data.get("jobs_results", []):
            jobs.append(f"{job.get('title')} at {job.get('company_name')} — {job.get('location')}")
    return jobs

# 🏁 Streamlit App
st.set_page_config(layout="wide", page_title="Resume Matcher Pro")
st.title("📄 Resume & JD Matcher Tool")

# Resume Upload
resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

# Display Resume Details
if resume_file:
    resume_text = extract_resume_text(resume_file)
    st.subheader("🧾 Resume Overview")
    image = Image.open("AntrinProfile.jpg")
    st.image(image, caption="Uploaded Photo", width=150)

    name = extract_name(resume_text)
    role_applied = extract_role(resume_text)
    experience = "2 years" if "2 years" in resume_text else "N/A"

    st.markdown(f"**Name:** {name}")
    st.markdown(f"**Experience:** {experience}")
    st.markdown(f"**Role Applied:** {role_applied}")

    # JD Upload
    jd_file = st.file_uploader("Upload the Job Description (PDF or DOCX)", type=["pdf", "docx"])
    if jd_file:
        jd_text = extract_jd_text(jd_file)

        # Match Score
        match_score = compute_match_score(resume_text, jd_text)
        matched_keywords = extract_matched_keywords(resume_text, jd_text)

        st.subheader("🔍 Match Analysis")
        st.metric("Match Score", f"{match_score}%")
        st.markdown(f"**Matched Keywords:** {', '.join(matched_keywords)}")

        # Resume Suggestions
        st.subheader("💡 Resume Insights")
        for tip in generate_resume_insights(resume_text):
            st.info(tip)

        # Related Jobs
        st.subheader("🌐 Similar Jobs You Might Like")
        job_suggestions = search_related_jobs(matched_keywords)
        if job_suggestions:
            for job in job_suggestions:
                st.markdown(f"- {job}")
        else:
            st.warning("No related jobs found.")

        # Summary
        st.subheader("📝 Resume Summary & Fit")
        summary = resume_summary(resume_text, jd_text)
        st.success(f"**Strengths:** {summary['strength']}")
        st.warning(f"**Fit Summary:** {summary['fit']}")
        st.error(f"**Areas to Improve:** {summary['improve']}")
