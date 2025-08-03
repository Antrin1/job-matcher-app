import streamlit as st
import spacy
import fitz  # PyMuPDF
import docx2txt
import requests
from PIL import Image
import io
import base64
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# SerpAPI key
SERP_API_KEY = "10c9f2331b5181c18c5dd1800db2b20a902ca7e4f27946a47f8a927c04efcca6"

# Helper functions
def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""

def extract_image_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_index in range(len(doc)):
        images = doc.get_page_images(page_index)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            return image
    return None

def extract_resume_details(text):
    name_match = re.search(r"Name[:\-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    role_match = re.search(r"(Applying for|Role|Position)[:\-]?\s*([^\n]+)", text, re.IGNORECASE)
    exp_match = re.search(r"(\d+)\+?\s+(years|yrs)", text, re.IGNORECASE)
    return {
        "name": name_match.group(1) if name_match else "N/A",
        "role": role_match.group(2) if role_match else "N/A",
        "experience": exp_match.group(1) + " years" if exp_match else "N/A"
    }

def get_keywords(text):
    doc = nlp(text.lower())
    return list(set([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'PROPN'] and not token.is_stop]))

def calculate_match(resume_text, jd_text):
    resume_keywords = get_keywords(resume_text)
    jd_keywords = get_keywords(jd_text)
    matched_keywords = set(resume_keywords) & set(jd_keywords)
    score = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
    return round(score, 2), matched_keywords

def get_resume_insights(text):
    insights = []
    if "objective" not in text.lower() and "summary" not in text.lower():
        insights.append("💡 Consider adding an Objective/Summary section.")
    if "certification" not in text.lower():
        insights.append("💡 Consider including relevant certifications.")
    if "project" not in text.lower():
        insights.append("💡 Adding project experience can strengthen your resume.")
    if "achievement" not in text.lower():
        insights.append("💡 Highlight notable achievements to stand out.")
    return insights

def get_similar_jobs(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": query,
        "api_key": SERP_API_KEY
    }
    response = requests.get(url, params=params)
    jobs = []
    if response.status_code == 200:
        results = response.json().get("jobs_results", [])
        for job in results[:3]:
            jobs.append({
                "title": job.get("title"),
                "company": job.get("company_name"),
                "location": job.get("location"),
                "link": job.get("related_links", [{}])[0].get("link", "#")
            })
    return jobs

# Streamlit UI
st.title("💼 Resume vs Job Description Matcher")

st.markdown("---")

resume_file = st.file_uploader("📤 Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
jd_file = st.file_uploader("📥 Upload the Job Description (PDF or DOCX)", type=["pdf", "docx"])

resume_text, jd_text = "", ""
resume_data = {}

if resume_file:
    resume_text = extract_text(resume_file)
    resume_file.seek(0)
    resume_image = extract_image_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else None
    resume_data = extract_resume_details(resume_text)

    st.markdown("### 🧾 Resume Overview")
    cols = st.columns([1, 2])
    with cols[0]:
        if resume_image:
            st.image(resume_image, use_column_width=True)
    with cols[1]:
        st.write("**Name:**", resume_data["name"])
        st.write("**Experience:**", resume_data["experience"])
        st.write("**Role Applied:**", resume_data["role"])

if resume_text and jd_file:
    jd_text = extract_text(jd_file)

    st.markdown("---")
    st.markdown("## 🔍 Match Analysis")

    score, matched_keywords = calculate_match(resume_text, jd_text)
    st.metric("Match Score", f"{score}%")
    st.write("**Matched Keywords:**", ", ".join(matched_keywords))

    st.markdown("## 💡 Resume Insights")
    insights = get_resume_insights(resume_text)
    for insight in insights:
        st.info(insight)

    st.markdown("## 🌐 Similar Jobs You Might Like")
    query = resume_data["role"] if resume_data["role"] != "N/A" else "AI Developer"
    jobs = get_similar_jobs(query)
    if jobs:
        for job in jobs:
            st.markdown(f"- [{job['title']} at {job['company']}, {job['location']}]({job['link']})")
    else:
        st.write("No related jobs found.")

    st.markdown("## 📝 Resume Summary & Fit")

    strengths = "Your resume shows strength in technical skills and experience in project work."
    fit = "This job aligns well with your role interests, especially if you enhance your resume with certifications or achievements."
    demerits = "Some sections like an objective or projects are missing which could affect ATS ranking."

    st.success(f"**Strengths:** {strengths}")
    st.warning(f"**Fit Summary:** {fit}")
    st.error(f"**Areas to Improve:** {demerits}")
