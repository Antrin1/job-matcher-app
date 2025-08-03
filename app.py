import streamlit as st
import spacy
import fitz  # PyMuPDF
import docx2txt
import requests
from io import BytesIO

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Get job suggestions using SerpAPI
def get_job_suggestions(query, location="India"):
    api_key = "10c9f2331b5181c18c5dd1800db2b20a902ca7e4f27946a47f8a927c04efcca6"
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "api_key": api_key
    }
    res = requests.get("https://serpapi.com/search", params=params)
    jobs = []
    if res.status_code == 200:
        results = res.json().get("jobs_results", [])
        for job in results[:5]:
            jobs.append({
                "title": job.get("title"),
                "company": job.get("company_name"),
                "link": job.get("related_links", [{}])[0].get("link", "#")
            })
    return jobs

# Extract images from PDF
def extract_image_from_pdf(uploaded_file):
    images = []
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images

# Extract text from uploaded resume
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    return text

# UI layout
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("💼 AI Resume & JD Matcher with Job Suggestions")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📤 Upload your Resume", type=["pdf", "docx"])

with col2:
    jd_file = st.file_uploader("📄 Upload Job Description", type=["pdf", "docx"])

# Show uploaded file preview/download
if resume_file:
    st.subheader("📎 Uploaded Resume")
    st.download_button("⬇️ Download Resume", resume_file.read(), resume_file.name)

if jd_file:
    st.subheader("📎 Uploaded Job Description")
    st.download_button("⬇️ Download JD", jd_file.read(), jd_file.name)

if resume_file and jd_file:
    resume_file.seek(0)
    jd_file.seek(0)
    resume_text = extract_text(resume_file)
    jd_text = extract_text(jd_file)

    # Show resume image if available
    resume_file.seek(0)
    images = extract_image_from_pdf(resume_file)
    if images:
        st.image(images[0], caption="👤 Extracted Photo", use_column_width=False, width=150)

    # NLP similarity check
    resume_doc = nlp(resume_text)
    jd_doc = nlp(jd_text)
    similarity = resume_doc.similarity(jd_doc)
    st.markdown(f"### 🔍 Match Score: **{similarity * 100:.2f}%**")

    # Suggested jobs based on top entity
    top_entities = [ent.text for ent in resume_doc.ents if ent.label_ in ["ORG", "WORK_OF_ART", "PRODUCT", "GPE"]]
    if top_entities:
        role_keyword = top_entities[0]
        st.markdown(f"#### 🔁 Similar Jobs based on: *{role_keyword}*")
        job_list = get_job_suggestions(role_keyword)
        for job in job_list:
            st.markdown(f"**{job['title']}** at *{job['company']}*")
            st.markdown(f"[Apply Here]({job['link']})", unsafe_allow_html=True)
else:
    st.info("👆 Please upload both Resume and Job Description.")
