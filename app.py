import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import spacy
import requests
from io import BytesIO
from PIL import Image
import base64
from serpapi import GoogleSearch
from openai import OpenAI

# Load NLP model
nlp = spacy.load("en_core_web_sm")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(layout="wide")
st.title("🔍 AI-Powered Resume Matcher & Insights")

# Extract resume text
def extract_resume_text(file):
    if hasattr(file, "name") and file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in pdf), pdf
    elif hasattr(file, "name") and file.name.endswith(".docx"):
        return docx2txt.process(file), None
    else:
        return "", None

# Get image from PDF (if any)
def extract_image_from_pdf(pdf):
    if not pdf:
        return None
    for page in pdf:
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            base_image = pdf.extract_image(xref)
            img_bytes = base_image["image"]
            return Image.open(BytesIO(img_bytes))
    return None

# Simple name extraction
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not found"

def extract_field(text, keywords):
    for line in text.split('\n'):
        for key in keywords:
            if key.lower() in line.lower():
                return line.split(":")[-1].strip()
    return "Not found"

def extract_section(text, keyword):
    lines = text.split('\n')
    section = []
    capture = False
    for line in lines:
        if keyword.lower() in line.lower():
            capture = True
        elif capture and line.strip() == "":
            break
        elif capture:
            section.append(line.strip())
    return "\n".join(section).strip() if section else "Not found"

# Match scoring
def keyword_match_score(resume_text, jd_text):
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    matched_keywords = list(resume_words.intersection(jd_words))
    score = (len(matched_keywords) / len(jd_words)) * 100 if jd_words else 0
    return score, matched_keywords

# Resume Suggestions
def resume_tips(text):
    tips = []
    if "objective" not in text.lower() and "summary" not in text.lower():
        tips.append("💡 Consider adding an Objective/Summary section.")
    if "achievement" not in text.lower():
        tips.append("📌 Include specific Achievements with measurable outcomes.")
    if "skills" not in text.lower():
        tips.append("🧠 Add a Skills section to highlight your expertise.")
    return tips

# Similar jobs using SerpAPI
def fetch_similar_jobs(role):
    params = {
        "engine": "google_jobs",
        "q": role,
        "location": "India",
        "api_key": st.secrets["SERPAPI_API_KEY"]
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        jobs = results.get("jobs_results", [])
        return jobs
    except Exception as e:
        st.error(f"Failed to fetch jobs: {e}")
        return []

# Summary from GPT
def get_resume_summary(resume_text, jd_text, score):
    try:
        if score > 75:
            summary = ("Your resume strongly aligns with the job role, showcasing relevant experience and skills. "
                       "Your projects and past work demonstrate good suitability for the job's responsibilities.")
            drawbacks = ("However, ensure your resume also includes measurable achievements or industry-specific keywords "
                         "to maximize ATS compatibility and stand out further.")
        else:
            summary = ("Your resume shows potential for the job, with some matching skills and relevant background. "
                       "To improve alignment, focus on tailoring your experience to match the job description more closely.")
            drawbacks = ("You might be missing out on some important keywords or domain-specific expertise that the job requires. "
                         "Consider enhancing your resume to bridge this gap.")

        prompt = f"""Resume Text:\n{resume_text}\n\nJob Description:\n{jd_text}
Score: {score}

Provide a professional 2-paragraph analysis based on this match.

1. Elaborate on strengths and suitability.
2. Point out shortcomings and how this job might help growth."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert career counselor."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"

# UI
resume_file = st.file_uploader("📄 Upload Your Resume (.pdf or .docx)", type=["pdf", "docx"])
jd_input = st.text_area("🧾 Paste Job Description", height=300)

if resume_file:
    resume_bytes = resume_file.read()
    text, pdf = extract_resume_text(BytesIO(resume_bytes))
    image = extract_image_from_pdf(pdf)

    with st.expander("📂 Uploaded Resume Content", expanded=False):
        st.code(text[:3000] + "\n..." if len(text) > 3000 else text)

    st.subheader("🧾 Resume Overview")
    if image:
        st.image(image, caption="Extracted Image", width=150)
    name = extract_name(text)
    role = extract_field(text, ["role", "job", "applying for"])
    experience = extract_section(text, "experience")
    education = extract_section(text, "education")

    st.markdown(f"**Name:** {name}")
    st.markdown(f"**Role Applied:** {role}")
    st.markdown(f"**Experience:** {experience[:300]}{'...' if len(experience) > 300 else ''}")
    st.markdown(f"**Education:** {education[:300]}{'...' if len(education) > 300 else ''}")

    if jd_input:
        st.subheader("🔍 Match Analysis")
        score, keywords = keyword_match_score(text, jd_input)
        st.metric("Match Score", f"{score:.2f}%")
        st.markdown(f"**Matched Keywords:** {', '.join(keywords[:15])}")

        st.subheader("💡 Resume Insights")
        for tip in resume_tips(text):
            st.info(tip)

        st.subheader("🌐 Similar Jobs You Might Like")
        for job in fetch_similar_jobs(role):
            st.markdown(f"- [{job['title']}]({job.get('via')}) at {job.get('company_name', 'Unknown')}")

        st.subheader("📝 Resume Summary & Fit")
        analysis = get_resume_summary(text, jd_input, score)
        st.write(analysis)
