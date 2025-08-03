import streamlit as st
import re
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

st.title("🤖 AI Job Description Matcher v2.0")
st.markdown("Upload your resume and job description to get match score, quality tips, and smart skill suggestions.")

resume_file = st.file_uploader("📄 Upload Resume (.pdf or .txt)", type=["pdf", "txt"])
jd_file = st.file_uploader("📑 Upload Job Description (.pdf or .txt)", type=["pdf", "txt"])

# -------- FILE READ & CLEAN --------
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    return ""

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower()

# -------- COSINE SIM --------
def get_cosine_similarity(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score * 100, tfidf

# -------- SECTION EXTRACTION --------
def extract_section(text, header):
    pattern = rf"{header}.*?(?=\n[A-Z ]+?:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else ""

# -------- RESUME QUALITY CHECK --------
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

    buzzwords = ["hardworking", "go-getter", "self-starter", "synergy", "team player"]
    bw_found = [word for word in buzzwords if word in text.lower()]
    if bw_found:
        results.append(f"🚫 Buzzwords found: {', '.join(bw_found)}")

    return results

# -------- SKILL SUGGESTIONS --------
def suggest_skills(missing_words, jd_text):
    suggestions = []
    for word in missing_words:
        token = nlp(word)
        for token2 in nlp(jd_text):
            if token.similarity(token2) > 0.75 and token.text.lower() != token2.text.lower():
                suggestions.append(f"💡 Consider adding: '{token.text}' → Related to '{token2.text}'")
    return list(set(suggestions))

# -------- MAIN LOGIC --------
if resume_file and jd_file:
    resume_text = extract_text(resume_file)
    jd_text = extract_text(jd_file)

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    # Overall Match
    match_score, tfidf_vec = get_cosine_similarity(cleaned_resume, cleaned_jd)
    st.subheader("🎯 Overall Match Score")
    st.progress(int(match_score))
    st.write(f"Your resume matches **{match_score:.2f}%** of the job description.")

    # Section-wise scores
    st.subheader("📊 Section-wise Matching")
    for section in ["summary", "skills", "experience"]:
        resume_section = extract_section(resume_text, section)
        jd_section = extract_section(jd_text, "responsibilities" if section != "skills" else "requirements")
        if resume_section and jd_section:
            sec_score, _ = get_cosine_similarity(clean_text(resume_section), clean_text(jd_section))
            st.write(f"**{section.capitalize()}** match: {sec_score:.2f}%")
            st.progress(int(sec_score))

    # Resume health
    st.subheader("🩺 Resume Health Check")
    for tip in resume_health_check(resume_text):
        st.write(tip)

    # Smart suggestions
    st.subheader("🧠 Smart Skill Suggestions")
    jd_words = set(tfidf_vec.get_feature_names_out()[1:])
    resume_words = set(cleaned_resume.split())
    missing = jd_words - resume_words

    suggestions = suggest_skills(list(missing)[:20], jd_text)
    if suggestions:
        for s in suggestions[:10]:
            st.write(s)
    else:
        st.success("✅ All relevant keywords are already present in your resume!")
