import os
import re
import json
import math
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

import PyPDF2
import pytesseract
from PIL import Image
from io import BytesIO


try:
    import docx  # python-docx
except Exception:
    docx = None


try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


JOB_CSV_PATH = r"C:\Users\Priyalakshmi\Downloads\postings\job_postings_clean.csv"
OUTPUT_DIR = r"C:\Users\Priyalakshmi\Downloads\postings\outputs"
VIS_DIR = os.path.join(OUTPUT_DIR, "visuals")
RECS_CSV = os.path.join(OUTPUT_DIR, "personalized_recommendations.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "summary_report.txt")
TOP_K = 25
BERT_MODEL = "all-MiniLM-L6-v2"  # compact & fast
RANDOM_SEED = 42
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


sns.set(style="whitegrid")

# UTILITIES
def pick_file_dialog(title="Select file", filetypes=(("All files", "*.*"),)):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return path

def save_fig(fig, filename):
    path = os.path.join(VIS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print("Saved:", path)

def simple_clean(text):
    if not isinstance(text, str):
        return ""
    t = text.lower().replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

# TEXT EXTRACTION
def extract_text_from_docx(path):
    """Extract text from a .docx file using python-docx"""
    if docx is None:
        return ""
    doc = docx.Document(path)
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts)

def extract_text_from_pdf_textlayer(path):
    """Try to extract text using PyPDF2 (works if PDF has text layer)"""
    try:
        text_pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text_pages.append(txt)
        return "\n".join(text_pages)
    except Exception:
        return ""

def ocr_image(pil_img):
    """Run Tesseract OCR on a PIL Image and return text"""
    return pytesseract.image_to_string(pil_img)

def extract_text_from_pdf_ocr(path):
    """
    Convert PDF pages to images and OCR each page.
    Tries pdf2image (requires poppler), else tries fitz (PyMuPDF) rendering.
    """
    texts = []
    # Try pdf2image first
    if convert_from_path is not None:
        try:
            pages = convert_from_path(path, dpi=200)
            for p in pages:
                texts.append(ocr_image(p))
            return "\n".join(texts)
        except Exception as e:
            print("pdf2image failed:", e)
    # Try fitz as fallback
    if fitz is not None:
        try:
            doc = fitz.open(path)
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                # PyMuPDF pixmap -> bytes -> PIL image
                img = Image.open(BytesIO(pix.tobytes("png")))
                texts.append(ocr_image(img))
            return "\n".join(texts)
        except Exception as e:
            print("PyMuPDF (fitz) failed:", e)
    raise RuntimeError("No PDF->image path available: install pdf2image (with poppler) or pymupdf for scanned PDF OCR.")

def extract_text_from_image(path):
    """Load image and OCR with pytesseract"""
    img = Image.open(path)
    return ocr_image(img)

def extract_text_from_file(path):
    """Unified function: handles pdf, docx, images (jpg/png), fallback to OCR for scanned PDFs"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pdf"]:
        txt = extract_text_from_pdf_textlayer(path)
        if txt.strip():
            return txt
        # fallback to OCR
        try:
            print("No text layer found in PDF — falling back to OCR (this requires pdf2image/poppler or pymupdf).")
            txt2 = extract_text_from_pdf_ocr(path)
            return txt2
        except Exception as e:
            print("PDF OCR failed:", e)
            return ""
    elif ext in [".docx"]:
        return extract_text_from_docx(path)
    elif ext in [".doc"]:
        # .doc is old MS Word binary; best to ask user to save as .docx.
        print("Legacy .doc file detected. Please convert to .docx if possible.")
        return ""
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return extract_text_from_image(path)
    else:
        # unknown: try reading as text
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


# JOBS LOADING & SKILL POOL
def parse_skill_field(s):
    """Normalize a job_skill_set field into a list of skills"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    # remove wrapping brackets/quotes
    s = re.sub(r'^[\[\(\{]+|[\]\)\}]+$', '', s)
    # replace separators with comma
    s = re.sub(r'[\|/;]', ',', s)
    # remove quotes
    s = s.replace("'", "").replace('"', '')
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def load_jobs(csv_path=JOB_CSV_PATH):
    print("Loading jobs from:", csv_path)
    jobs = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected = ["company_name","category","job_title","job_skill_set","experience_level",
                "education_required","employment_type","salary_min","salary_max","date_posted"]
    for c in expected:
        if c not in jobs.columns:
            jobs[c] = ""
    # Salary numeric handling (avoid chained inplace warnings)
    jobs["salary_min"] = pd.to_numeric(jobs["salary_min"], errors="coerce")
    jobs["salary_max"] = pd.to_numeric(jobs["salary_max"], errors="coerce")
    med_min = jobs["salary_min"].median()
    med_max = jobs["salary_max"].median()
    jobs["salary_min"] = jobs["salary_min"].fillna(med_min)
    jobs["salary_max"] = jobs["salary_max"].fillna(med_max)
    jobs["salary_mid"] = (jobs["salary_min"] + jobs["salary_max"]) / 2.0
    # popularity
    jobs["company_postings"] = jobs.groupby("company_name")["job_title"].transform("count")
    maxp = jobs["company_postings"].max() if jobs["company_postings"].max() > 0 else 1
    jobs["popularity"] = jobs["company_postings"] / maxp
    # prepare text used for matching
    jobs["text_for_match"] = (jobs["job_title"].fillna("") + " " + jobs["category"].fillna("") + " " + jobs["job_skill_set"].fillna(""))
    jobs["text_for_match_clean"] = jobs["text_for_match"].apply(simple_clean)
    # parse skill sets into lists and build global pool
    skill_lists = []
    counter = Counter()
    for s in jobs["job_skill_set"].fillna("").astype(str):
        skl = parse_skill_field(s)
        skill_lists.append(skl)
        for k in skl:
            counter[k.lower()] += 1
    unique_skills = list(counter.keys())
    return jobs, unique_skills, counter, skill_lists


# SKILL MATCHING & UPSKILL ESTIMATE
def extract_candidate_skills(resume_text, skill_pool):
    """Return matched skills (case-insensitive substring match). If none matched, return top tokens as fallback."""
    text_l = resume_text.lower()
    matched = []
    for sk in skill_pool:
        if sk and sk.lower() in text_l:
            matched.append(sk)
    if matched:
        # de-duplicate preserving order
        seen = set()
        out = []
        for s in matched:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
    # fallback: pick top words (not great but simple)
    tokens = re.findall(r"[a-zA-Z]{3,}", text_l)
    freq = Counter(tokens)
    most = [w for w, _ in freq.most_common(20)]
    return most[:10]


def estimate_upskilling_effect(current_salary, missing_skills):
    """Heuristic uplift estimation (same approach as before)."""
    high = ["machine learning","deep learning","data science","nlp","cloud","aws","azure","spark"]
    core = ["python","sql","pandas","excel","javascript","react"]
    uplift = 0.0
    for s in missing_skills:
        s_l = s.lower()
        if any(h in s_l for h in high):
            uplift += 0.07
        elif any(c in s_l for c in core):
            uplift += 0.04
        else:
            uplift += 0.02
    uplift = min(uplift, 0.6)
    new_salary = current_salary * (1 + uplift)
    return round(uplift,4), float(new_salary)


# EMBEDDINGS & RECOMMENDATION
class SimpleEmbedder:
    def __init__(self, model_name=BERT_MODEL):
        print("Loading sentence-transformers model:", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def build_tfidf(jobs_texts):
    tf = TfidfVectorizer(max_features=12000)
    mat = tf.fit_transform(jobs_texts)
    return tf, mat


def hybrid_score(candidate_tfidf, candidate_bert, jobs_tfidf_mat, jobs_bert_mat, jobs_popularity, w_tfidf=0.4, w_bert=0.5, w_pop=0.1):
    s1 = cosine_similarity(candidate_tfidf, jobs_tfidf_mat).ravel()
    s2 = cosine_similarity(candidate_bert.reshape(1,-1), jobs_bert_mat).ravel()
    combined = w_tfidf * s1 + w_bert * s2 + w_pop * jobs_popularity
    return combined


# VISUALIZATIONS
def plot_salary_distribution(jobs):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(jobs["salary_mid"], bins=40, kde=True, ax=ax)
    ax.set_title("Salary Distribution (job postings, midpoint)")
    save_fig(fig, "salary_distribution.png")
    plt.close(fig)


def plot_top_skills(skill_counter, top_n=30):
    top = skill_counter.most_common(top_n)
    names = [t[0] for t in top]
    vals = [t[1] for t in top]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names)*0.25)))
    sns.barplot(x=vals, y=names, ax=ax)
    ax.set_title("Top Skills by Frequency")
    save_fig(fig, "top_skills.png")
    plt.close(fig)


def plot_skill_gap_for_recs(recs_df):
    if recs_df.empty:
        return
    df = recs_df.copy()
    df["num_matched"] = df["matched_skills"].apply(lambda s: 0 if not s else len(str(s).split(";")))
    df["num_missing"] = df["skills_to_learn"].apply(lambda s: 0 if not s else len(str(s).split(";")))
    df["matched_pct"] = df.apply(lambda r: r["num_matched"]/(r["num_matched"]+r["num_missing"]) if (r["num_matched"]+r["num_missing"])>0 else 0, axis=1)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df)*0.25)))
    sns.barplot(x="matched_pct", y="job_title", data=df, ax=ax)
    ax.set_title("Matched Skills Percent (Top Recommendations)")
    save_fig(fig, "skill_gap_top_recs.png")
    plt.close(fig)


def plot_skill_demand_vs_salary(skill_counter, jobs):
    rows = []
    for sk, cnt in skill_counter.items():
        mask = jobs["job_skill_set"].fillna("").str.lower().str.contains(re.escape(sk.lower()), na=False)
        avg_sal = jobs.loc[mask, "salary_mid"].mean() if mask.any() else None
        if avg_sal and not math.isnan(avg_sal):
            rows.append({"skill": sk, "count": cnt, "avg_salary": avg_sal})
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(50)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=df, x="count", y="avg_salary", size="count", sizes=(30,200), ax=ax)
    ax.set_title("Skill Demand (count) vs Average Salary")
    save_fig(fig, "skill_demand_vs_salary.png")
    plt.close(fig)


def plot_clusters(jobs_bert, jobs):
    try:
        feats = np.hstack([jobs_bert, jobs["salary_mid"].values.reshape(-1,1)])
        scaler = StandardScaler()
        feats_s = scaler.fit_transform(feats)
        k = min(6, max(2, int(len(jobs)/50)))
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED)
        labs = km.fit_predict(feats_s)
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        two = pca.fit_transform(feats_s)
        dfc = pd.DataFrame({"x": two[:,0], "y": two[:,1], "cluster": labs, "salary_mid": jobs["salary_mid"].values})
        fig, ax = plt.subplots(figsize=(10,8))
        sns.scatterplot(data=dfc, x="x", y="y", hue="cluster", size="salary_mid", sizes=(20,200), alpha=0.7, ax=ax)
        ax.set_title("Job Clusters (PCA projection)")
        save_fig(fig, "job_clusters.png")
        plt.close(fig)
    except Exception as e:
        print("Cluster plot failed:", e)


# MAIN PIPELINE
def run_pipeline():
    print("Loading job postings from fixed path (no dialog)...")
    jobs, unique_skills, skill_counter, skill_lists = load_jobs(JOB_CSV_PATH)
    print(f"Loaded {len(jobs)} jobs with {len(unique_skills)} unique skills (sample):", list(skill_counter.most_common(10))[:10])

    # Ask user to pick resume file (multi-format)
    print("\nPick your resume file (PDF / DOCX / PNG / JPG).")
    resume_path = pick_file_dialog("Select resume (PDF/DOCX/PNG/JPG)", filetypes=(("All supported", "*.pdf;*.docx;*.png;*.jpg;*.jpeg"), ("All files","*.*")))
    if not resume_path:
        print("No resume selected — exiting.")
        return pd.DataFrame(), jobs

    print("Extracting text from resume...")
    resume_text = extract_text_from_file(resume_path)
    if not resume_text.strip():
        print("Failed to extract text from resume. If it's a scanned PDF, make sure pdf2image (poppler) or PyMuPDF is installed for OCR fallback.")
        return pd.DataFrame(), jobs

    # Get candidate contact info (optional)
    email_m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text)
    phone_m = re.search(r"(\+?\d{1,3}[\s-]?)?(\(?\d{3,4}\)?[\s-]?)?\d{6,10}", resume_text)
    candidate_email = email_m.group(0) if email_m else ""
    candidate_phone = phone_m.group(0) if phone_m else ""
    print("Parsed contact (email/phone):", candidate_email, candidate_phone)

    # Extract candidate skills
    candidate_skills = extract_candidate_skills(resume_text, unique_skills)
    print(f"Extracted {len(candidate_skills)} candidate skills (sample):", candidate_skills[:20])

    # Ask for expected salary (try to detect)
    detected_nums = [int(n) for n in re.findall(r"\b\d{4,7}\b", resume_text.replace(",", ""))]
    detected_salary = max(detected_nums) if detected_nums else None
    expected_salary = None
    if detected_salary:
        resp = input(f"Detected number in resume that may be salary: {detected_salary}. Use it? (y/n) [n]: ").strip().lower()
        if resp == "y":
            expected_salary = float(detected_salary)
    if expected_salary is None:
        val = input("Enter your expected/current yearly salary (numeric). Press Enter to use dataset median: ").strip()
        if val == "":
            expected_salary = float(jobs["salary_mid"].median())
            print("Using dataset median salary:", expected_salary)
        else:
            try:
                expected_salary = float(val)
            except Exception:
                expected_salary = float(jobs["salary_mid"].median())
                print("Invalid input. Using dataset median:", expected_salary)

    # Build TF-IDF and BERT embeddings for jobs
    print("\nBuilding TF-IDF (jobs) and loading BERT model...")
    tfidf, jobs_tfidf = build_tfidf(jobs["text_for_match_clean"].tolist())
    embedder = SimpleEmbedder(BERT_MODEL)
    jobs_bert = embedder.encode(jobs["text_for_match"].tolist())

    # Candidate vectors
    cand_text = " ".join(candidate_skills) if candidate_skills else resume_text[:2000]
    cand_tfidf = tfidf.transform([simple_clean(cand_text)])
    cand_bert = embedder.encode([cand_text])[0]

    # Hybrid scoring
    print("Computing hybrid score (TF-IDF + BERT + popularity)...")
    scores = hybrid_score(cand_tfidf, cand_bert, jobs_tfidf, jobs_bert, jobs["popularity"].values)
    jobs = jobs.copy()
    jobs["match_score"] = scores
    ranked = jobs.sort_values("match_score", ascending=False).reset_index(drop=True)

    # Build recommendations rows (TOP_K)
    rec_rows = []
    for i, row in ranked.head(TOP_K).iterrows():
        req_skills = parse_skill_field(row["job_skill_set"])
        cand_set = set([c.lower() for c in candidate_skills])
        matched = [s for s in req_skills if s and s.lower() in cand_set]
        missing = [s for s in req_skills if s and s.lower() not in cand_set]
        salary_mid = float(row["salary_mid"])
        salary_gap = salary_mid - expected_salary
        uplift_pct, new_salary = estimate_upskilling_effect(expected_salary, missing)
        rec_rows.append({
            "company_name": row["company_name"],
            "category": row["category"],
            "job_title": row["job_title"],
            "salary_min": row["salary_min"],
            "salary_max": row["salary_max"],
            "salary_mid": salary_mid,
            "match_score": float(row["match_score"]),
            "matched_skills": "; ".join(matched),
            "skills_to_learn": "; ".join(missing),
            "salary_gap": float(salary_gap),
            "estimated_upskilling_pct": uplift_pct,
            "estimated_salary_after_upskilling": new_salary
        })

    recs_df = pd.DataFrame(rec_rows)
    RECS_CSV = "C:/Users/Priyalakshmi/Downloads/postings/outputs/personalized_recommendations.csv"

    # If the file exists, save a new version instead of overwriting
    if os.path.exists(RECS_CSV):
        base, ext = os.path.splitext(RECS_CSV)
        RECS_CSV = f"{base}_new{ext}"

    recs_df.to_csv(RECS_CSV, index=False)
    print(f"Saved recommendations to: {RECS_CSV}")
    print("Saved personalized recommendations CSV:", RECS_CSV)

    # Visuals focused on salary & skills
    print("Generating visuals (salary, skills, skill-gap, clusters)...")
    try:
        plot_salary_distribution(jobs)
    except Exception as e:
        print("salary plot error:", e)
    try:
        plot_top_skills(skill_counter)
    except Exception as e:
        print("top skills plot error:", e)
    try:
        plot_skill_gap_for_recs(recs_df)
    except Exception as e:
        print("skill gap plot error:", e)
    try:
        plot_skill_demand_vs_salary(skill_counter, jobs)
    except Exception as e:
        print("skill demand vs salary error:", e)
    try:
        plot_clusters(jobs_bert, jobs)
    except Exception as e:
        print("clusters plot error:", e)

    # Summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "jobs_loaded": int(len(jobs)),
        "unique_skills_count": len(unique_skills),
        "candidate_skills_count": len(candidate_skills),
        "expected_salary_used": expected_salary,
        "recommendations_csv": RECS_CSV,
        "visuals_folder": VIS_DIR,
        "candidate_email": candidate_email,
        "candidate_phone": candidate_phone
    }
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary:", SUMMARY_TXT)

    # Quick console preview
    print("\nTop recommendations (preview):")
    if recs_df.empty:
        print("No recommendations found.")
    else:
        print(recs_df.head(10).to_string(index=False))

    print("\nPipeline complete — outputs are in:", OUTPUT_DIR)
    print("You can display 'personalized_recommendations.csv' later in an HTML page as needed.")
    return recs_df, jobs


# POST-ANALYSIS: UPSKILLING & VISUALS
def analyze_upskilling_opportunities(recommendations_df, jobs_df, skill_counter):
    """
    Analyze what new skills will give the highest salary boost.
    """
    print("\nAnalyzing Upskilling Opportunities...\n")

    if recommendations_df is None or recommendations_df.empty:
        print("No recommendations to analyze.")
        return {}

    # Ensure skills_to_learn is split into list form
    def split_skills_field(x):
        if not x or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, list):
            return x
        # string like "skill1; skill2"
        return [s.strip() for s in str(x).split(";") if s.strip()]

    valid = recommendations_df.copy()
    # calculate avg_salary for recs
    valid["avg_salary"] = valid.apply(lambda r: (r.get("salary_min", np.nan) + r.get("salary_max", np.nan))/2.0, axis=1)
    valid = valid.dropna(subset=["avg_salary"])
    # Flatten missing skills and map to average salary
    skill_salary = []
    for _, row in valid.iterrows():
        missing = split_skills_field(row.get("skills_to_learn", ""))
        for sk in missing:
            skill_salary.append((sk.lower(), float(row["avg_salary"])))
    if not skill_salary:
        print("No missing skills found in recommendations.")
        return {}

    skill_salary_df = pd.DataFrame(skill_salary, columns=["skill", "avg_salary"])
    top_skill_salary = skill_salary_df.groupby("skill")["avg_salary"].mean().sort_values(ascending=False).head(15)

    # Skill frequency (demand) using skill_counter (jobs-level)
    top_freq = pd.Series({k: v for k, v in skill_counter.items()}).sort_values(ascending=False).head(30)

    # Compute estimated salary hike
    current_avg = jobs_df["salary_mid"].mean()
    potential_avg = top_skill_salary.mean()
    hike_percent = ((potential_avg - current_avg) / current_avg) * 100 if current_avg and not math.isnan(current_avg) else 0.0

    print(f" Average Salary Now (jobs dataset mid): ₹{current_avg:,.0f}")
    print(f"Estimated Salary After Targeted Upskilling: ₹{potential_avg:,.0f}")
    print(f"Potential Growth: +{hike_percent:.2f}%")

    # Visualization Section using VIS_DIR
    visuals_dir = VIS_DIR
    os.makedirs(visuals_dir, exist_ok=True)

    # 1) Bar chart: Top-paying upskill opportunities
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_skill_salary.values, y=top_skill_salary.index, palette="crest", ax=ax)
        ax.set_title("💰 Top Upskilling Opportunities by Estimated Salary Impact")
        ax.set_xlabel("Average Salary Offered (midpoint)")
        ax.set_ylabel("Skill to Learn")
        plt.tight_layout()
        save_fig(fig, "top_upskill_salary.png")
        plt.close(fig)
    except Exception as e:
        print("Error making top_upskill_salary:", e)

    # 2) Horizontal bar: Most Demanded Skills (from job pool)
    try:
        top_freq_plot = top_freq.head(15)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x=top_freq_plot.values, y=top_freq_plot.index, palette="flare", ax=ax)
        ax.set_title("🔥 Most Demanded Skills in Job Market (job postings)")
        ax.set_xlabel("Frequency in Job Listings")
        ax.set_ylabel("Skill")
        plt.tight_layout()
        save_fig(fig, "top_demanded_skills.png")
        plt.close(fig)
    except Exception as e:
        print("Error making top_demanded_skills:", e)

    # 3) Pie chart: Category-wise Job Distribution (top 6 categories)
    try:
        fig, ax = plt.subplots(figsize=(8,8))
        jobs_df["category"].value_counts().head(6).plot.pie(autopct="%1.1f%%", startangle=90, cmap="Pastel1", ax=ax)
        ax.set_title("🏢 Job Distribution by Category")
        ax.set_ylabel("")
        plt.tight_layout()
        save_fig(fig, "job_category_distribution.png")
        plt.close(fig)
    except Exception as e:
        print("Error making job_category_distribution:", e)

    # 4) Heatmap: correlation between demand and avg salary for top skills found
    try:
        heat_rows = []
        for sk in top_skill_salary.index:
            demand = skill_counter.get(sk, 0)
            avg_sal = top_skill_salary.loc[sk]
            heat_rows.append({"skill": sk, "demand": demand, "avg_salary": avg_sal})
        heat_df = pd.DataFrame(heat_rows).set_index("skill")
        if not heat_df.empty:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(heat_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("📉 Correlation: Skill Demand vs Salary Impact")
            plt.tight_layout()
            save_fig(fig, "skill_correlation_heatmap.png")
            plt.close(fig)
    except Exception as e:
        print("Error making skill_correlation_heatmap:", e)

    # 5) Lollipop Chart: Salary Growth Potential by Skill
    try:
        vals = top_skill_salary.values
        labels = top_skill_salary.index
        y = np.arange(len(vals))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hlines(y, [0], vals, color="gray", alpha=0.7)
        ax.plot(vals, y, "o")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Average Salary (midpoint)")
        ax.set_title("📈 Salary Growth Potential by Skill (Lollipop)")
        plt.tight_layout()
        save_fig(fig, "salary_growth_lollipop.png")
        plt.close(fig)
    except Exception as e:
        print("Error making lollipop:", e)

    # 6) Treemap: top skills by uplift potential (visual size ~ avg_salary)
    try:
        treemap_df = top_skill_salary.reset_index().rename(columns={0:"avg_salary", "index":"skill"})
        treemap_df.columns = ["skill", "avg_salary"]
        treemap_df = treemap_df.head(20)
        sizes = treemap_df["avg_salary"].values
        labels = [f"{s}\n₹{v:,.0f}" for s, v in zip(treemap_df["skill"], treemap_df["avg_salary"])]
        plt.figure(figsize=(12,6))
        squarify.plot(sizes=sizes, label=labels, alpha=.9)
        plt.axis("off")
        plt.title("Treemap: Top Skills by Average Salary (potential)")
        plt.tight_layout()
        save_fig(plt.gcf(), "skill_treemap.png")
        plt.close()
    except Exception as e:
        print("Error making treemap:", e)

    # 7) Funnel (plotly) - a simple career funnel demonstration
    try:
        stages = ['Current Role', 'Skill Acquisition', 'Target Role']
        values = [100, 70, 45]
        fig = go.Figure(go.Funnel(y=stages, x=values, textinfo="value+percent previous"))
        fig.update_layout(title="Career Transition Funnel (Upskilling Path)")
        funnel_path = os.path.join(visuals_dir, "career_transition_funnel.png")
        fig.write_image(funnel_path)
        print("Saved:", funnel_path)
    except Exception as e:
        print("Error making funnel chart (requires plotly and kaleido):", e)

    # 8) Radar chart (plotly) — career growth projection
    try:
        dims = ['Salary Growth','Job Match','Skill Breadth','Career Versatility']
        # relative scales 0..1
        current = [0.4, 0.5, 0.45, 0.4]
        uplift = min(max(hike_percent/100.0, 0), 0.6)
        projected = [min(1, current[0] + uplift), 0.7, 0.8, 0.75]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=current, theta=dims, fill='toself', name='Current'))
        fig.add_trace(go.Scatterpolar(r=projected, theta=dims, fill='toself', name='After Upskilling'))
        fig.update_layout(title="Career Growth Projection (Radar Chart)", polar=dict(radialaxis=dict(visible=True, range=[0,1])))
        radar_path = os.path.join(visuals_dir, "career_growth_radar.png")
        fig.write_image(radar_path)
        print("Saved:", radar_path)
    except Exception as e:
        print("Error making radar chart (plotly+kaleido):", e)

    # Save a textual summary
    try:
        top3 = list(top_skill_salary.index[:3])
        mean_uplift = (potential_avg - current_avg)/current_avg if current_avg else 0
        summary_text = (
            f"Generated at: {datetime.now().isoformat()}\n\n"
            f"Average salary (dataset mid): ₹{current_avg:,.0f}\n"
            f"Estimated salary after targeted upskilling: ₹{potential_avg:,.0f}\n"
            f"Potential average growth: {hike_percent:.2f}%\n\n"
            f"Top skills to learn (sample): {', '.join(top3)}\n"
        )
        with open(os.path.join(visuals_dir, "growth_summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_text)
        print("Saved growth summary.")
    except Exception as e:
        print("Error saving growth summary:", e)

    return {
        "avg_salary_now": current_avg,
        "avg_salary_after": potential_avg,
        "growth_percent": hike_percent,
        "top_upskills": list(top_skill_salary.index[:5])
    }


# RUN EVERYTHING
if __name__ == "__main__":
    recs_df, jobs_df = run_pipeline()
    # recs_df may be empty if user cancelled or error
    if recs_df is not None and not recs_df.empty:
        upskill_summary = analyze_upskilling_opportunities(recs_df, jobs_df, Counter({k:v for k,v in (jobs_df["job_skill_set"].fillna("").astype(str).map(lambda s: ";".join(parse_skill_field(s))).str.split(";").explode().value_counts().to_dict()).items()}))
        if upskill_summary:
            print("\n\nSummary:")
            print(f"- Current Avg Salary: ₹{upskill_summary['avg_salary_now']:.0f}")
            print(f"- After Upskilling: ₹{upskill_summary['avg_salary_after']:.0f}")
            print(f"- Salary Growth: +{upskill_summary['growth_percent']:.1f}%")
            print(f"- Recommended Skills to Learn (top): {', '.join(upskill_summary['top_upskills'])}")
    else:
        print("No recommendations generated — skipping upskilling analysis.")
