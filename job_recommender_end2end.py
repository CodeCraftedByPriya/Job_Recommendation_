import os
import re
import ast
import math
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wordcloud import WordCloud
from tkinter import filedialog, Tk
import docx
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config & paths
# -----------------------
CSV_PATH = r"C:/Users/Priyalakshmi/Downloads/postings/clean_all_job_post.csv"
OUT_DIR = r"C:/Users/Priyalakshmi/Downloads/postings"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_N = 50   # how many recommendations to save
BERT_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast

# -----------------------
# Utilities: file dialog & resume reader
# -----------------------
def file_upload_dialog(title="Upload your resume (PDF/DOCX/TXT)"):
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title,
                                      filetypes=[("PDF files","*.pdf"),("Word Documents","*.docx;*.doc"),("Text files","*.txt")])
    root.destroy()
    return path if path else None

def read_resume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text_parts = []
        with open(path, "rb") as fh:
            reader = PdfReader(fh)
            for pg in reader.pages:
                pg_text = pg.extract_text()
                if pg_text:
                    text_parts.append(pg_text)
        return "\n".join(text_parts)
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    else:
        raise ValueError("Unsupported resume format.")

# -----------------------
# Parse job_skill_set column (keep list format)
# -----------------------
def parse_skill_list(s):
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return [str(x).strip().lower() for x in s if str(x).strip()]
    s = str(s).strip()
    if s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x).strip().lower() for x in val if str(x).strip()]
    except Exception:
        pass
    parts = re.split(r'[,;/\n]+', s)
    parts = [re.sub(r'^[\"\']+|[\"\']+$', '', p).strip().lower() for p in parts if p.strip()]
    return parts

# -----------------------
# Text normalize & skill find (vocab from job data)
# -----------------------
def normalize_text(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'[^a-z0-9\+\#\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_skills_from_text(text, vocab):
    txt = normalize_text(text)
    found = set()
    for skill in sorted(vocab, key=lambda x: -len(x)):
        if re.search(r'\b' + re.escape(skill) + r'\b', txt):
            found.add(skill)
    return sorted(found)

# -----------------------
# Small helper for plot border (draw rectangle around axes content)
# -----------------------
def add_plot_border(ax, pad=0.02, color='black', linewidth=1.2):
    # get axis position in figure coords
    ax_pos = ax.get_position()
    fig = ax.figure
    # add rectangle in figure coords
    rect = Rectangle((ax_pos.x0 - pad, ax_pos.y0 - pad),
                     ax_pos.width + 2*pad, ax_pos.height + 2*pad,
                     fill=False, edgecolor=color, linewidth=linewidth, transform=fig.transFigure, zorder=1000)
    fig.patches.append(rect)

# -----------------------
# Load & prepare dataset
# -----------------------
print("Loading cleaned dataset...")
df = pd.read_csv(CSV_PATH, dtype=str)
required = ['job_id','category','job_title','job_description','job_skill_set']
for col in required:
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' missing in CSV. Aborting.")

# Drop duplicates & missing essential fields
df.drop_duplicates(subset=['job_title','job_description'], inplace=True)
df = df.dropna(subset=['job_title','job_description']).reset_index(drop=True)
df['job_skill_set'] = df['job_skill_set'].fillna('[]')

# parse skills
df['parsed_skills'] = df['job_skill_set'].apply(parse_skill_list)
global_vocab = sorted({s for lst in df['parsed_skills'] for s in lst if s})

# prepare clean text columns for vectorizers
df['clean_title'] = df.get('clean_title', df['job_title'].fillna('').apply(normalize_text))
df['clean_description'] = df.get('clean_description', df['job_description'].fillna('').apply(normalize_text))
df['combined_text'] = (df['clean_title'].fillna('') + ' ' + df['clean_description'].fillna('') + ' ' + df['parsed_skills'].apply(lambda l: ' '.join(l))).astype(str)

print("Rows:", len(df), "Unique skills:", len(global_vocab))

# -----------------------
# Candidate resume: prompt upload and extract skills matched against global_vocab
# -----------------------
print("\nPlease choose your resume file in the popup...")
resume_path = file_upload_dialog()
if not resume_path:
    raise SystemExit("No resume uploaded. Exiting.")
resume_text = read_resume(resume_path)
candidate_skills = extract_skills_from_text(resume_text, global_vocab)
# fallback: token intersection
if not candidate_skills:
    tokens = re.split(r'[^a-z0-9\+\#]+', resume_text.lower())
    candidate_skills = sorted({t for t in tokens if t in set(global_vocab)})
print("Extracted candidate skills:", candidate_skills)

# -----------------------
# Build TF-IDF embeddings
# -----------------------
print("\nBuilding TF-IDF vectors...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(df['combined_text'].fillna(''))

candidate_text = ' '.join(candidate_skills) if candidate_skills else normalize_text(resume_text)
candidate_tfidf_vec = tfidf.transform([candidate_text])

# -----------------------
# Build BERT embeddings (sentence-transformers)
# -----------------------
print("Loading BERT model (sentence-transformers)...")
bert = SentenceTransformer(BERT_MODEL_NAME)
# compute embeddings for jobs in batches to avoid memory spike
def embed_texts(model, texts, batch_size=512):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        embeddings.append(model.encode(chunk, show_progress_bar=False))
    return np.vstack(embeddings)

job_texts = df['combined_text'].fillna('').tolist()
X_bert = embed_texts(bert, job_texts, batch_size=512)
candidate_bert_vec = bert.encode([candidate_text])[0].reshape(1, -1)

# -----------------------
# Similarities: TF-IDF and BERT
# -----------------------
print("Computing similarities...")
sim_tfidf = cosine_similarity(candidate_tfidf_vec, X_tfidf).flatten()   # size = n_jobs
sim_bert = cosine_similarity(candidate_bert_vec, X_bert).flatten()

# -----------------------
# Collaborative/popularity signal: # of skills listed (simple)
# -----------------------
popularity = df['parsed_skills'].apply(lambda lst: len(lst)).astype(float).values
# normalize popularity to [0,1]
if popularity.max() > 0:
    pop_norm = (popularity - popularity.min()) / (popularity.max() - popularity.min())
else:
    pop_norm = popularity

# -----------------------
# Hybrid score: weighted sum (you can tune weights)
# -----------------------
# weights
w_tfidf = 0.35
w_bert = 0.45
w_pop = 0.20

# normalize both sim arrays to 0-1
def minmax(a):
    if a.max() - a.min() == 0:
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min())

s_tfidf_n = minmax(sim_tfidf)
s_bert_n = minmax(sim_bert)

hybrid_score = w_tfidf * s_tfidf_n + w_bert * s_bert_n + w_pop * pop_norm

df['score_tfidf'] = s_tfidf_n
df['score_bert'] = s_bert_n
df['popularity'] = pop_norm
df['hybrid_score'] = hybrid_score

# top recommendations
top_rec = df.sort_values('hybrid_score', ascending=False).head(TOP_N).copy()
rec_csv = os.path.join(OUT_DIR, "recommendations_hybrid_top_{}.csv".format(TOP_N))
top_rec.to_csv(rec_csv, index=False, encoding='utf-8')
print("Saved top recommendations CSV:", rec_csv)

# -----------------------
# Skill-gap computation for top recs
# -----------------------
def compute_skill_gap(candidate_skills, job_skills):
    cand = set([s.lower() for s in candidate_skills])
    job = set([s.lower() for s in job_skills])
    matching = sorted(list(cand & job))
    missing = sorted(list(job - cand))
    return matching, missing

skillgap_rows = []
for _, r in top_rec.iterrows():
    match, missing = compute_skill_gap(candidate_skills, r['parsed_skills'])
    skillgap_rows.append({
        'job_id': r['job_id'],
        'job_title': r['job_title'],
        'category': r['category'],
        'hybrid_score': r['hybrid_score'],
        'matching_skills': ", ".join(match),
        'missing_skills': ", ".join(missing)
    })
skillgap_df = pd.DataFrame(skillgap_rows)
skillgap_csv = os.path.join(OUT_DIR, "skill_gap_top_{}.csv".format(TOP_N))
skillgap_df.to_csv(skillgap_csv, index=False, encoding='utf-8')
print("Saved skill-gap CSV:", skillgap_csv)

# -----------------------
# Analytics & Visualizations (save PNGs with borders)
# -----------------------

def save_fig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", path)

# 1) Skill Demand Analysis - bar and wordcloud
all_skills = [s for lst in df['parsed_skills'] for s in lst]
skill_counts = Counter(all_skills)
top_sk = skill_counts.most_common(30)

if top_sk:
    skills, counts = zip(*top_sk)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(skills, counts)
    ax.set_xticklabels(skills, rotation=75, ha='right')
    ax.set_title("Top In-Demand Skills (by # job listings)")
    add_plot_border(ax)
    save_fig(fig, "skill_demand_bar.png")

    # wordcloud
    try:
        wc = WordCloud(width=1200, height=400, background_color="white").generate_from_frequencies(skill_counts)
        fig, ax = plt.subplots(figsize=(12,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Top In-Demand Skills (WordCloud)")
        add_plot_border(ax)
        save_fig(fig, "skill_demand_wordcloud.png")
    except Exception:
        pass

# 2) Skill Gap Analysis (stacked bar for top recs)
sg = skillgap_df.copy()
if not sg.empty:
    sg['num_match'] = sg['matching_skills'].apply(lambda s: 0 if not s else len([i for i in s.split(',') if i.strip()]))
    sg['num_missing'] = sg['missing_skills'].apply(lambda s: 0 if not s else len([i for i in s.split(',') if i.strip()]))
    plot_n = min(20, sg.shape[0])
    sgp = sg.head(plot_n)
    fig, ax = plt.subplots(figsize=(14,6))
    idx = np.arange(plot_n)
    ax.bar(idx, sgp['num_match'], label='Matching')
    ax.bar(idx, sgp['num_missing'], bottom=sgp['num_match'], label='Missing')
    ax.set_xticks(idx)
    ax.set_xticklabels([str(t)[:60] for t in sgp['job_title']], rotation=75, ha='right')
    ax.set_ylabel("Number of Skills")
    ax.set_title("Skill Gap: Matching vs Missing (Top Recommendations)")
    ax.legend()
    add_plot_border(ax)
    save_fig(fig, "skill_gap_stacked_top_recs.png")

# 3) Skill Distribution by Category (stacked bar)
cat_skill_counts = {}
for cat, g in df.groupby('category'):
    c = Counter([s for lst in g['parsed_skills'] for s in lst])
    cat_skill_counts[cat] = dict(c.most_common(10))
all_skills_set = sorted({k for d in cat_skill_counts.values() for k in d.keys()})
if all_skills_set:
    cat_df = pd.DataFrame(0, index=sorted(cat_skill_counts.keys()), columns=all_skills_set)
    for cat, d in cat_skill_counts.items():
        for skill, cnt in d.items():
            cat_df.at[cat, skill] = cnt
    # pick top overall 8 skills to avoid crowd
    top_overall_sk = [s for s,_ in Counter(all_skills).most_common(8)]
    cat_df_small = cat_df[top_overall_sk]
    fig = cat_df_small.plot(kind='bar', stacked=True, figsize=(12,6)).figure
    ax = fig.axes[0]
    ax.set_title("Skill Distribution by Category (Top skills)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    add_plot_border(ax)
    save_fig(fig, "skill_distribution_by_category.png")

# 4) Job Match Score Distribution histogram
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(df['hybrid_score'].fillna(0), bins=30)
ax.set_title("Job Match Score Distribution (all jobs)")
ax.set_xlabel("Hybrid match score")
ax.set_ylabel("Count")
add_plot_border(ax)
save_fig(fig, "match_score_distribution_all_jobs.png")

# 5) Job Openings by Category (horizontal bar)
cat_counts = df['category'].value_counts().head(30)
fig, ax = plt.subplots(figsize=(10,8))
ax.barh(cat_counts.index[::-1], cat_counts.values[::-1])
ax.set_title("Job Openings by Category (Top 30)")
add_plot_border(ax)
save_fig(fig, "job_openings_by_category.png")

# 6) Trending Job Titles
title_counts = df['job_title'].value_counts().head(30)
fig, ax = plt.subplots(figsize=(12,6))
ax.bar(title_counts.index, title_counts.values)
ax.set_xticklabels(title_counts.index, rotation=75, ha='right')
ax.set_title("Trending Job Titles (Top 30)")
add_plot_border(ax)
save_fig(fig, "trending_job_titles.png")

# 7) NLP Embeddings visualization: PCA (2D) on BERT embeddings + candidate point
print("Creating embedding visualizations (PCA + t-SNE & clustering)...")
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_bert)
cand_pca2 = pca2.transform(candidate_bert_vec)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X_pca2[:,0], X_pca2[:,1], s=8, alpha=0.5, label='Jobs')
ax.scatter(cand_pca2[:,0], cand_pca2[:,1], s=120, marker='*', color='red', label='Candidate')
ax.set_title("BERT Embeddings (PCA 2D) - Jobs and Candidate")
ax.legend()
add_plot_border(ax)
save_fig(fig, "bert_pca_jobs_candidate.png")

# t-SNE is slower — do PCA(50)->t-SNE on subset if dataset is large
subset_for_tsne = min(2000, X_bert.shape[0])
X_sub = X_bert[:subset_for_tsne]
pca50 = PCA(n_components=50, random_state=42)
X_pca50 = pca50.fit_transform(X_sub)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=700, init='pca')
X_tsne = tsne.fit_transform(X_pca50)
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X_tsne[:,0], X_tsne[:,1], s=6, alpha=0.6)
ax.set_title("Jobs - t-SNE (subset)")
add_plot_border(ax)
save_fig(fig, "bert_tsne_jobs_subset.png")

# 8) Job clustering on BERT embeddings (KMeans) and show representative titles per cluster
n_clusters = 8
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
km_labels = km.fit_predict(X_bert)
df['cluster'] = km_labels
cluster_summary = []
for cl in range(n_clusters):
    sample_titles = df[df['cluster'] == cl]['job_title'].value_counts().head(5).index.tolist()
    cluster_summary.append({"cluster": cl, "top_titles": "; ".join(sample_titles)})
cluster_df = pd.DataFrame(cluster_summary)
cluster_csv = os.path.join(OUT_DIR, "cluster_summary.csv")
cluster_df.to_csv(cluster_csv, index=False)
print("Saved cluster summary:", cluster_csv)

# 9) User-Centric: Skill Improvement suggestions (bar)
missing_counter = Counter()
for ms in skillgap_df['missing_skills'].dropna():
    for s in [x.strip().lower() for x in ms.split(',') if x.strip()]:
        missing_counter[s] += 1
top_missing = missing_counter.most_common(15)
if top_missing:
    skills, counts = zip(*top_missing)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(skills, counts)
    ax.set_xticklabels(skills, rotation=65, ha='right')
    ax.set_title("Top Skill Improvement Suggestions (from top recommendations)")
    add_plot_border(ax)
    save_fig(fig, "skill_improvement_suggestions.png")

# 10) Heatmap of Job-Skill matches (top recommended jobs vs top skills)
top_sk_list = [s for s,_ in skill_counts.most_common(20)]
top_jobs_small = top_rec.head(40)
matrix = np.zeros((len(top_jobs_small), len(top_sk_list)), dtype=int)
for i, (_, r) in enumerate(top_jobs_small.iterrows()):
    js = set(r['parsed_skills'])
    for j, sk in enumerate(top_sk_list):
        matrix[i,j] = 1 if sk in js else 0

fig, ax = plt.subplots(figsize=(12,8))
cax = ax.imshow(matrix, aspect='auto', cmap='Blues')
ax.set_yticks(range(len(top_jobs_small)))
ax.set_yticklabels([str(t)[:60] for t in top_jobs_small['job_title']])
ax.set_xticks(range(len(top_sk_list)))
ax.set_xticklabels(top_sk_list, rotation=75, ha='right')
ax.set_title("Heatmap: Top Recommended Jobs (rows) vs Top Skills (cols)")
fig.colorbar(cax, ax=ax, fraction=0.02)
add_plot_border(ax)
save_fig(fig, "job_skill_heatmap_top_recs.png")

# Save short outputs
candidate_csv = os.path.join(OUT_DIR, "candidate_skills_extracted.csv")
pd.DataFrame({"skill": sorted(candidate_skills)}).to_csv(candidate_csv, index=False)
print("Saved candidate skills:", candidate_csv)

summary_text = os.path.join(OUT_DIR, "run_summary.txt")
with open(summary_text, "w", encoding="utf-8") as fh:
    fh.write("Recommender run summary\n")
    fh.write(f"Input CSV: {CSV_PATH}\n")
    fh.write(f"Rows: {len(df)}\n")
    fh.write(f"Candidate skills: {candidate_skills}\n")
    fh.write(f"Top recommendations CSV: {rec_csv}\n")
    fh.write(f"Skill-gap CSV: {skillgap_csv}\n")
    fh.write("PNG charts saved in this folder.\n")
print("Saved run summary:", summary_text)

print("\nAll done — outputs saved to:", OUT_DIR)
