<h1 align="center">💼 Job-Market Recommender System</h1>
<p align="center"><i>Bridging Skills, Jobs, and Market Insights with Predictive Analytics</i></p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/YourGitHubUsername/JobMarketRecommender?style=flat-square">
  <img src="https://img.shields.io/badge/python-70%25-blue?style=flat-square">
  <img src="https://img.shields.io/badge/languages-2-blue?style=flat-square">
</p>

---

<h3 align="center">🛠 Tools & Technologies</h3>
<div align="center">  
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python" />  
  <img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas" />  
  <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy" />  
  <img src="https://img.shields.io/badge/-Scikit--learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white" />  
  <img src="https://img.shields.io/badge/-SentenceTransformers-FFCC00?style=for-the-badge" />  
  <img src="https://img.shields.io/badge/-Matplotlib-2067b8?style=for-the-badge" />  
  <img src="https://img.shields.io/badge/-Seaborn-10069F?style=for-the-badge" />  
  <img src="https://img.shields.io/badge/-NLTK-3A9CDA?style=for-the-badge" />  
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />  
  <img src="https://img.shields.io/badge/-WordCloud-8A2BE2?style=for-the-badge" />  
  <img src="https://img.shields.io/badge/-re-ff69b4?style=for-the-badge" />  
  <img src="https://img.shields.io/badge/-OS-808080?style=for-the-badge" />  
</div>  

---

## Project Overview
The **Job-Market Recommender System** is a hybrid machine learning model designed to connect **job seekers** with the most relevant job openings using a blend of **text similarity**, **semantic embeddings**, and **data-driven insights**.  

It not only recommends jobs but also identifies:
- **Skill gaps** between a resume and job roles  
- **Salary trends** and **industry demand**  
- **Career insights** for students, universities, and recruiters  

---

## Objectives
- Build a **content-based recommender** using resume and job postings  
- Detect **skill gaps** and visualize market demand  
- Predict **salary variations** based on skill patterns  
- Create an **interactive dashboard** for insights and exploration  

---

## Project Workflow

| **Stage** | **File** | **Purpose** | **Output** |
|------------|-----------|--------------|-------------|
| **Data Cleaning** | `clean.py` | Cleans and standardizes raw job data | `job_postings_clean.csv` |
| **Model Pipeline** | `JR.py` | ML engine: job matching, embeddings, skill gap, visualizations | `recommendations.csv`, `charts/` |
| **Dashboard** | `app.py` | Streamlit app for visualization and interactivity | Live web interface |

---

## File 1: `clean.py` — Data Cleaning & Preprocessing

| **Step** | **Description** |
|-----------|-----------------|
| **Import Libraries** | Uses `pandas` and `ast` for data manipulation and safe parsing. |
| **Load Raw Dataset** | Reads the original job postings file (`job_postings_raw.csv`). |
| **Drop Unwanted Columns** | Removes unnecessary info like location, job ID, and full descriptions. |
| **Parse Skills** | Converts messy skill text (like `['Python', 'SQL']`) into clean, lowercased tuples. |
| **Normalize Column Names** | Converts headers to lowercase and replaces spaces with underscores. |
| **Remove Duplicates** | Ensures each posting is unique. |
| **Clean Text Columns** | Removes extra spaces and newlines for uniform formatting. |
| **Handle Missing Values** | Replaces missing fields with `'not specified'`. |
| **Save Output** | Exports the cleaned CSV for model input. |

**Output:**  
`C:\Users\Priyalakshmi\Downloads\postings\job_postings_clean.csv`

---

## File 2: `JR.py` — Machine Learning & Insights

| **Component** | **Functionality** |
|----------------|------------------|
| **Data Loading** | Reads the cleaned CSV created by `clean.py`. |
| **Text Preprocessing** | Lowercases, removes punctuation, and filters stopwords using NLTK. |
| **Feature Engineering** | Combines **TF-IDF** (keyword-based) and **BERT** (semantic-based) embeddings. |
| **Similarity Calculation** | Uses **Cosine Similarity** to measure how close a resume is to each job posting. |
| **KNN Model** | Finds the top `N` most similar jobs using nearest-neighbor logic. |
| **Skill Gap Analysis** | Compares resume and job skills → lists matching vs missing skills. |
| **Salary & Market Analysis** | Uses `matplotlib` and `seaborn` for salary insights and skill demand visualization. |
| **Visualization Generation** | Creates charts like:<br>• Top skills in demand<br>• Skill gap heatmap<br>• Salary vs skills<br>• Word clouds |
| **Output Files** | Saves: `recommended_jobs.csv`, skill gap reports, and plots. |

**Evaluation Metrics:**
- **Cosine Similarity Score:** Measures closeness of resume-job embeddings.  
- **KNN Distance:** Helps rank jobs by proximity in the vector space.  
- **Skill Overlap Percentage:** Quantifies how much a candidate matches job skills.

**Result:** Highly personalized, explainable job recommendations with interpretive visuals.

---

## File 3: `app.py` — Streamlit Dashboard

| **Feature** | **Purpose** |
|--------------|-------------|
| **Streamlit Framework** | Builds the entire user interface interactively. |
| **Page Configuration** | Custom title, layout, and favicon for a polished look. |
| **Custom CSS Styling** | Ensures white text, dark background, and uniform visuals. |
| **Header Section** | Displays project name and tagline. |
| **Data Loading** | Reads cleaned data and recommendations for display. |
| **Job Recommendations Display** | Shows job titles, company, skills matched, and skill gaps dynamically. |
| **Visual Insights Section** | Displays plots: salary distributions, skill gap charts, word clouds. |
| **Footer** | Adds developer credits and final branding. |

**Output:**  
Interactive dashboard hosted via Streamlit (local or cloud).

---

## Models & Metrics Summary

| **Model** | **Purpose** | **Explanation** |
|------------|-------------|-----------------|
| **TF-IDF** | Feature Extraction | Captures keyword importance in job text. |
| **BERT Embeddings** | Semantic Understanding | Extracts context and meaning beyond keywords. |
| **Cosine Similarity** | Recommendation Engine | Measures textual similarity between resume & job. |
| **KNN (Nearest Neighbor)** | Ranking | Finds top-N job matches by distance measure. |

---

## Libraries Used

| **Category** | **Libraries** |
|---------------|---------------|
| Data Handling | `pandas`, `numpy` |
| NLP & Text | `nltk`, `sentence-transformers`, `re`, `ast` |
| Machine Learning | `sklearn` (TF-IDF, cosine similarity, KNN) |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Dashboard | `streamlit`, `PIL`, `os` |

---

## How It Works

1. **Upload & Clean** job postings → `clean.py`  
2. **Generate ML Recommendations** using embeddings → `JR.py`  
3. **Visualize Results** and interact via web app → `app.py`  

The system then displays:
- Relevant job matches
- Skill gaps
- Market insights
- Salary trends  

---

## Contributors
- **Priyalakshmi S.P.**  
- **Karismaa**

