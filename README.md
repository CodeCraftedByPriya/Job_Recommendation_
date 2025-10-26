<h1 align="center">💼 Job-Market Recommender System</h1>
<p align="center"><i>Bridging Skills, Jobs, and Market Insights with Predictive Analytics</i></p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/YourGitHubUsername/JobMarketRecommender?style=flat-square">
  <img src="https://img.shields.io/badge/python-70%25-blue?style=flat-square">
  <img src="https://img.shields.io/badge/languages-2-blue?style=flat-square">
</p>

---

<h3 align="center">Tools & Technologies</h3>
<div align="center">  
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-Scikit--learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-Matplotlib-2067b8?style=for-the-badge&logo=python" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-Seaborn-10069F?style=for-the-badge" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-re-ff69b4?style=for-the-badge" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-Collections-9cf?style=for-the-badge" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-HTML5-e34c26?style=for-the-badge&logo=html5&logoColor=white" style="margin: 4px;" />  
  <img src="https://img.shields.io/badge/-CSS3-264de4?style=for-the-badge&logo=css3&logoColor=white" style="margin: 4px;" />  
</div>  

---

## Project Overview
The **Job-Market Recommender System** matches **job seekers** with relevant jobs based on **skills, education, and experience**.  
It also highlights **skill gaps**, **salary mismatches**, and **labour market inefficiencies**.

---

## Objectives
- Content-based matching of candidate skills with job postings  
- Extract and analyze in-demand skills using **NLP**  
- Identify skill gaps and salary differences  
- Provide insights for students, universities, and policymakers  

---

## Methodology
1. **Data Collection:** Job postings (`all_job_skills.csv`) & candidate resumes  
2. **Data Cleaning:** Handle missing values, tokenize and normalize text, standardize salaries  
3. **Feature Engineering:** TF-IDF vectorization, spaCy noun extraction, embeddings  
4. **Modeling:**  
   - **Content-Based Filtering:** Cosine similarity  
   - **K-Means Clustering:** Group jobs by skill patterns  
   - **Skill Gap Analysis:** Compare candidate skills vs job requirements  
5. **Evaluation:** Precision, recall, qualitative validation  
6. **Visualization:** Skill demand, salary mismatch graphs, heatmaps  

---

## Models Used
- **TF-IDF:** Vectorize job descriptions and skills  
- **Cosine Similarity:** Match candidates with jobs  
- **K-Means Clustering:** Identify job clusters by skill requirements  

---

## Libraries Used
| Category | Libraries |
|----------|-----------|
| Data Handling | `pandas`, `numpy` |
| NLP & Feature Extraction | `scikit-learn`, `spaCy`, `re` |
| Machine Learning | `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `collections`, `os`, `warnings` |

---

## How It Works
1. Upload and clean job posting dataset  
2. Extract skills using NLP  
3. Candidate enters their skill profile  
4. Compute cosine similarity between candidate and jobs  
5. Display top job recommendations  
6. Visualize trends and clusters  


## 👩‍💻 Contributors
- **Priyalakshmi**  
- **Karismaa**  
