import streamlit as st
import pandas as pd
import os
import JR
from JR import run_pipeline, JOB_CSV_PATH, OUTPUT_DIR, VIS_DIR, RECS_CSV, SUMMARY_TXT


st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide",
)

# CUSTOM STYLES
st.markdown("""
    <style>
        body {
            background-color: #FFFFFF;
        }
        .main-title {
            text-align: center;
            color: white;
            font-size: 65px;
            font-weight: 700;
            margin-top: 160px;
        }
        .sub-title {
            text-align: center;
            color: white;
            font-size: 22px;
            margin-top: -10px;
        }
        .header-section {
            background-color: #001B3A;
            height: 70px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-container {
            background-color: #001B3A;
            color: white;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s ease;
        }
        .metric-container:hover {
            transform: scale(1.05);
        }
        .section-title {
            font-size: 28px;
            font-weight: 600;
            color: white;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .upload-box {
            border: 2px dashed #001B3A;
            padding: 40px;
            border-radius: 15px;
            background-color: #f9fafb;
            text-align: center;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 14px;
            margin-top: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# HERO HEADER
st.markdown('<div class="header-section">', unsafe_allow_html=True)
st.markdown('<div class="main-title">Job Search Made Easy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Job search made smarter with AI</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# LOAD DATA
if os.path.exists(JOB_CSV_PATH):
    df = pd.read_csv(JOB_CSV_PATH)
else:
    st.error("Job dataset not found. Please check your JR pipeline output.")
    st.stop()

# DATA SUMMARY
st.markdown('<div class="section-title">Job Market Overview</div>', unsafe_allow_html=True)

# Try to detect a salary column automatically
salary_col = None
for c in df.columns:
    if 'salary' in c.lower() or 'pay' in c.lower():
        salary_col = c
        break

if salary_col:
    mean_salary = df[salary_col].mean().round(2)
    max_salary = df[salary_col].max().round(2)
else:
    mean_salary = max_salary = 0

num_companies = df['company_name'].nunique()
num_industries = df['category'].nunique() if 'category' in df.columns else 0
num_roles = df['job_title'].nunique()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"<div class='metric-container'><h3>Mean Salary</h3><h2>{mean_salary}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-container'><h3>Max Salary</h3><h2>{max_salary}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-container'><h3>Companies</h3><h2>{num_companies}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-container'><h3>Industries</h3><h2>{num_industries}</h2></div>", unsafe_allow_html=True)
with col5:
    st.markdown(f"<div class='metric-container'><h3>Roles</h3><h2>{num_roles}</h2></div>", unsafe_allow_html=True)

# VISUAL INSIGHTS
st.markdown('<div class="section-title">Visual Insights</div>', unsafe_allow_html=True)

chart_files = [
    "salary_distribution.png",
    "top_skills.png",
    "job_clusters.png",
    "job_category_distribution.png",
    "skill_correlation_heatmap.png",
    "top_demanded_skills.png",
    "top_upskill_salary.png",
    "skill_treemap.png",
    "skill_demand_vs_salary.png",
    "salary_growth_lollipop.png"
]

for chart in chart_files:
    chart_path = os.path.join(VIS_DIR, chart)
    if os.path.exists(chart_path):
        st.image(chart_path, use_container_width=True, caption=chart.replace("_", " ").replace(".png", "").title())
    else:
        st.warning(f"{chart} not found. Please run jr.py to generate charts.")

# RESUME UPLOAD
st.markdown('<div class="section-title">Upload Your Resume</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a PDF/DOCX/JPG/PNG resume", type=["pdf", "docx", "jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None:
    temp_path = os.path.join(OUTPUT_DIR, uploaded.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.info("Running your personalized job recommendations...")

    recs_df, jobs_df = run_pipeline()
    st.success("Personalized Analysis Generated!")

    # PERSONALIZED SECTION
    st.markdown('<div class="section-title">💼 Personalized Analysis</div>', unsafe_allow_html=True)

    # Top metrics summary
    if not recs_df.empty:
        mean_rec_salary = recs_df['salary_mid'].mean().round(2)
        matched_skills = recs_df['matched_skills'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0).mean()
        skill_gap = recs_df['skills_to_learn'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0).mean()
        unique_companies = recs_df['company_name'].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-container'><h3>Avg Recommended Salary</h3><h2>{mean_rec_salary}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-container'><h3>Avg Matched Skills</h3><h2>{matched_skills:.0f}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-container'><h3>Avg Skill Gap</h3><h2>{skill_gap:.0f}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-container'><h3>Companies Matched</h3><h2>{unique_companies}</h2></div>", unsafe_allow_html=True)

        st.markdown("### Recommended Roles")
        st.dataframe(recs_df.head(20))

        # Display extra visuals
        extra_charts = [
            "skill_gap_top_recs.png",
            "skills_matched_treemap.png",
            "companies_salary_treemap.png"
        ]
        for chart in extra_charts:
            cpath = os.path.join(VIS_DIR, chart)
            if os.path.exists(cpath):
                st.image(cpath, use_container_width=True, caption=chart.replace("_", " ").replace(".png", "").title())

# FOOTER
st.markdown("<div class='footer'>Built by Priyalakshmi SP| Karismaa Sheth — Job Search Made Smarter with AI</div>", unsafe_allow_html=True)

