import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/Priyalakshmi/Downloads/postings.csv/postings.csv")

# Drop duplicates and irrelevant columns
irrelevant_cols = [
    'job_id', 'company_id', 'views', 'expiry', 'closed_time',
    'original_listed_time', 'listed_time', 'posting_domain',
    'sponsored', 'zip_code', 'fips', 'application_url',
    'job_posting_url', 'application_type'
]
df.drop(columns=irrelevant_cols, inplace=True, errors='ignore')
df.drop_duplicates(subset=['title', 'company_name', 'description'], inplace=True)

# Handle missing values
# Remove rows with no job title or description
df = df.dropna(subset=['title', 'description'])

# Fill missing text-based columns
text_cols = ['company_name', 'formatted_experience_level', 'skills_desc', 'location']
for col in text_cols:
    df[col] = df[col].fillna("Not Specified")

# Fill salary columns — create a median salary estimate if missing
df['min_salary'] = df['min_salary'].fillna(0)
df['max_salary'] = df['max_salary'].fillna(0)
df['med_salary'] = df['med_salary'].fillna((df['min_salary'] + df['max_salary']) / 2)

# Standardize formats
# Normalize work type and experience
df['formatted_work_type'] = df['formatted_work_type'].str.upper().str.strip()
df['formatted_experience_level'] = df['formatted_experience_level'].str.upper().str.strip()

# Clean and simplify location
df['location'] = df['location'].str.replace(r'\s*,\s*', ', ', regex=True).str.strip()

# Text cleaning for NLP
import re


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['clean_description'] = df['description'].apply(clean_text)
df['clean_skills'] = df['skills_desc'].apply(clean_text)
df['clean_title'] = df['title'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)).lower().strip())

# Salary normalization
# Convert all salaries to yearly if pay_period exists


def normalize_salary(row):
    salary = row['med_salary']
    if row['pay_period'] == 'HOURLY':
        salary *= 2080  # approx 40 hrs * 52 weeks
    elif row['pay_period'] == 'MONTHLY':
        salary *= 12
    return salary


df['normalized_annual_salary'] = df.apply(normalize_salary, axis=1)

# Final cleanup
# Keep only relevant columns for your analysis
clean_df = df[[
    'company_name', 'title', 'clean_title', 'clean_description', 'clean_skills',
    'formatted_experience_level', 'formatted_work_type', 'location',
    'normalized_annual_salary'
]]

# Reset index
clean_df.reset_index(drop=True, inplace=True)

# Save the cleaned file
clean_df.to_csv("C:/Users/Priyalakshmi/Downloads/postings.csv/clean_postings.csv", index=False)
print("Data cleaned and saved as clean_postings.csv")
