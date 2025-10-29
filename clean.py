import pandas as pd
import ast

# File paths
raw_path = r"C:\Users\Priyalakshmi\Downloads\postings\job_postings_raw.csv"
clean_path = r"C:\Users\Priyalakshmi\Downloads\postings\job_postings_clean.csv"

# Load dataset
df = pd.read_csv(raw_path, encoding='utf-8', low_memory=False)

# Drop unnecessary columns
cols_to_drop = ['location', 'job_description', 'job_id']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')


# Clean 'job_skill_set' column
def parse_skills(x):
    if pd.isna(x):
        return ()
    try:
        # Safely evaluate string like "[skill1, skill2]" to list or tuple
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return tuple(str(i).strip().lower() for i in val)
        elif isinstance(val, str):
            return tuple(s.strip().lower() for s in val.split(','))
        else:
            return ()
    except:
        return tuple(str(s).strip().lower() for s in str(x).split(','))


if 'job_skill_set' in df.columns:
    df['job_skill_set'] = df['job_skill_set'].apply(parse_skills)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Remove duplicates
df = df.drop_duplicates()

# Trim whitespace and normalize text columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

# Handle missing values (optional simple impute)
df = df.fillna('not specified')

# Save cleaned file
df.to_csv(clean_path, index=False, encoding='utf-8')
print(f"Cleaned dataset saved to: {clean_path}")
print(f"Shape after cleaning: {df.shape}")
print("Columns:", list(df.columns))
