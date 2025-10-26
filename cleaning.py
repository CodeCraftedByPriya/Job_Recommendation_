import pandas as pd
import re


# Load the dataset
df = pd.read_csv("C:/Users/Priyalakshmi/Downloads/postings/all_job_post.csv", encoding='utf-8')

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())


# Remove duplicates
df.drop_duplicates(subset=['job_title', 'job_description'], inplace=True)


# Drop rows where title or description are missing
df = df.dropna(subset=['job_title', 'job_description'])

# Fill missing category or job_skill_set if any
df['category'] = df['category'].fillna('Not Specified')
df['job_skill_set'] = df['job_skill_set'].fillna('[]')


def clean_text(text):
    """Lowercase, remove special chars, extra spaces"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)       # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()   # normalize whitespace
    return text


df['clean_title'] = df['job_title'].apply(clean_text)
df['clean_description'] = df['job_description'].apply(clean_text)

# Keep job_skill_set as is, without cleaning
# But ensure it’s properly formatted (string, not NaN)
df['job_skill_set'] = df['job_skill_set'].astype(str)


clean_df = df[['job_id', 'category', 'job_title', 'clean_title',
               'job_description', 'clean_description', 'job_skill_set']].copy()

# Reset index
clean_df.reset_index(drop=True, inplace=True)


# Save cleaned file
output_path = "C:/Users/Priyalakshmi/Downloads/postings/clean_all_job_post.csv"
clean_df.to_csv(output_path, index=False, encoding='utf-8')

print("\n✅ Data cleaned and saved successfully!")
print("Saved file:", output_path)
print("Final shape:", clean_df.shape)

