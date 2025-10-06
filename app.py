import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

clean_df = pd.read_csv("C:/Users/Priyalakshmi/Downloads/postings.csv/clean_postings.csv")
clean_df.head()

# Extract Skills (Using NLP)
'''using TF-IDF and Named Entity Recognition (NER) to identify and extract skill-related terms from job descriptions.'''
# Combine description + title and ensure all are strings
texts = (clean_df['clean_description'].fillna('') + ' ' + clean_df['clean_title'].fillna('')).astype(str)

# Create TF-IDF representation
tfidf = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(texts)

print("TF-IDF matrix created successfully with shape:", tfidf_matrix.shape)

# Build Content-Based Recommender (Skills ↔ Jobs)
'''This model recommends jobs that best match a candidate’s skills.'''
# Suppose candidate inputs their skills:
candidate_skills = "python sql data analysis machine learning econometrics"
candidate_vec = tfidf.transform([candidate_skills])

# Compute similarity between candidate skills and job descriptions
similarity_scores = cosine_similarity(candidate_vec, tfidf_matrix).flatten()

# Top job matches
top_indices = np.argsort(similarity_scores)[::-1][:10]
recommended_jobs = clean_df.iloc[top_indices][['title', 'company_name', 'location', 'normalized_annual_salary']]
print("Top recommended jobs:")
print(recommended_jobs)

# Skill Demand & Salary Insights
# Most In-Demand Skills
all_skills = [s for sublist in clean_df['clean_skills'].dropna().str.split() for s in sublist]
top_skills = Counter(all_skills).most_common(20)
print("Top 20 In-Demand Skills:", top_skills)

# Salary by Region
region_salary = clean_df.groupby('location')['normalized_annual_salary'].mean().sort_values(ascending=False).head(10)
region_salary.plot(kind='bar', title="Top Regions by Average Salary")
plt.ylabel("Average Annual Salary (USD)")
plt.show()

# Cluster Jobs by Skill Patterns
''' This helps you see which industries or roles group together in terms of skill requirements.'''
kmeans = KMeans(n_clusters=6, random_state=42)
clean_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Display representative job titles for each cluster
for i in range(6):
    print(f"\nCluster {i}:")
    print(clean_df[clean_df['cluster'] == i]['title'].head(5).to_list())
    
