import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import random

# List to store all queries
queries = []

# Iterate through all files in the directory
for filename in os.listdir('data_json'):
    if filename.endswith('.json'):
        filepath = os.path.join("data_json", filename)
        
        # Open and read the file
        with open(filepath, 'r') as file:
            data = json.load(file)
            
            # Extract queries
            for case in data:
                query = case.get('Query')
                if query:
                    queries.append(query)


# Load the cases from the CSV file
csv_file_path = 'data.csv'  # Update with the path to your CSV file
cases_df = pd.read_csv(csv_file_path)
cases_df = cases_df.dropna()


# Extract relevant columns
case_ids = cases_df['Case ID'].tolist()
case_titles = cases_df['Title'].tolist()
case_texts = cases_df['Key Issues'].tolist()
print(len(case_texts))

# Function to map queries to cases
def map_queries_to_cases(queries, case_texts, case_titles, case_ids, top_n=3):
    # Combine case texts and queries for vectorization
    combined_texts = case_texts + queries
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer().fit_transform(combined_texts)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between queries and cases
    cosine_similarities = cosine_similarity(vectors[len(case_texts):], vectors[:len(case_texts)])
    
    # Map each query to top_n relevant cases
    query_mappings = {}
    for i, query in enumerate(queries):
        # Get indices of the most similar cases
        relevant_indices = cosine_similarities[i].argsort()[-top_n:][::-1]

        # Get indices of the least similar 7 cases
        similarity_threshold = 0.3  # Define your threshold for irrelevance
        irrelevant_indices = [idx for idx in cosine_similarities[i].argsort()[:7] 
                    if cosine_similarities[i][idx] < similarity_threshold]
        
        # Map query to case IDs and titles
        relevant_case_list = [(case_texts[index], 1)for index in relevant_indices]
        irrelevant_case_list = [(case_texts[index], 0) for index in irrelevant_indices]
        query_mappings[query] = relevant_case_list + irrelevant_case_list
    
    return query_mappings

# Map the queries to relevant cases
mapped_queries = map_queries_to_cases(queries, case_texts, case_titles, case_ids, top_n=3)

# Output the mapping results
for query, cases in list(mapped_queries.items())[-2:]:
    print(f"Query: {query}")
    # for case_id, case_title in cases:
    #     print(f"  Relevant Case ID: {case_id}, Title: {case_title}")
    print()


# Save the mapping results to a JSON file
with open('fine-tune-data.json', 'w') as file:
    json.dump(mapped_queries, file, indent=4)