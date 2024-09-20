import pandas as pd
import re
from ydata_profiling import ProfileReport
from embeddings.models import get_embeddings

columns_to_concatenate = ['txt', 'addr', 'addr2', 'city', 'zip', 'nation']


def profiling_report(data, data_name):
    # Generate a profile report
    records_profile = ProfileReport(data, title=f"{data_name} - Pandas Profiling Report", explorative=True)

    # Save the report to an HTML file
    records_profile.to_file(f"./Report/{data_name}_eda_report.html")

    # # Display it in a Jupyter Notebook
    # records_profile.to_notebook_iframe()



def clean_text(text):
    # Replace "::" with ","
    text = text.replace('::', ',')
    # Replace multiple commas with a single comma
    text = re.sub(r',+', ',', text)
    # Remove leading and trailing spaces
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',$', '', text)
    return text


def concat_and_clean(row):
    # Concatenate columns into a single string with commas
    concatenated = ','.join(row.astype(str).str.strip())
    # Apply cleaning function
    return clean_text(concatenated)


def main(records_data, base_data):

    ########### TASK 1 ##############
    # profiling_report(records_data, "Records")
    # profiling_report(base_data, "Base")
    # print("Task 1: Profiling report created.")


    ########### TASK 2 ##############
    # Drop rows where column 'A' has NaN or None values
    filtered_base_data = base_data.dropna(subset=['txt']).reset_index(drop=True).fillna('')
    filtered_records_data = records_data.dropna(subset=['frm']).reset_index(drop=True).fillna('')
    print("Task 2: Empty rows dropped.")


    ########### TASK 3 ##############
    filtered_records_data['cleaned_frm'] = filtered_records_data['frm'].astype(str).apply(clean_text)


    ########### TASK 4 ##############
    filtered_base_data['new_txt'] = filtered_base_data[columns_to_concatenate].fillna("").apply(concat_and_clean, axis=1)


    ########### TASK 5 ##############
    tfidf_base_embedding_vector, tfidf_records_embedding_vector  = get_embeddings("tfidf", filtered_base_data['new_txt'].to_list(), filtered_records_data['cleaned_frm'].to_list())

    ########### TASK 6 ##############
    # word2vec_base_embedding_vector, word2vec_records_embedding_vector = get_embeddings("word2vec", filtered_base_data['new_txt'].to_list(), filtered_records_data['cleaned_frm'])

    return None



if __name__ == "__main__":
    # records_data = pd.read_csv("./Evaluation_Assignment/Records.csv")
    # base_data = pd.read_csv("./Evaluation_Assignment/Base.csv")

    # main(records_data, base_data)


    # import numpy as np
    # import pandas as pd
    # from sklearn.metrics.pairwise import cosine_similarity

    # # Step 1: Load the embeddings from .npy files
    # records_embeddings = np.load('./embeddings/records_tfidf_embeddings.npy', allow_pickle=True)
    # base_embeddings = np.load('./embeddings/base_tfidf_embeddings.npy', allow_pickle=True)

    # # Step 2: Calculate cosine similarity between Records and Base embeddings
    # similarity_matrix = cosine_similarity(records_embeddings, base_embeddings)

    # # Step 3: Convert similarity matrix into a DataFrame for better readability
    # similarity_df = pd.DataFrame(similarity_matrix)

    from openai import OpenAI
    import numpy as np
    import pickle
    import os
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    # Set your OpenAI API
    client = OpenAI( api_key= os.environ['OPENAI_API_KEY'])

    def get_gpt3_embedding(text):
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"  # GPT-3 embedding model
        )
        # Access the embedding directly
        return np.array(response.data[0].embedding)

    def get_corpus_embeddings(corpus):
        return [get_gpt3_embedding(sentence) for sentence in corpus]

    # Load your DataFrame
    # df = pd.read_csv('your_file.csv')

    # Assuming your DataFrame has the following columns:
    # 'base_text' for the base sentences and 'record_text' for the record sentences
    base_corpus = ["Hi"]
    records_corpus = ["WHat"]

    # Get embeddings for both corpora
    base_embeddings = get_corpus_embeddings(base_corpus)
    # records_embeddings = get_corpus_embeddings(records_corpus)

    # Save embeddings using pickle
    with open('base_embeddings.pkl', 'wb') as f:
        pickle.dump(base_embeddings, f)

    # with open('records_embeddings.pkl', 'wb') as f:
    #     pickle.dump(records_embeddings, f)

    # # Calculate similarity for each record sentence with every base sentence
    # for i, record_embedding in enumerate(records_embeddings):
    #     similarities = cosine_similarity([record_embedding], base_embeddings)
    #     max_index = np.argmax(similarities)
    #     print(f"Most similar base sentence for record {i+1} is at index {max_index}, similarity: {similarities[0][max_index]}")



