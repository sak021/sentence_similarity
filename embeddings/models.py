from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def get_tfidf_embeddings(base_data, records_data, model_name):
    if os.path.exists('./embeddings/tfidf_vectorizer.pkl'):
        vectorizer = joblib.load('./embeddings/tfidf_vectorizer.pkl')
        tfidf_vectors_base = np.load('./embeddings/base_data_tfidf_embeddings.npy')

    else:
        vectorizer = TfidfVectorizer()
        tfidf_vectors_base = vectorizer.fit_transform(base_data)
        joblib.dump(vectorizer, './embeddings/tfidf_vectorizer.pkl')
        np.save(f"./embeddings/base_{model_name}_embeddings.npy", tfidf_vectors_base)

    tfidf_vectors_records = vectorizer.transform(records_data)
    np.save(f"./embeddings/records_{model_name}_embeddings.npy", tfidf_vectors_records)

    tfidf_array_base = tfidf_vectors_base.toarray()
    tfidf_array_records = tfidf_vectors_records.toarray()
    return tfidf_array_base, tfidf_array_records



def get_embeddings(model_name, base, record, save_embedding=False):
    if model_name == 'word2vec':
        if os.path.exists('./embeddings/base_word2vec_embeddings.wv'):
            embedding_vectors_base = Word2Vec(base, min_count=1, vector_size=100).wv  # Adjust size as needed
            embedding_vectors_record = Word2Vec(record, min_count=1, vector_size=100).wv
        if save_embedding:
            embedding_vectors_base.save(f"./embeddings/{base}_{model_name}_embeddings.wv")
            embedding_vectors_record.save(f"./embeddings/{record}_{model_name}_embeddings.wv")

    elif model_name == 'tfidf':
        base_vectors, record_vector  = get_tfidf_embeddings(base, record, model_name)
    
    elif model_name=='gpt3':
        embedding_vectors = []

    return base_vectors, record_vector