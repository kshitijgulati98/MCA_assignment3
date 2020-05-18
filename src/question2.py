#Assignment3
#Question2
#relevance feedback

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#import scipy.sparse.csr_matrix.sum as matrixsum


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    iter=3
    alpha=0.8
    beta=0.2
    for i in range(0,iter):
        for j in range(sim.shape[1]):
            relevant_docs_index=np.argsort(-sim[:, j])[:n]
            relevant_docs=vec_docs[np.array(relevant_docs_index),:]
            non_relevant_docs=vec_docs[~np.array(relevant_docs_index),:]
            vec_queries[j,:]+= alpha*np.sum(relevant_docs,axis=0) - beta*np.sum(non_relevant_docs,axis=0)
            
        sim=cosine_similarity(vec_docs,vec_queries)
        
    rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    iter=3
    alpha=0.8
    beta=0.2
    for i in range(0,iter):
        for j in range(sim.shape[1]):
            relevant_docs_index=np.argsort(-sim[:, j])[:n]
            relevant_docs=vec_docs[np.array(relevant_docs_index),:]
            non_relevant_docs=vec_docs[~np.array(relevant_docs_index),:]
            r=relevant_docs
            feature_names=tfidf_model.get_feature_names()
            features=[]
            
            for k in relevant_docs_index :
                rel = vec_docs[k].toarray()
                rel=rel[0].ravel()
                ind = np.argsort(-rel)
                top_ind=ind[:n]
                words=[]
                
                for t in top_ind:
                    words.append(feature_names[t])
                
                feature=' '.join(word for word in words)
                features.append(feature)
                
            tf=tfidf_model.transform(features)
            vec_queries[j,:]+= alpha*np.sum(relevant_docs,axis=0) - beta*np.sum(non_relevant_docs,axis=0) + np.sum(tf,axis=0)
                
        sim=cosine_similarity(vec_docs,vec_queries)
        
    rf_sim = sim  # change
    return rf_sim
