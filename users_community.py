import networkx as nx
import numpy as np
import pandas as pd

from functional import seq
from resources import analyzed_comments_file, topics_file
from utils import load_dataset
from scipy import sparse
from scipy.sparse.linalg import svds
from utils import top_k_sim_docs, network_plot


if __name__ == '__main__':

    """
    Dato il dataset dei topic e il dataset dei commenti analizzati, andiamo a ricavare il grafo 
    che in base alle reazioni concordi degli utenti inserisce un link tra due topic. 

    Si basa sulle informazioni che si hanno dalla sentiment analysis, creando una matrice
    utente/topic. Questa matrice per ovvie ragioni è molto grande e sparsa, quindi effettuiamo `svds`
    ottenendo le matrici `u`, `s` e `v`. Manteniamo le prime 10 dimensioni.

    Successivamente, calcoliamo la similarità coseno a partire dalla matrice `v` per individuare i link da inserire 
    all'interno del grafo. 

    Infine, visualizziamo il grafo.
    """

    topic_dataset = load_dataset(topics_file())
    analyzed_comments = load_dataset(analyzed_comments_file())

    topic2terms = {i: topic['terms'] for i, topic in enumerate(topic_dataset)}
    post2topic = {post: i for i, topic in enumerate(
        topic_dataset) for post in topic['posts']}

    edges_user_post = [(comm['owner'], post2topic[comm['post']], comm['sentiment'][0])
                       for comm in analyzed_comments if 'post' in comm and comm['post'] in post2topic.keys()]

    edges_user_post = (seq(edges_user_post)
                       .group_by_key()
                       .map(lambda x: (x[0], set(x[1])))
                       .filter(lambda x: len(x[1]) > 1)
                       .flat_map(lambda x: [(x[0], y) for y in x[1]])
                       .to_list())

    df = pd.DataFrame(edges_user_post, columns=['user', 'topic', 'sentiment'])

    df = df.assign(sentiment=df.sentiment.apply(
        lambda x: -1 if x == 'negative' else 1))
    df = df.groupby(['user', 'topic']).sentiment.sum().reset_index()
    df = df.pivot(index='user', columns='topic', values='sentiment')
    df.fillna(0, inplace=True)
    matrix = df.values
    matrix_sparse = sparse.csr_matrix(matrix)
    u_mtx, s_mtx, v_mtx_t = svds(matrix, k=10)

    res = (u_mtx @ np.diag(s_mtx) @ v_mtx_t)

    edges = []

    for i in range(0, len(topic2terms)):
        elems = top_k_sim_docs(v_mtx_t.T, i, k=5, with_sims=True)

        for e, s in zip(*elems):  # type: ignore
            if (s > 0.7):
                edges.append((i, e, s))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    idx2graph_label = ['<br>'.join(
        [f'<b>{key}</b>'] + value) for key, value in topic2terms.items()]
    network_plot(G, nx.kamada_kawai_layout(G), idx2graph_label)
