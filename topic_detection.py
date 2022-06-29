import gensim as gs
import numpy as np
from resources import processed_dataset_file, topics_file
from functools import reduce
from nltk.corpus import wordnet as wn
from stop_words import get_stop_words
from utils import load_dataset, store_dataset, docs_to_graph
from networkx.algorithms import community
from resources import processed_dataset_file, topics_file


def tfidf_representation(corpus_bow):
    """
    Funzione che ricevuto un corpus in forma BOW lo restituisce secondo la 
    rappresentazione vettoriale del TF-IDF.
    """
    tfidf = gs.models.TfidfModel(corpus_bow)
    return tfidf[corpus_bow]


def lsi_representation(corpus_tfidf, corpus_dict, num_topics=100):
    """
    Funzione che ricevuto un corpus in forma TF-IDF lo restituisce secondo la rappresentazione
    in uno spazio semantico utilizzando un modello LSI con num_topics come dimensioni.

    Riceve una lista di liste di tuple (id, tfidf) e restituisce una matrice.
    """
    lsi_model = gs.models.LsiModel(corpus_tfidf, id2word=corpus_dict, num_topics=num_topics) 
    lsi_corpus = lsi_model[corpus_tfidf]
    lsi_corpus_mtx = np.zeros((len(lsi_corpus), num_topics), dtype=np.float32)
    for i, doc in enumerate(lsi_corpus):
        for topic in doc:
            lsi_corpus_mtx[i, topic[0]] = topic[1]
    return lsi_corpus_mtx


def communities_from_graph(G, members_threshold=3):
    """
    Funzione che ricevuto un grafo restituisce una lista di liste di indici che rappresentano
    le community trovate. Le community sono trovate utilizzando il metodo Louvain di networkx.
    Ogni community è una lista di indici che rappresenta gli indici dei documenti inseriti
    al suo interno. Questo metodo è utilizzato per trovare i topic in un dataset.

    Considerando il grafo dei documenti, assumiamo che un topic sia dato da un insieme di documenti
    che hanno una certa densità di link verso altri documenti del topic (in confronto ai
    documenti esterni al topic).

    Vengono restituite solo le community che hanno più di members_threshold membri.
    """
    communities_generator = community.louvain_partitions(G)
    next_level_communities = next(communities_generator)
    comms = sorted(map(sorted, next_level_communities))
    comms = [comm for comm in comms if len(comm) > members_threshold]
    return comms


def terms_filter(terms):
    """
    Funzione per filtrare i termini che non sono verbi, aggettivi, avverbi o altro. Viene utilizzata
    per ottenere le parole chiave di un topic. 

    Nota: eseguiamo nuovamente il filtraggio di certi termini con WordNet perché alle volte alcuni termini
    superano il precedente filtraggio eseguito con Spacy.
    """
    sw = get_stop_words('it')
    filtered = []
    for term in terms:
        if (not any(word.pos() in ['v', 'a', 'r', 'x'] for word in wn.synsets(term, lang='ita')) and term not in sw):
            filtered.append(term)
    return filtered


def topic_characterization(corpus_terms, communities, corpus_dict, idx2id, num_kw=15):
    """
    Funzione che riceve:
    - corpus in forma di lista di termini, 
    - la lista con le comunità,
    - il dizionario impiegato per la rappresentazione del corpus, 
    - un dizionario per convertire gli indici dei documenti verso i corrispondenti id nel dataset, 
    - il numero di parole chiave da utilizzare nella rappresentazione.
    
    Restituisce una lista di coppie parole chiavi e post corrispondenti (terms, post) per ogni comunità.
    """

    corpus_comm = []

    for comm in communities:
        corpus_comm.append(reduce(lambda x, y: x + y, [corpus_terms[idx] for idx in comm]))

    corpus_comm_bow = [corpus_dict.doc2bow(terms) for terms in corpus_comm]
    corpus_comm_tfidf = tfidf_representation(corpus_comm_bow)

    corpus_kw = []
    for comm_terms, comm in zip(corpus_comm_tfidf, communities):
        terms = [corpus_dict[term[0]] for term in sorted(comm_terms, key=lambda x: x[1], reverse=True)]
        terms = terms_filter(terms)
        corpus_kw.append({'terms': terms[:num_kw], 'posts': [idx2id[id] for id in comm]})

    return corpus_kw


if __name__ == '__main__':
    
    dataset_path = processed_dataset_file()
    dataset = load_dataset(dataset_path)

    idx2id = {idx: id for idx, id in enumerate(dataset.keys())}

    corpus_tokens = [value['tokens'].split(' ') for value in dataset.values()]
    corpus_dict = gs.corpora.Dictionary(corpus_tokens)
    corpus_bow = [corpus_dict.doc2bow(tokens) for tokens in corpus_tokens]

    corpus_tfidf = tfidf_representation(corpus_bow)
    
    corpus_lsi_mtx = lsi_representation(corpus_tfidf, 80)
    
    G = docs_to_graph(corpus_lsi_mtx, 0.5)

    commns = communities_from_graph(G)

    commns_char = topic_characterization(corpus_tokens, commns, corpus_dict, idx2id)

    dataset_path = topics_file()
    store_dataset(commns_char, dataset_path)






