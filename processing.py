# %%
import json
import os
import spacy
from gensim.parsing import preprocessing
import re
from resources import caption_patterns_to_filter, processed_dataset_file, raw_datasets_folder, spacy_dict
from tqdm import tqdm
from stop_words import get_stop_words


nlp = spacy.load(spacy_dict)


def load_raw_dataset(datasets_folder_path: str) -> dict:
    """
    Funzione che carica tutti i file json relativi ad un giornale, e crea un unico dataset.

    Restituisce un dizionario che ha come chiavi i nomi dei 
    giornali e come valori l'insieme dei post e i relativi commenti.
    """
    dataset = {}
    for file in os.listdir(datasets_folder_path):
        if file.endswith(".json"):
            with open(os.path.join(datasets_folder_path, file)) as f:
                key, value = list(json.load(f).items())[0]
                dataset[key] = value
    return dataset

def pivot_dataset(dataset: dict) -> dict:
    """
    Funzione che effettua il pivot su un dizionario. Cambia la chiave, facendola diventare l'id del post,
    la chiave precedente sarà inserita come attributo.

    Restituisce il nuovo dizionario indicizzato per l'id del post.
    """
    posts = {}
    for key, value in dataset.items():
        for post in value:
            post = post.copy()
            post['profile'] = key
            id = post.pop('id')
            posts[id] = post
    return posts

def extract_corpus(dataset: dict) -> dict:
    """
    Funzione che estrae il contenuto testuale dei post (caption) dal dataset.

    Restituisce un dizionario che ha come chiavi l'id dei post e come valori le caption.
    """
    return {id: post['caption'].lower() for id, post in dataset.items()}  


def get_hashtags(raw_caption: str) -> list:
    """
    Funzione che restituisce una lista degli hashtag individuati all'interno di una stringa.
    """
    hashtags = []
    for word in raw_caption.split():
        if word.startswith('#'):
            hashtags.append(word)
    hashtags = [preprocessing.strip_punctuation(tag).strip().lower() for tag in hashtags]
    return hashtags

def get_citations(raw_caption: str) -> list:
    """
    Funzione che restituisce una lista di citazioni individuate all'interno di una stringa.
    """
    citations = []
    for word in raw_caption.split():
        if word.startswith('@'):
            citations.append(word)
    citations = [preprocessing.strip_punctuation(tag).strip().lower() for tag in citations]
    return citations

def filter_recurring_pattern(corpus: dict, terms: list) -> dict:
    """
    Funzione che riceve un dizionario come corpus e una lista di termini da filtrare nel caption.

    Restituisce un dizionario come corpus con i caption filtrati dei pattern da rimuovere. 
    """
    corpus = corpus.copy()
    for id, caption in corpus.items():
        corpus[id] = re.sub(r'\b' + '|'.join(terms) + r'\b', '', caption)
    return corpus


def process_caption(caption: str) -> str:
    """
    Funzione che riceve una stringa (caption di un post) e la filtra togliendo:
    - punteggiatura,
    - simboli,
    - spazi,
    - stopword,
    - carattere '#',
    - caratteri che iniziano con @, ovvero le citazioni (poi verranno aggiunte successivamente),
    - caratteri non alfanumerici.
    
    Inoltre, si effettua la lemmatizzazione sui token che non sono nomi propri e nomi comuni.

    Restituisce una stringa che contiene i token validi.

    """
    doc = nlp(caption)
    result = []
    citations = get_citations(caption)

    sw = get_stop_words('it')

    for token in doc:

        pos_to_remove = ['PUNCT', 'DET', 'ADP', 'NUM', 'SPACE', 'SYM']

        if not (token.pos_ in pos_to_remove 
            or token.is_stop 
            or token.lemma_ == '#' 
            or token.lemma_.startswith('@')):

            if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
                to_filter = token.text.lower()
            else:
                to_filter = token.lemma_.lower()

            result.append(re.sub(r'[^A-zÀ-ú0-9 ]', '', to_filter).strip())

        
    return ' '.join([elem.lower() for elem in (result + citations) if elem != '' and elem not in sw])


def process_corpus(raw_corpus: dict) -> dict:
    """
    Funzione che riceve un dataset contenente caption non processati.

    Restituisce un dizionario come dataset che ha per chiavi l'id dei post e come valore i caption processati 
    utilizzando la funzione `process_caption`.
    """
    return {id: process_caption(caption) for id, caption in tqdm(raw_corpus.items())}


def store_pipeline_dataset(datasets_folder_path: str, out_file_path: str) -> None:
    """
    Funzione che esegue gli step necessari per processare i dati e renderli utili al fine dell'analisi.

    Gli step che esegue sono:
    - caricamento dei dati grezzi,
    - pivot della chiave del dizionario,
    - estrazione dei corpus,
    - filtraggio di pattern ricorrenti,
    - filtraggio di stopwords, etc...
    - estrazione citazioni,
    - estrazione hashtags,
    - estrazione dei tokens.

    Infine, si salva su un file json il dizionario di partenza con 
    l'aggiunta dei campi hashtags, citations e tokens.
    """
    
    raw_dataset = load_raw_dataset(datasets_folder_path)
    raw_dataset = pivot_dataset(raw_dataset)

    raw_corpus = extract_corpus(raw_dataset)
    raw_corpus = filter_recurring_pattern(raw_corpus, caption_patterns_to_filter)
    proc_corpus = process_corpus(raw_corpus)

    
    for id, post in raw_dataset.items():
        post['hashtags'] = get_hashtags(post['caption'])
        post['citations'] = get_citations(post['caption'])
        post['tokens'] = proc_corpus[id]
        raw_dataset[id] = post
        
    with open(out_file_path, 'w') as f:
        json.dump(raw_dataset, f)


if __name__ == '__main__':
    store_pipeline_dataset(raw_datasets_folder(), processed_dataset_file())



        


