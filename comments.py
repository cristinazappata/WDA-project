import re
import spacy_fastlang  # anche se il modulo non è utilizzato, non cancellarlo
import spacy

from typing import Iterable, List, Optional
from functional import seq
from tqdm import tqdm
from feel_it import EmotionClassifier, SentimentClassifier
from utils import load_dataset, store_dataset

from resources import analyzed_comments_file, comm_spam_patterns, comm_patterns_to_remove, processed_dataset_file, spacy_dict


def extract_comments(dataset: dict, posts_to_keep: Optional[Iterable[str]] = None) -> List[dict]:
    """
    Funzione per estrarre i commenti dal dataset, in base ai post da prendere in considerazione
    """
    comments_list = []

    if not posts_to_keep:
        posts_to_keep = dataset.keys()

    for id in posts_to_keep:
        content = dataset[id]
        post_comments = content['comments']
        for comment in post_comments:
            comment['post'] = id
            comments_list.append(comment)
    return comments_list


def filter_comments(comments: List[dict], spam_patterns: str, patterns_to_remove: str, with_progress: bool = False) -> List[dict]:
    """
    Funzione per rimuovere i commenti non opportuni e per filtrarne i contenuti nel loro testo.

    Vengono restituiti solo i commenti in italiano e che non contengono pattern di spam. 

    Inoltre vengono rimosse le occorrenze dei pattern di rimozione.
    """
    comments = comments.copy()

    nlp = spacy.load(spacy_dict)
    nlp.add_pipe("language_detector")

    def pattern_remove(comm: dict) -> dict:
        comm = comm.copy()
        comm['text'] = re.sub(patterns_to_remove, '', comm['text']).strip()
        return comm

    def progress(element, index, n_elements):
        print(f'{(index / n_elements) * 100}%', end='\r')
        return element

    n_comms = len(comments)

    res_comments = (seq(comments)
                    .enumerate()
                    .map(lambda x: progress(x[1], x[0], n_comms))
                    .filter(lambda comm: 'id' in comm)
                    .distinct_by(lambda comm: comm['id'])
                    .filter(lambda comm: not re.search(spam_patterns, comm['text']))
                    .map(pattern_remove)
                    .filter(lambda comm: comm['text'] != '')
                    .filter(lambda comm: nlp(comm['text'])._.language == 'it'))

    return res_comments.to_list()


def sentiment_emotion_analysis(comments: List[dict]) -> List[dict]:
    """""
    Funzione che esegue la sentiment e emotion analysis sui commenti estratti e filtrati.

    Vengono restituiti i valori di sentiment (positive o negative) e i valori di emotion
    (anger, joy, fear e sadness) per ogni commento analizzato.
    """

    emotion_classifier = EmotionClassifier()
    sentiment_classifier = SentimentClassifier()

    analys_comms = []
    for comment in tqdm(comments, desc='Sentiment analysis', unit='comments'):
        comment = comment.copy()
        comment['emotion'] = emotion_classifier.predict([comment['text']])[0]
        comment['sentiment'] = sentiment_classifier.predict([comment['text']])[
            0]
        analys_comms.append(comment)
    return analys_comms


if __name__ == '__main__':

    dataset = load_dataset(dataset_path=processed_dataset_file())
    comments = extract_comments(dataset)

    filtered_comments = filter_comments(
        comments, comm_spam_patterns, comm_patterns_to_remove, with_progress=True)

    analyzed_comments = sentiment_emotion_analysis(filtered_comments)
    store_dataset(analyzed_comments, analyzed_comments_file())
