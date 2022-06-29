# Progetto Web Data Analytics

L'obiettivo di questo progetto è di raccogliere post Instagram da profili che trattano notizie e attualità, con i relativi commenti. Dopodiché eseguire una topic detection sui post, per svolgere successivamente emotion e sentiment classification sui commenti. Abbiamo individuato i temi che riscontrano un maggiore interesse e per ciascuno di essi abbiamo ricavato la distribuzione delle opinioni da parte dei commentatori. Abbiamo anche messo in evidenza le differenti reazioni che uno stesso argomento suscita in base al pubblico di riferimento.

## Librerie utilizzate

Le librerie utilizzate sono:

- [`numpy`](https://www.numpy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`scipy`](https://www.scipy.org/)
- [`networkx`](https://networkx.org/)
- [`pyfunctional`](https://pypi.org/project/pyfunctional/)
- [`tqmd`](https://github.com/tqdm/tqdm)
- [`json`](https://docs.python.org/3/library/json.html)
- [`re`](https://docs.python.org/3/library/re.html)
- [`feel-it`](https://huggingface.co/MilaNLProc/feel-it-italian-emotion)
- [`spacy`](https://spacy.io/)
- [`gensim`](https://radimrehurek.com/gensim/)
- [`nltk`](https://www.nltk.org/)
- [`stop-words`](https://pypi.org/project/stop-words/)
- [`instaloader`](https://instaloader.github.io/)
- [`termcolor`](https://pypi.org/project/termcolor/)

Le librerie grafiche per la visualizzazione dei dati sono:

- [`plotly`](https://plot.ly/)
- [`matplotlib`](https://matplotlib.org/)


## Resources

Nel file `resources.py` vengono dichiarate variabili che contengono le risorse utilizzate nel progetto. Come per esempio:

- la lista dei profili dal quale estrarre i dati,
- percorsi per salvare e leggere i dati,
- variabili temporali utili per lo scraping,
- patterns presenti nei post o nei commenti da filtrare o eliminare.

## Scraping 

Lo scraping dei post di Instagram viene fatto a partire da una lista di profili di nostro interesse, contenuti in `profiles`, restituisce tutti i loro post a partire da `hours_offset` ore fa fino a `days_back` giorni indietro.

Per evitare che Instagram blocchi lo scraping, il numero massimo di commenti scaricabili per un singolo post è di `max_comments`.

La variabile `user` contiene l'username dell'utente Instagram che sta effettuando lo scraping. Prima di procedere occorre digitare da terminale `instaloader -l USERNAME` e procedere al login.
 
*NOTA*: per ogni profilo viene realizzato e memorizzato un file, per rendere agevole la ripresa dello scraping in caso di blocco del profilo da parte di Instagram, in una fase intermedia.

`raw_datasets_folder` è il percorso dove verranno salvati i file json dei dataset. In caso di file già presenti, il loro contenuto viene aggiornato.

## Processing

In `processing.py` sono contenute tutte le funzioni utili per il processing del testo di ogni post. Con queste funzioni è possibile:

- caricare i file json ottenuti dallo scraping e unirli in un unico dataset,
- fare il pivot del dataset sull'id del post,
- estrarre il testo da ogni post,
- estrarre gli hashtag da ogni post,
- estrarre le citazioni da ogni post,
- estrarre i token significativi.

Vengono utilizzate le librerie `re`, `stop_words`, `spacy` e `gensim`.

## Topic detection 

In `topic_detection.py` sono presenti le funzioni necessarie per la topic detection. Si procede con il caricamento del dataset processato in precedenza. Successivamente si definiscono il dizionario, la rappresentazione TD-IDF e la rappresentazione LSI, da quest'ultima si considerano solo 80 dimensioni.

Per ottenere i topic, si rappresentano gli elementi dello spazio semantico attraverso un grafo, tra un post ed un altro è presente un link se la loro similarità coseno supera una certa soglia. 

Successivamente, si va alla ricerca delle comunità all'interno del grafo ottenuto. Infine, per ogni comunità, si calcola nuovamente il TF-IDF per determinare i termini più importanti all'interno di essa, dai quali otteniamo le parole chiave per rappresentare il topic.

Vengono utilizzate le librerie `gensim`, `nltk`, `stop_words` e `networkx`.

## Comments

In `comments.py` vi solo le funzioni necessarie per:

- l'estrazione dei commenti dal dataset grezzo,
- il filtraggio di commenti in base ai pattern definiti,
- la sentiment e l'emotion analysis.

Vengono le utilizzate le librerie `spacy` e `feel-it`.

## Users community

In `users_community.py` si va alla ricerca di similarità tra i commentatori dei i vari topic.

Dato il dataset dei topic e il dataset dei commenti analizzati, andiamo a ricavare il grafo, dove ogni nodo rappresenta un topic, vi è un link tra i due topic se le reazioni dei commentatori sono simili.

Si crea una matrice utente/topic, che per ovvie ragioni è molto grande e sparsa, quindi effettuiamo `svds`ottenendo le matrici `u`, `s` e `v`, di cui manteniamo le prime 10 dimensioni.

Successivamente, calcoliamo la similarità coseno a partire dalla matrice `v` per individuare i link da inserire all'interno del grafo. Infine, visualizziamo il grafo.

Vengono utilizzate le librerie `pandas`, `networkx`, `numpy` e `scipy`.


## Data exploration

Nel notebook `data_exploration.ipynb` è presente tutta la pipeline da eseguire. Inoltre, se si vuole evitare di eseguire le operazioni onerose (scraping, processing dei post, community detection, processing e l'analisi dei commenti) si può settare `presentation_mode` a `True` e visualizzare quindi i risultati utilizzando i dati precedentemente ricavati, memorizzati all'interno di file json. 

I plot interattivi presenti sono:

- network dei post (rispetto alla similarità),
- network dei topic (rispetto ai commentatori),
- hot topics (rispetto al numero dei commenti),
- sentiment per topic,
- emotion per topic,
- sentiment per topic (rispetto al profilo di pubblicazione).

Infine facciamo un piccolo esempio del calcolo di precision e recall su un determinato topic.