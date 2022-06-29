# %%
from typing import List
import instaloader
from credentials import USER
from datetime import datetime, timedelta
import json
from tqdm import tqdm
from termcolor import colored
import os
from resources import raw_datasets_folder, profiles, hours_offset, days_ahead


def scrape(profiles: List[str], raw_datasets_folder: str, user: str, 
           days_back: int = 1, hours_offset: int = 3, max_comments: int = 300):

    """
    Data una lista di profili Instagram di cui fare scraping, restituisce tutti i loro post
    a partire da "hours_offset" ore fa fino a "days_back" giorni indietro.

    Per evitare che Instagram blocchi lo scraping, il numero massimo di commenti scaricabili
    per un singolo post è di "max_comments".

    "user" contiene l'username dell'utente Instagram che sta effettuando lo scraping.
    Prima di procedere occorre digitare da terminale `instaloader -l USERNAME` e 
    procedere al login.

    NOTA: per ogni profilo viene realizzato e memorizzato un file, per rendere agevole
    la ripresa dello scraping in caso di blocco del profilo da parte di Instagram, in 
    una fase intermedia.

    "raw_datasets_folder" è il percorso dove verranno salvati i file json dei dataset.
    In caso di file già presenti, il loro contenuto viene aggiornato.

    """

    loader = instaloader.Instaloader() # type: ignore
    loader.load_session_from_file(user)  # `instaloader -l USERNAME`

    now = datetime.now()
    starting_time = now - timedelta(hours=hours_offset)
    ending_time = starting_time - timedelta(days=days_back)

    os.makedirs(raw_datasets_folder, exist_ok=True)

    for profile in profiles:

        dataset_path = os.path.join(raw_datasets_folder, f"{profile}.json")

        if os.path.exists(dataset_path):
            with open(dataset_path) as f:
                dataset = json.load(f)
        else:
            dataset = {profile: []}

        print(f"Scraping {colored(profile, 'green')}...")

        profile_content = instaloader.Profile.from_username( # type: ignore
            loader.context, profile)

        posts = []

        for i, post in enumerate(profile_content.get_posts()):

            if post.date > starting_time:
                continue

            if post.date < ending_time:
                break

            label = colored(post.caption[:20], 'red') # type: ignore

            print(f'- post n. {i} "{label}..."')

            post_dict = {'caption': post.caption,
                         'id': post.shortcode,
                         'date': str(post.date),
                         'likes': post.likes}

            comments_list = []

            for j, comm in tqdm(enumerate(post.get_comments())):
                comm_dict = {'text': comm.text,
                             'likes': comm.likes_count,
                             'owner': comm.owner.username,
                             'id': comm.id}

                comments_list.append(comm_dict)

                if j > max_comments:
                    break

            post_dict['comments'] = comments_list

            dataset[profile].append(post_dict)

        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)


if __name__ == '__main__':
    scrape(profiles, raw_datasets_folder(), USER, days_ahead, hours_offset)