def raw_datasets_folder(): return 'raw_datasets'
def processed_dataset_file(): return 'processed_datasets/pipeline_dataset.json'
def topics_file(): return 'processed_datasets/topics.json'
def analyzed_comments_file(): return 'processed_datasets/analyzed_comments.json'
def filtered_comments_file(): return 'processed_datasets/filtered_comments.json'

spacy_dict = "it_core_news_md"

profiles = ['il_post', 'larepubblica', 'will_ita', 'open_giornaleonline', 'la_stampa',
            'agenzia_ansa', 'corriere', 'tgcom24',  'skytg24', 'fanpage.it', 'ilmessaggero.it',  'ilsole_24ore']
hours_offset = 1
days_ahead = 1

comm_spam_patterns = r'|'.join([r'@gary_mary_fx'])
comm_patterns_to_remove = r'|'.join([r'@[\w]+'])

caption_patterns_to_filter = ['su repubblica', 'articolo completo', 'intervista completa di', '@repubblicaidee', 
'su open', 'su open.online', 'su la stampa', 'tutti gli aggiornamenti', 'intervista integrale', 'commento completo', 
'#ansa', 'ansa', 'leggi l.*articolo completo', 'sul corriere', 'del corriere', '#mediaset', '#news', '#tgcom24', 
'Per leggere .* a questo post', '@skytg24', '#skytg24', 'editoriale', 'link in bio', '#sole24ore', '#linkinbio',
'#notiziadelgiorno', 'fanpage.it', 'cronaca', '\\(✍️.*\\)', 'accadeoggi', 'losapeviche', '7corriere', 'tgcom24',
'tg24', '@il_post', '@larepubblica', '@will_ita', '@open_giornaleonline', '@la_stampa', '@agenzia_ansa', '@corriere',
'@tgcom24', '@skytg24', '@fanpage.it', '@ilmessaggero.it', '@ilsole_24ore', 'openonline', 'open.online']
