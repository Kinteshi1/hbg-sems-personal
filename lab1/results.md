# Synonym Search Results with only Select Queries
/home/KInteshi/miniconda3/envs/hbg-sems-personal/bin/python /home/KInteshi/workspace/github.com/Kinteshi1/hbg-sems-personal/lab1/synonym_search.py
INFO:assignment:Building document index...
INFO:assignment:Finished indexing 25 documents.

--- Running Synonym Expansion Queries ---

QUERY (Synonyms): 'How warm is New Zealand usually?' (Expected Doc ID: 8)
	Rank 1: QueryResult(weight=2, docid=8, title='New Zealand - Climate', hits='new zealand')
	Rank 2: QueryResult(weight=2, docid=9, title='New Zealand - Biodiversity', hits='new zealand')
	-> CORRECT: Expected document found at Rank 1.

QUERY (Synonyms): 'What is Canada's size ranking?' (Expected Doc ID: 17)
	Rank 1: QueryResult(weight=2, docid=8, title='New Zealand - Climate', hits='range 's')
	Rank 2: QueryResult(weight=1, docid=23, title='Brazil - Climate', hits='range')
	-> INCORRECT: Expected Doc ID 17, Got 8 at Rank 1.

QUERY (Synonyms): 'What is the political system in Canada like?' (Expected Doc ID: 20)
	Rank 1: QueryResult(weight=2, docid=20, title='Canada - Government and Politics', hits='political canada')
	Rank 2: QueryResult(weight=1, docid=2, title='South Korea - Geography', hits='like')
	-> CORRECT: Expected document found at Rank 1.

## Synonym Search Metrics 
Total Queries: 3

Correct @ Rank 1: 2

Accuracy @ Rank 1: 66.67%

--- Synonym Search Finished ---

Process finished with exit code 0

# word2vect Search Results with only Select Queries
INFO:assignment:Attempting to load Word2Vec model from: /home/KInteshi/workspace/github.com/Kinteshi1/hbg-sems-personal/lab1/../model_cache/enwiki_20180420_100d.bin
INFO:gensim.utils:loading KeyedVectors object from /home/KInteshi/workspace/github.com/Kinteshi1/hbg-sems-personal/lab1/../model_cache/enwiki_20180420_100d.bin
INFO:gensim.utils:KeyedVectors lifecycle event {'fname': '/home/KInteshi/workspace/github.com/Kinteshi1/hbg-sems-personal/lab1/../model_cache/enwiki_20180420_100d.bin', 'datetime': '2025-05-01T12:19:15.091607', 'gensim': '4.3.3', 'python': '3.12.2 | packaged by conda-forge | (main, Feb 16 2024, 20:50:58) [GCC 12.3.0]', 'platform': 'Linux-6.14.4-300.fc42.x86_64-x86_64-with-glibc2.41', 'event': 'loaded'}
INFO:assignment:Word2Vec model loaded successfully.
INFO:assignment:Building document index...
INFO:assignment:Finished indexing 25 documents.

--- Running Word2Vec Expansion Queries ---

QUERY (Word2Vec): 'Who ruled during Korea's renaissance?' (Expected Doc ID: 1)
	Rank 1: QueryResult(weight=2, docid=4, title='South Korea - Biodiversity', hits=''s korea')
	Rank 2: QueryResult(weight=1, docid=8, title='New Zealand - Climate', hits=''s')
	-> INCORRECT: Expected Doc ID 1, Got 4 at Rank 1.

QUERY (Word2Vec): 'Tell me about South Korea's seasonal changes.' (Expected Doc ID: 3)
	Rank 1: QueryResult(weight=3, docid=4, title='South Korea - Biodiversity', hits='south 's korea')
	Rank 2: QueryResult(weight=3, docid=8, title='New Zealand - Climate', hits='south north 's')
	-> INCORRECT: Expected Doc ID 3, Got 4 at Rank 1.

QUERY (Word2Vec): 'Describe the fate of the Moriori people.' (Expected Doc ID: 6)
	Rank 1: QueryResult(weight=2, docid=6, title='New Zealand - History', hits='moriori māori')
	Rank 2: QueryResult(weight=1, docid=20, title='Canada - Government and Politics', hits='describe')
	-> CORRECT: Expected document found at Rank 1.

QUERY (Word2Vec): 'Where is Iceland located relative to major bodies of water?' (Expected Doc ID: 12)
	Rank 1: QueryResult(weight=2, docid=7, title='New Zealand - Geography', hits='locate water')
	Rank 2: QueryResult(weight=1, docid=4, title='South Korea - Biodiversity', hits='major')
	-> INCORRECT: Expected Doc ID 12, Got 7 at Rank 1.

QUERY (Word2Vec): 'What makes Iceland warmer than expected?' (Expected Doc ID: 13)
	Rank 1: QueryResult(weight=1, docid=7, title='New Zealand - Geography', hits='make')
	Rank 2: QueryResult(weight=1, docid=11, title='Iceland - History', hits='iceland')
	-> INCORRECT: Expected Doc ID 13, Got 7 at Rank 1.

QUERY (Word2Vec): 'How did the first people arrive in the Americas according to theory?' (Expected Doc ID: 16)
	Rank 1: QueryResult(weight=2, docid=11, title='Iceland - History', hits='accord arrive')
	Rank 2: QueryResult(weight=2, docid=16, title='Canada - History', hits='arrive first')
	-> INCORRECT: Expected Doc ID 16, Got 11 at Rank 1.

QUERY (Word2Vec): 'What ancient human remains were discovered in Brazil?' (Expected Doc ID: 21)
	Rank 1: QueryResult(weight=3, docid=21, title='Brazil - History', hits='human gerais remain')
	Rank 2: QueryResult(weight=2, docid=24, title='Brazil - Biodiversity', hits='brazil animal')
	-> CORRECT: Expected document found at Rank 1.

QUERY (Word2Vec): 'Describe Brazil's general weather conditions.' (Expected Doc ID: 23)
	Rank 1: QueryResult(weight=3, docid=23, title='Brazil - Climate', hits='brazil condition weather')
	Rank 2: QueryResult(weight=2, docid=22, title='Brazil - Geography', hits='brazil 's')
	-> CORRECT: Expected document found at Rank 1.

## Word2Vec Search Metrics 
Total Queries: 8

Correct @ Rank 1: 3

Accuracy @ Rank 1: 37.50%

--- Word2Vec Search Finished ---

Process finished with exit code 0
