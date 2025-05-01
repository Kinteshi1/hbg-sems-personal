import sys
import logging
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

WN = WordNetLemmatizer()
LOG = logging.getLogger(__name__)
STOPWORDS = stopwords.words("english")
PUNCTUATIONS = ",.:!?-"


class Document:

    def __init__(self, id: int, title: str, text: str):
        self.id = id
        self.title = title
        self.text = text
        self.normalized_tokens = []

    def __repr__(self) -> str:
        return f"Document(id={self.id} text={self.text})"


class QueryResult:

    def __init__(self, doc: Document, hit: str):
        self.document = doc
        self.hitlist = hit
        self.weight = 1

    def add_hit(self, hit: str):
        self.hitlist += " " + hit
        self.weight += 1

    def __repr__(self) -> str:
        return f"weight={self.weight}, docid={self.document.id}, hits={self.hitlist}, text={self.document.text}"


class DocumentIndex:

    def __init__(self):
        self.index = []

    def add_to_index(self, doc: Document) -> None:
        doc.normalized_tokens = (extract_normalized_tokens(doc.text))
        self.index.append(doc)

    def find_token(self, token: str) -> list[Document]:
        result = [doc for doc in self.index if token in doc.normalized_tokens]
        return result


class QueryProcessor:
    project_dir = ".."
    model = model = KeyedVectors.load(f"{project_dir}/model_cache/enwiki_20180420_100d.bin")

    def __init__(self, index: DocumentIndex):
        self.index = index

    @staticmethod
    def expand_query_tokens(q_tokens: list[str]) -> list[str]:
        result = set()
        for token in q_tokens:
            synsets = wordnet.synonyms(token)
            for syn in synsets:
                result.update([w.lower() for w in syn])
            result.add(token)
        return list(result)

    @staticmethod
    def is_comparable(token: str) -> bool:
        tags = nltk.pos_tag([token])
        if tags:
            tag = tags[0][1]
            return tag.startswith("N") or tag.startswith("R") or tag.startswith("J")
        return False

    @staticmethod
    def find_similar_tokens(query: list[str]) -> list[str]:
        tokens = extract_relevant_tokens(query)
        result = []
        for t in (t for t in tokens if QueryProcessor.is_comparable(t)):
            _s = QueryProcessor.model.most_similar(positive=t, topn=5)
            simset = [s[0] for s in _s]
            lemmas = lemmatize_tokens(simset)
            result.extend(lemmas)
        LOG.debug(f"similar tokens found: {result}")
        return result

    def query(self, query: str) -> list[QueryResult]:
        tokens = extract_normalized_tokens(query)
        tokens = QueryProcessor.expand_query_tokens(tokens)
        tokens = set(tokens)
        tokens.update(QueryProcessor.find_similar_tokens(query))
        result = []
        for token in tokens:
            docs = index.find_token(token)
            for doc in docs:
                lst = list(filter(lambda qr: qr.document == doc, result))
                if lst:
                    lst[0].add_hit(token)
                else:
                    result.append(QueryResult(doc, token))
        return sorted(result, key=lambda qr: qr.weight, reverse=True)


def extract_normalized_tokens(text: str) -> list[str]:
    tokens = extract_relevant_tokens(text)
    return lemmatize_tokens(tokens)


def extract_relevant_tokens(text: str) -> list[str]:
    tok = word_tokenize(text)
    return [w.lower() for w in tok if (w not in STOPWORDS and w not in PUNCTUATIONS)]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    postags = []
    for token in tokens:
        p = nltk.pos_tag([token])
        postags.append(p[0])
    result = [get_lemma(e) for e in postags]
    return result


def get_lemma(entry: tuple[str, str]) -> str:
    pos = entry[1]
    pos1 = None
    if pos.startswith("J") or pos.startswith("R"):
        pos1 = "a"
    elif pos.startswith("N"):
        pos1 = "n"
    elif pos.startswith("V"):
        pos1 = "v"
    return WN.lemmatize(entry[0], pos1) if pos1 else entry[0]


def initialize_data(documents: list[Document], questions: list[dict]) -> None:
    from csv import DictReader
    doc_lines = [
    "id;title;text",  # Header row
    "1;South Korea - History;Kings Yeongjo and Jeongjo particularly led a new renaissance of the Joseon dynasty during the 18th century.",
    "2;South Korea - Geography;South Korea has 20 national parks and popular nature places like the Boseong Tea Fields, Suncheon Bay Ecological Park, and Jirisan.",
    "3;South Korea - Climate;Spring usually lasts from late March to early May, summer from mid-May to early September, autumn from mid-September to early November, and winter from mid-November to mid-March.",
    "4;South Korea - Biodiversity;It is acknowledged that many of these difficulties are a result of South Korea's proximity to China, which is a major air polluter.",
    "5;South Korea - Government and Politics;However, some political experts has argued that South Korea has been experiencing democratic backsliding and the reemergence of authoritarianism, particularly under the presidency of Yoon Suk Yeol, which culminated when he declared martial law for the first time since the 1980 military coup d\'état after the assassination of dictator Park Chung Hee, and the first since democratization in 1987.",
    "6;New Zealand - History;Some Māori later migrated to the Chatham Islands where they developed their distinct Moriori culture; a later 1835 invasion by Māori iwi resulted in the massacre and virtual extinction of the Moriori.",
    "7;New Zealand - Geography;Zealand is located near the centre of the water hemisphere and is made up of two main islands and more than 700 smaller islands.",
    "8;New Zealand - Climate;New Zealand's climate is predominantly temperate maritime (Köppen: Cfb), with mean annual temperatures ranging from 10 °C (50 °F) in the south to 16 °C (61 °F) in the north.",
    "9;New Zealand - Biodiversity;New Zealand's geographic isolation for 80 million years and island biogeography has influenced evolution of the country's species of animals, fungi and plants.",
    "10;New Zealand - Government and Politics;New Zealand is a constitutional monarchy with a parliamentary democracy, although its constitution is not codified.",
    "11;Iceland - History;According to both Landnámabók and Íslendingabók, monks known as the Papar lived in Iceland before Scandinavian settlers arrived, possibly members of a Hiberno-Scottish mission.",
    "12;Iceland - Geography;Iceland is at the juncture of the North Atlantic and Arctic Oceans.",
    "13;Iceland - Climate;The warm North Atlantic Current ensures generally higher annual temperatures than in most places of similar latitude in the world.",
    "14;Iceland - Biodiversity;The entire country is in a single ecoregion, the Iceland boreal birch forests and alpine tundra.",
    "15;Iceland - Government and Politics;Iceland is a representative democracy and a parliamentary republic.",
    "16;Canada - History;The first inhabitants of North America are generally hypothesized to have migrated from Siberia by way of the Bering land bridge and arrived at least 14,000 years ago.",
    "17;Canada - Geography;By total area (including its waters), Canada is the second-largest country.",
    "18;Canada - Climate;Average winter and summer high temperatures across Canada vary from region to region.",
    "19;Canada - Biodiversity;Canada is divided into 15 terrestrial and five marine ecozones.",
    "20;Canada - Government and Politics;Canada is described as a full democracy, with a tradition of liberalism, and an egalitarian, moderate political ideology.",
    "21;Brazil - History;Some of the earliest human remains found in the Americas, Luzia Woman, were found in the area of Pedro Leopoldo, Minas Gerais and provide evidence of human habitation going back at least 11,000 years.",
    "22;Brazil - Geography;Brazil occupies a large area along the eastern coast of South America and includes much of the continent's interior, sharing land borders with Uruguay to the south; Argentina and Paraguay to the southwest; Bolivia and Peru to the west; Colombia to the northwest; and Venezuela, Guyana, Suriname and France (French overseas region of French Guiana) to the north.",
    "23;Brazil - Climate;The climate of Brazil comprises a wide range of weather conditions across a large area and varied topography, but most of the country is tropical.",
    "24;Brazil - Biodiversity;The wildlife of Brazil comprises all naturally occurring animals, plants, and fungi in the South American country.",
    "25;Brazil - Government and Politics;The form of government is a democratic federative republic, with a presidential system."
    ]
    documents.extend(list(DictReader(doc_lines, delimiter=';', skipinitialspace=True)))

    question_lines = [
        "question;doc;method",
        "Who ruled during Korea's renaissance?;1;word2vec",
        "Tell me about South Korea's seasonal changes.;3;word2vec",
        "Describe the fate of the Moriori people.;6;word2vec",
        "How warm is New Zealand usually?;8;synonyms",
        "Where is Iceland located relative to major bodies of water?;12;word2vec",
        "What makes Iceland warmer than expected?;13;word2vec",
        "How did the first people arrive in the Americas according to theory?;16;word2vec",
        "What is Canada's size ranking?;17;synonyms",
        "What is the political system in Canada like?;20;synonyms",
        "What ancient human remains were discovered in Brazil?;21;word2vec",
        "Describe Brazil's general weather conditions.;23;word2vec"
    ]
    questions.extend(list(DictReader(question_lines, delimiter=';', skipinitialspace=True)))


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout)
    LOG.setLevel(logging.DEBUG) # Kept as DEBUG as per original working example

    documents = [] # Holds the raw dictionaries from DictReader
    queries = []   # Holds the raw query dictionaries from DictReader
    # Assuming initialize_data populates 'documents' and 'queries'
    # with lists of dictionaries from DictReader
    initialize_data(documents, queries)
    print(f"read {len(documents)} documents and {len(queries)} queries")

    index = DocumentIndex()
    # Iterate through the dictionaries read from the data
    for doc_dict in documents: # Changed loop variable name for clarity
        try:
            # --- FIX IS HERE ---
            # Create the Document instance with explicit type conversion for id
            doc_instance = Document(
                id=int(doc_dict['id']),      # Convert id string to int
                title=str(doc_dict['title']), # Ensure title is string
                text=str(doc_dict['text'])    # Ensure text is string
            )
            # Add the correctly typed Document object to the index
            index.add_to_index(doc_instance)
            # --- END FIX ---
        except KeyError as e:
            # Basic error logging matching original style if needed
            LOG.error(f"Missing key {e} in document data: {doc_dict}")
        except ValueError as e:
            LOG.error(f"Cannot convert id to int for document data: {doc_dict}. Error: {e}")
        except Exception as e:
             LOG.error(f"Unexpected error processing document: {doc_dict}. Error: {type(e).__name__}: {e}")

    # --- Check Model Path ---
    # Ensure this path is correct relative to where you RUN the script.
    try:
        query_processor = QueryProcessor(index)
    except Exception as e:
        LOG.error(f"Error initializing QueryProcessor (check model path?): {type(e).__name__}: {e}")
        sys.exit(1) # Exit if processor fails to load


    # Iterate through the query dictionaries
    for q in queries:
        try:
            result = query_processor.query(q["question"])
            # result=[Document(**doc) for doc in documents[0:2]] # Original commented line kept
            print(f"QUERY={q['question']}") # Using original key access
            for qr in result[0:2]:
                print(f"\t{qr}")
        except KeyError as e:
             LOG.error(f"Missing key {e} in query data: {q}")
        except Exception as e:
             LOG.error(f"Unexpected error processing query: {q}. Error: {type(e).__name__}: {e}")

