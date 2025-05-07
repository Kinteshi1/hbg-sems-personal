# assignment.py
import logging
from csv import DictReader
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet # Keep wordnet import here for get_lemma
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Constants and Setup ---
WN = WordNetLemmatizer()
LOG = logging.getLogger(__name__)
try:
    STOPWORDS = stopwords.words("english")
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    STOPWORDS = stopwords.words("english")

PUNCTUATIONS = ",.:!?-"

class Document:
    def __init__(self, id: int, title: str, text: str):
        self.id = id
        self.title = title
        self.text = text
        self.normalized_tokens = []

    def __repr__(self) -> str:
        # Shorter representation for cleaner output
        return f"Doc(id={self.id}, title='{self.title}')"


class QueryResult:
    def __init__(self, doc: Document, hit: str):
        self.document = doc
        self.hitlist = hit # Store the first hit token
        self.weight = 1    # Start weight at 1

    def add_hit(self, hit: str):
        # Only add unique hits to the list for clarity, increment weight always
        if hit not in self.hitlist.split():
             self.hitlist += " " + hit
        self.weight += 1

    def __repr__(self) -> str:
        # Include title and a snippet of the text in representation
        text_snippet_length = 120 # Adjust as needed
        doc_text = self.document.text.replace("\n", " ") # Replace newlines for cleaner one-line snippet
        if len(doc_text) > text_snippet_length:
            text_snippet = doc_text[:text_snippet_length] + "..."
        else:
            text_snippet = doc_text

        return (f"QueryResult(weight={self.weight}, docid={self.document.id}, "
                f"title='{self.document.title}', hits='{self.hitlist}', "
                f"text='{text_snippet}')")

# --- Indexing Class ---
class DocumentIndex:
    def __init__(self):
        self.index: list[Document] = [] # List to store Document objects

    def add_to_index(self, doc: Document) -> None:
        # Normalize tokens when adding the document
        doc.normalized_tokens = extract_normalized_tokens(doc.text)
        self.index.append(doc)
        LOG.debug(f"Indexed {doc}, tokens: {doc.normalized_tokens}")


    def find_token(self, token: str) -> list[Document]:
        """Finds documents containing the exact normalized token."""
        result = [doc for doc in self.index if token in doc.normalized_tokens]
        LOG.debug(f"Token '{token}' found in docs: {[d.id for d in result]}")
        return result

    def get_doc_by_id(self, doc_id: int) -> Document | None:
        """Helper to get a document object by its ID."""
        for doc in self.index:
            if doc.id == doc_id:
                return doc
        return None


# --- Token Processing Functions ---
def extract_normalized_tokens(text: str) -> list[str]:
    """Extracts, cleans, and lemmatizes tokens from text."""
    tokens = extract_relevant_tokens(text)
    return lemmatize_tokens(tokens)


def extract_relevant_tokens(text: str) -> list[str]:
    """Tokenizes and removes stopwords and punctuation."""
    try:
        tok = word_tokenize(text)
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Please ensure setup is complete.")
        return []
    return [w.lower() for w in tok if w.lower() not in STOPWORDS and w not in PUNCTUATIONS and len(w) > 1]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatizes tokens using WordNet based on POS tags."""
    try:
        postags = nltk.pos_tag(tokens)
    except LookupError:
        print("NLTK 'averaged_perceptron_tagger' not found. Please ensure setup is complete.")
        return tokens # Return unlemmatized tokens if tagger fails
    result = [get_lemma(tag) for tag in postags]
    return result


def get_lemma(entry: tuple[str, str]) -> str:
    """Converts NLTK POS tag to WordNet format and lemmatizes."""
    token = entry[0]
    pos = entry[1]
    wn_pos = None
    if pos.startswith("J"): # Adjective
        wn_pos = wordnet.ADJ
    elif pos.startswith("R"): # Adverb (often treated as adjective in WordNet lemmatization)
        wn_pos = wordnet.ADV # or wordnet.ADJ
    elif pos.startswith("N"): # Noun
        wn_pos = wordnet.NOUN
    elif pos.startswith("V"): # Verb
        wn_pos = wordnet.VERB

    # Use the WN lemmatizer instance defined at the top
    return WN.lemmatize(token, wn_pos) if wn_pos else token # Lemmatize if POS is mapped


# --- Data Initialization ---
def initialize_data() -> tuple[list[dict], list[dict]]:
    """Loads document and question data."""
    # Document Data (Countries/Sections)
    doc_lines = [
        "id;title;text",
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
    documents_data = list(DictReader(doc_lines, delimiter=';', skipinitialspace=True))

    # Question Data (Synonyms and Word2Vec)
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
    questions_data = list(DictReader(question_lines, delimiter=';', skipinitialspace=True))

    return documents_data, questions_data

# --- Helper for Building Index ---
def build_index(documents_data: list[dict]) -> DocumentIndex:
    """Creates and populates the document index."""
    index = DocumentIndex()
    LOG.info("Building document index...")
    for doc_dict in documents_data:
        try:
            # Use the Document class defined above
            doc_instance = Document(
                id=int(doc_dict['id']),
                title=str(doc_dict['title']),
                text=str(doc_dict['text'])
            )
            index.add_to_index(doc_instance)
        except (KeyError, ValueError) as e:
            LOG.error(f"Skipping invalid document data: {doc_dict}. Error: {e}")
        except Exception as e:
             LOG.error(f"Unexpected error processing document: {doc_dict}. Error: {type(e).__name__}: {e}")
    LOG.info(f"Finished indexing {len(index.index)} documents.")
    return index

# --- Helper for Performing Search ---
def perform_search(expanded_tokens: set[str], index: DocumentIndex) -> list[QueryResult]:
    """Performs the core search logic using expanded tokens."""
    result_map = {} # Use dict for efficient aggregation: doc_id -> QueryResult
    for token in expanded_tokens:
        # Use the find_token method from DocumentIndex class
        docs_found = index.find_token(token)
        for doc in docs_found:
            if doc.id not in result_map:
                # Use the QueryResult class defined above
                result_map[doc.id] = QueryResult(doc, token)
            else:
                result_map[doc.id].add_hit(token)

    # Convert map values to list and sort by weight
    result_list = list(result_map.values())
    return sorted(result_list, key=lambda qr: qr.weight, reverse=True)

if __name__ == "__main__":
    print("This file contains common utilities and data. Run synonym_search.py or word2vec_search.py instead.")
    # You could add basic tests here if desired
    logging.basicConfig(level=logging.INFO)
    docs, quests = initialize_data()
    print(f"Loaded {len(docs)} docs, {len(quests)} questions.")
    idx = build_index(docs)
    print(f"Index built with {len(idx.index)} documents.")
