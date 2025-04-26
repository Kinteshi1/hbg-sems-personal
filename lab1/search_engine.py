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

    def __init__(self, index: DocumentIndex):
        self.index = index

    def query(self, query: str) -> list[QueryResult]:
        tokens = extract_normalized_tokens(query)
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
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk

    stops = stopwords.words("english")
    puncts = ",.:!?-"
    tok = word_tokenize(text)
    #tok = [w.lower() for w in tok if (w not in stops and w not in puncts)]
    tok = [w.lower() for w in tok if (w not in stops and w.isalpha())]
    postok = nltk.pos_tag(tok)
    tok = [get_lemma(e) for e in postok]
    return tok


from nltk.stem import WordNetLemmatizer

wn = WordNetLemmatizer()


def get_lemma(entry: tuple[str, str]) -> str:
    pos = entry[1]
    pos1 = None
    if pos.startswith("J") or pos.startswith("R"):
        pos1 = "a"
    elif pos.startswith("N"):
        pos1 = "n"
    elif pos.startswith("V"):
        pos1 = "v"
    return wn.lemmatize(entry[0], pos1) if pos1 else entry[0]


def initialize_data(documents: list[Document], questions: list[dict]) -> None:
    from csv import DictReader
    doc_lines = [
        "id;title;text",
        "1;Oldest Known Song; The oldest known song is the Hurrian Hymn No. 6, which dates back to around 1400 BCE.",
        "2;Bohemian Rhapsody; The iconic song Bohemian Rhapsody, released in 1975, is known for its unique structure and lack of a traditional chorus.",
        "3;Mozart's Early Start; Wolfgang Amadeus Mozart composed his first piece of music at the age of five.",
        "4;Eye of the Tiger; The iconic rock song Eye of the Tiger is known for its motivational lyrics and driving rhythm.",
        "5;Music and the Brain; Listening to music can stimulate the brain and improve memory, mood, and cognitive function.",
        "6;Universal Language; Music is often referred to as a universal language because it can convey emotions and stories without words.",
        "7;Largest Orchestra; The largest orchestra ever assembled consisted of 8,097 musicians and performed in Frankfurt, Germany, in 2019.",
        "8;Music Therapy; Music therapy is used to help patients with various conditions, including depression, anxiety, and chronic pain.",
        "9;Birds and Music; Some birds, like the lyrebird, can mimic musical instruments and human-made sounds.",
        "10;Quintet; A team of five people can form a band and create music as a quintet."
    ]
    documents.extend(list(DictReader(doc_lines, delimiter=';', skipinitialspace=True)))

    question_lines = [
        "question;doc;method",
        "what are very old songs;1;keyword-search",
        "what was the oldest vocal ever sung;1;synonyms",
        "can animals make music;9;meronyms",
        "what was the first song;1;word-vector-search",
        "can music bring me back to an active life;8;passage-retrieval",
        "can a five years old make music;3;passage-retrieval",
        "is there music about animals;4;passage-retrieval",
    ]
    questions.extend(list(DictReader(question_lines, delimiter=';', skipinitialspace=True)))


if __name__ == "__main__":

    documents = []
    queries = []
    initialize_data(documents, queries)
    print(f"read {len(documents)} documents and {len(queries)} queries")

    index = DocumentIndex()
    for doc in documents:
        index.add_to_index(Document(**doc))
    query_processor = QueryProcessor(index)

    for q in queries:
        result = query_processor.query(q["question"])
        # result=[Document(**doc) for doc in documents[0:2]]
        print(f"QUERY={q["question"]}")
        for qr in result[0:2]:
            print(f"\t{qr}")