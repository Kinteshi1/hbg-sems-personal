# word2vec_search.py

# ... (imports and function definitions remain the same) ...
import sys
import logging
import os
import nltk # Needed for pos_tag
from gensim.models import KeyedVectors

from assignment import (
    LOG, DocumentIndex, QueryResult, Document, # Added Document
    extract_normalized_tokens, extract_relevant_tokens, lemmatize_tokens,
    initialize_data, build_index, perform_search, get_lemma
)

# Define model path relative to this script's location
# Assumes script is in 'lab1' and model_cache is one level up
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_cache')
MODEL_FILE = os.path.join(MODEL_DIR, 'enwiki_20180420_100d.bin')

def load_word2vec_model(path: str) -> KeyedVectors | None:
    """Loads the Gensim KeyedVectors model."""
    LOG.info(f"Attempting to load Word2Vec model from: {path}")
    if not os.path.exists(path):
        LOG.error(f"Model file not found at {path}. Please check the path.")
        return None
    try:
        model = KeyedVectors.load(path)
        LOG.info("Word2Vec model loaded successfully.")
        return model
    except Exception as e:
        LOG.error(f"Error loading Word2Vec model: {type(e).__name__}: {e}")
        return None

def is_comparable(token: str) -> bool:
    """Checks if a token's POS tag suggests it's suitable for Word2Vec comparison."""
    try:
        tags = nltk.pos_tag([token])
    except LookupError:
         LOG.warning("NLTK tagger not found, cannot determine comparability for Word2Vec.")
         return False # Cannot compare if tagger is missing
    if tags:
        tag = tags[0][1]
        # Nouns (NN), Adjectives (JJ), Adverbs (RB), Verbs (VB) are often good candidates
        return tag.startswith("N") or tag.startswith("J") or tag.startswith("R") or tag.startswith("V")
    return False

def find_similar_tokens(tokens: list[str], model: KeyedVectors, top_n: int = 3) -> set[str]:
    """Finds similar tokens using the Word2Vec model."""
    similar_set = set()
    if not model:
        LOG.warning("Word2Vec model not loaded, cannot find similar tokens.")
        return similar_set

    comparable_tokens = [t for t in tokens if is_comparable(t)]
    LOG.debug(f"Comparable tokens for Word2Vec: {comparable_tokens}")

    for token in comparable_tokens:
        try:
            # Find most similar tokens (returns list of (word, score))
            similar_raw = model.most_similar(positive=[token], topn=top_n)
            # Extract just the words, lowercase them
            sim_words = {sim[0].lower() for sim in similar_raw if '_' not in sim[0]} # Clean underscores
            similar_set.update(sim_words)
        except KeyError:
            LOG.debug(f"Token '{token}' not found in Word2Vec vocabulary.")
        except Exception as e:
            LOG.error(f"Error finding similar tokens for '{token}': {type(e).__name__}: {e}")

    # Lemmatize the found similar tokens for consistency with index
    lemmatized_similar = set(lemmatize_tokens(list(similar_set)))
    LOG.debug(f"Found and lemmatized similar tokens: {lemmatized_similar}")
    return lemmatized_similar


def search_with_word2vec(query: str, index: DocumentIndex, model: KeyedVectors) -> list[QueryResult]:
    """Processes query, expands with Word2Vec, and searches the index."""
    normalized_tokens = extract_normalized_tokens(query)
    if not normalized_tokens:
        return []

    # Find similar tokens based on the *original* normalized query tokens
    similar_tokens = find_similar_tokens(normalized_tokens, model)

    # Combine original normalized tokens with the similar ones
    expanded_tokens = set(normalized_tokens)
    expanded_tokens.update(similar_tokens)
    LOG.debug(f"Word2Vec combined expanded tokens: {expanded_tokens}")

    return perform_search(expanded_tokens, index)


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # LOG.setLevel(logging.DEBUG)

    # Load Word2Vec Model first
    w2v_model = load_word2vec_model(MODEL_FILE)
    if not w2v_model:
        sys.exit(1) # Exit if model loading fails

    documents_data, questions_data = initialize_data()
    index = build_index(documents_data)

    print("\n--- Running ALL Queries through WORD2VEC Expansion ---") # Changed title
    correct_count = 0
    total_queries_run = 0 # Renamed for clarity

    # Iterate through ALL queries
    for q_dict in questions_data:
        # REMOVED: if q_dict.get('method') != 'word2vec': continue

        total_queries_run += 1
        try:
            question_text = q_dict["question"]
            expected_doc_id_str = q_dict["doc"]
            original_method = q_dict.get("method", "N/A") # Keep track of original intent

            # Indicate which method is being USED, and the original intent
            print(f"\nQUERY (Using Word2Vec): '{question_text}' (Expected Doc ID: {expected_doc_id_str}, Originally for: {original_method})")

            results = search_with_word2vec(question_text, index, w2v_model)

            if not results:
                print("\tNo results found.")
            else:
                # Print top 2 results
                for i, qr in enumerate(results[0:2]):
                    print(f"\tRank {i+1}: {qr}")

                # --- Simple Metric Calculation ---
                top_result_doc_id_str = str(results[0].document.id)
                if top_result_doc_id_str == expected_doc_id_str:
                    print(f"\t-> CORRECT: Expected document found at Rank 1.")
                    correct_count += 1
                else:
                    print(f"\t-> INCORRECT: Expected Doc ID {expected_doc_id_str}, Got {top_result_doc_id_str} at Rank 1.")

        except KeyError as e:
            LOG.error(f"Skipping query due to missing key {e}: {q_dict}")
        except Exception as e:
            LOG.error(f"Error processing query: {q_dict}. Error: {type(e).__name__}: {e}")

    # --- Print Final Metrics ---
    if total_queries_run > 0:
        accuracy = (correct_count / total_queries_run) * 100
        print("\n--- Word2Vec Search Metrics (All Queries) ---") # Changed title
        print(f"Total Queries Run: {total_queries_run}")
        print(f"Correct @ Rank 1: {correct_count}")
        print(f"Accuracy @ Rank 1: {accuracy:.2f}%")
    else:
        print("\nNo queries found in the data.") # Should not happen if data loads

    print("\n--- Word2Vec Search Finished ---")

