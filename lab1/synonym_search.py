import sys
import logging
from nltk.corpus import wordnet

# Import shared components from assignment.py (CHANGED)
from assignment import (
    LOG, DocumentIndex, QueryResult, Document, # Added Document for type hints if needed elsewhere
    extract_normalized_tokens, initialize_data, build_index, perform_search
)

def expand_query_synonyms(q_tokens: list[str]) -> set[str]:
    """Expands query tokens using WordNet synonyms."""
    expanded_set = set(q_tokens) # Start with original tokens
    for token in q_tokens:
        # wordnet.synonyms returns list of lists, flatten it
        syn_lists = wordnet.synonyms(token)
        synonyms = {syn.lower() for syn_list in syn_lists for syn in syn_list if '_' not in syn} # Flatten and clean
        expanded_set.update(synonyms)
    LOG.debug(f"Synonym expanded tokens: {expanded_set}")
    return expanded_set

def search_with_synonyms(query: str, index: DocumentIndex) -> list[QueryResult]:
    """Processes query, expands with synonyms, and searches the index."""
    normalized_tokens = extract_normalized_tokens(query)
    if not normalized_tokens:
        return []
    expanded_tokens = expand_query_synonyms(normalized_tokens)
    return perform_search(expanded_tokens, index)


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # LOG.setLevel(logging.DEBUG)

    documents_data, questions_data = initialize_data()
    index = build_index(documents_data)

    print("\n--- Running ALL Queries through SYNONYM Expansion ---") # Changed title
    correct_count = 0
    total_queries_run = 0 # Renamed for clarity

    # Iterate through ALL queries
    for q_dict in questions_data:
        # REMOVED: if q_dict.get('method') != 'synonyms': continue

        total_queries_run += 1
        try:
            question_text = q_dict["question"]
            expected_doc_id_str = q_dict["doc"]
            original_method = q_dict.get("method", "N/A") # Keep track of original intent

            # Indicate which method is being USED, and the original intent
            print(f"\nQUERY (Using Synonyms): '{question_text}' (Expected Doc ID: {expected_doc_id_str}, Originally for: {original_method})")

            results = search_with_synonyms(question_text, index)

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
        print("\n--- Synonym Search Metrics (All Queries) ---") # Changed title
        print(f"Total Queries Run: {total_queries_run}")
        print(f"Correct @ Rank 1: {correct_count}")
        print(f"Accuracy @ Rank 1: {accuracy:.2f}%")
    else:
        print("\nNo queries found in the data.") # Should not happen if data loads

    print("\n--- Synonym Search Finished ---")

