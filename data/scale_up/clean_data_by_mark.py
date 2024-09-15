from cal_similar import semantic_search

import json

# Filter question-answer pairs based on semantic search scores to ensure logical consistency
def clean_data_semantic_search():
    # Read the original JSON data
    with open('result_cogvlm_data-new.json', 'r') as f:
        data = json.load(f)

    # Extract question and answers from each JSON object
    for obj in data:
        question = [obj['question']]
        answers = obj['answers']
        semantic_search_result = semantic_search(question, answers)

        # If the score is greater than 0.5, consider it a valid question-answer pair
        if semantic_search_result > 0.5:
            # Save the valid responses to a new JSON file
            with open('clean_data_semantic_search_threshold_0.5.json', 'a') as f:
                json.dump(obj, f, indent=2)
                f.write(',\n')


if __name__ == '__main__':
    clean_data_semantic_search()
    #test_clean_data_semantic_search()