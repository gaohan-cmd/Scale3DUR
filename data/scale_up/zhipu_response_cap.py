# -*- coding: utf-8 -*-
import json
import os
from json import JSONDecodeError
from time import sleep
import re

from zhipuai import ZhipuAI
from get_qa_ids import get_random_qa, get_qa
import random


# Generate new descriptions for dense subtitle tasks
def captioning_generation_item():
    # Load the original JSON data
    with open('json_result.json', 'r') as f:
        data = json.load(f)

    # Initialize ZhipuAI client
    client = ZhipuAI(api_key="")

    # Iterate over scenes and objects in the JSON data
    for scene_id, objects in data[0].items():
        for obj in objects:
            # Prepare the request content with scene descriptions and object name
            description = str(obj['descriptions'])
            object_name = str(obj['object_name'])
            content = (
                f"I'll give you a description for a scene, which includes two parts: "
                f"'description' and 'object_name'.\n"
                f"The 'description' part is: {description}.\n"
                f"The 'object_name' part is: {object_name}.\n"
                "Please rewrite the 'description' according to the following rules:\n"
                "1. You can use synonym substitution, modify sentence structure, adjust grammar, or reorder the information.\n"
                "2. The rewritten 'description' should be concise, logical, entirely in English, and should not start with a capital letter.\n"
                "3. Return the response in the following format: [response content].\n\n"
                "Here’s an example: 'there is a white two-door refrigerator. it has two white trash cans to its right.' "
                "can be rewritten as 'here is a white double-door refrigerator. to its right are two white garbage bins.'"
            )

            # Call ZhipuAI for the rewritten description
            response = client.chat.completions.create(
                model="glm-4-0520",
                messages=[
                    {"role": "system", "content": "You are an AI assistant proficient in both English and Chinese. Please strictly follow the user's instructions."},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                max_tokens=128
            )

            # Extract the response from the returned message
            response_content = response.choices[0].message.content
            start_index = response_content.find('[')
            end_index = response_content.rfind(']')
            if start_index != -1 and end_index != -1:
                rewritten_description = response_content[start_index + 1:end_index]

                # Construct the new QA item
                new_qa_item = {
                    'description': rewritten_description,
                    'object_name': obj['object_name'],
                    'scene_id': obj['scene_id'],
                    'object_id': obj['object_id'],
                    'ann_id': obj['ann_id'],
                    'token': re.findall(r'\w+|[^\w\s]', rewritten_description)
                }

                # Write the new QA item to the output file
                with open('new_description_data_all_scanRefer_descriptions.json', 'a') as f:
                    json.dump(new_qa_item, f, indent=2)
                    f.write(',\n')

        print(f"Description for scene {scene_id} generated for dense subtitle task.")    


# Generate new descriptions for dense subtitle tasks - Convert QA style to ScanRefer
def qa_generation():
    # Initialize ZhipuAI client
    client = ZhipuAI(api_key="")

    # Fetch QA data (use get_random_qa() or get_qa() as appropriate)
    qas = get_qa()

    # Iterate over QA data for each scene
    for qa in qas:
        for scene_id, qa_list in qa.items():
            new_qa_items = []
            for qa_item in qa_list:
                # Construct the content string with the QA pair
                question = str(qa_item['question'])
                answers = str(qa_item['answers'])
                content = (
                    f"I will give you a QA pair, consisting of 'answers' and 'question'.\n"
                    f"The 'question' is: {question}, and the 'answers' are: {answers}.\n"
                    "Please rewrite this QA pair into a single sentence, according to the following rules:\n"
                    "1. The meaning must remain the same.\n"
                    "2. The sentence must be logical and written entirely in English.\n"
                    "Please respond using the exact format: ['response content'].\n\n"
                    "Example:\n"
                    '"question": "What is in the right corner of room by curtains?", "answers": ["brown cabinet with tv sitting in it"]\n'
                    "Rewritten as: In the right corner of the room by the curtains, there is a brown cabinet with a TV sitting in it."
                )

                # Call ZhipuAI to generate the rewritten description
                response = client.chat.completions.create(
                    model="glm-4-0520",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant proficient in rewriting English and Chinese sentences. Please follow the user's instructions strictly."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.2,
                    max_tokens=128
                )

                # Extract the rewritten description from the response
                response_content = response.choices[0].message.content
                start_index = response_content.find('[')
                end_index = response_content.rfind(']')
                if start_index != -1 and end_index != -1:
                    rewritten_description = response_content[start_index + 2:end_index - 1]

                    # Construct new QA item
                    new_qa_item = {
                        'description': rewritten_description,
                        'object_name': qa_item['object_names'][0],
                        'scene_id': scene_id,
                        'object_id': qa_item['object_ids'][0],
                        'ann_id': str(random.randint(0, 5)),
                        'token': re.findall(r'\w+|[^\w\s]', rewritten_description)
                    }

                    # Write the new QA item to the file
                    with open('new_description_data_scanqa_to_scanRefer.json', 'a') as f:
                        json.dump(new_qa_item, f, indent=2)
                        f.write(',\n')

            print(f"Dense subtitles for scene {scene_id} have been generated.")

    print("All data has been successfully written to 'new_description_data_scanqa_to_scanRefer.json'.")


if __name__ == '__main__':
    qa_generation()

