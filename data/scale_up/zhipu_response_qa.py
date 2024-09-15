# -*- coding: utf-8 -*-
import json
import os
from json import JSONDecodeError
from time import sleep
import re
from zhipuai import ZhipuAI
from get_qa_ids import get_random_qa, get_qa



# Generating similar questions for the ScanQA dataset
import json
from time import sleep
from ZhipuAI import ZhipuAI  # Assuming ZhipuAI is an external library

def qa_generation():
    client = ZhipuAI(api_key="f94203c69adf74d440fc3a5a65805889.K5aW3WcP4HMkkJSl")
    
    # Fetch Q&A data (replace with actual function to retrieve Q&A)
    # qas = get_random_qa()
    qas = get_qa()

    # Loop through the Q&A data for each scene
    for qa in qas:
        result = {}
        for scene_id, qa_list in qa.items():
            new_qa_items = []
            for qa_item in qa_list:
                new_qa_item = {}
                # Construct the content to send to the API
                content = "I am giving you a Q&A pair, including 'answers' and 'question' parts. The question part is: " + str(
                    qa_item['question']) + " and the answers part is: " + str(qa_item['answers']) + """
                    Help me rewrite the question part according to the following requirements:
                    (1. Keep it as concise as possible
                    2. The question part should not consist of multiple sentences
                    3. The meaning must remain unchanged)
                    The rewrites may include, but are not limited to, synonym replacement, grammatical adjustments, or rephrasing.
                    The final question must be in English, and it must be returned in the following format: ["Rewritten question"].
                    Here's an example: 'What is in the right corner of room by curtains?' was rewritten as 'What is in the right corner of the room near the curtains?'.
                    """
                
                # Send the request to the API
                response = client.chat.completions.create(
                    model="glm-4-0520",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant proficient in both Chinese and English sentence rewriting. Please strictly follow the user's instructions."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.2,
                    max_tokens=128
                )
                
                # Extract the rewritten question from the API response
                s = response.choices[0].message.content
                start_index = s.find('[')
                end_index = s.rfind(']')
                rewritten_question = s[start_index + 2:end_index - 1]  # Extract the content inside the brackets
                
                # Construct a new Q&A item
                new_qa_item['question'] = rewritten_question
                new_qa_item['answers'] = qa_item['answers']
                new_qa_item['object_names'] = qa_item['object_names']
                new_qa_item['scene_id'] = scene_id
                new_qa_item['object_ids'] = qa_item['object_ids']
                
                # Append the new Q&A item to the list
                new_qa_items.append(new_qa_item)

                # Optional delay
                # sleep(5)
        
        # Store the result for the current scene
        result[scene_id] = new_qa_items
        
        # Write the result to a JSON file
        with open('new_qa_data.json', 'a') as f:
            json.dump(result, f, indent=2)
            f.write(',\n')
        
        print(f"Q&A pairs for scene {scene_id} have been generated.")
    
    print("All data has been successfully appended to 'new_qa_data_xsp_result_0728.json'.")


# Add question_id to generated Q&A pairs
def add_question_id():
    # Load the newly generated Q&A data
    with open('new_qa_data.json', 'r') as f:
        new_qa_data = json.load(f)
    
    # Load the existing training data to get the current max question_id for each scene
    with open('ScanQA_v1.0_train.json', 'r') as f:
        data = json.load(f)
    
    # Dictionary to store the maximum question_id for each scene_id
    max_question_ids = {}

    # Iterate over the existing training data to populate max_question_ids
    for item in data:
        scene_id = item['scene_id']
        question_id = item['question_id']

        # Extract the numeric part of question_id and update the maximum value
        if scene_id in max_question_ids:
            max_question_ids[scene_id] = max(max_question_ids[scene_id], int(question_id.split('-')[-1]))
        else:
            max_question_ids[scene_id] = int(question_id.split('-')[-1])
    
    # Add question_id to the new Q&A pairs and increment it for each new pair
    for new_qa_data_item in new_qa_data:
        scene_id = list(new_qa_data_item.keys())[0]

        # Ensure that each scene_id has a starting question_id, defaulting to -1 if not found
        if scene_id not in max_question_ids:
            max_question_ids[scene_id] = -1

        # Loop through each Q&A item in the scene and assign a new question_id
        for qa_item in new_qa_data_item[scene_id]:
            # Generate a new question_id by incrementing the last used question_id
            qa_item['question_id'] = f"train-{scene_id.split('_')[0]}-{max_question_ids[scene_id] + 1}"
            max_question_ids[scene_id] += 1
            
            # Append the updated Q&A data with question_id to the output file
            with open('result_qa_data.json', 'a') as f:
                json.dump(qa_item, f, indent=2)
                f.write(',\n')


# Generate a question and answer for each description in a scene
def captioning_generation_item(scene_id):
    # Load the existing data to find the maximum question_id for each scene
    with open('ScanQA_v1.0_train.json', 'r') as f:
        data = json.load(f)
    
    # Dictionary to store the maximum question_id for each scene_id
    max_question_ids = {}

    # Iterate over the JSON data to populate max_question_ids
    for item in data:
        current_scene_id = item['scene_id']
        question_id = item['question_id']
        
        # Update the max question_id for each scene
        if current_scene_id in max_question_ids:
            max_question_ids[current_scene_id] = max(max_question_ids[current_scene_id], int(question_id.split('-')[-1]))
        else:
            max_question_ids[current_scene_id] = int(question_id.split('-')[-1])

    # Initialize the ZhipuAI client with the API key
    client = ZhipuAI(api_key="")
    
    # Fetch the scene description and object information for the given scene_id
    scene_data = json.loads(get_random_description()).get(scene_id)

    if scene_data is not None:
        descriptions = scene_data.get('descriptions')
        objects = scene_data.get('object_info')

        for i in range(len(descriptions)):
            description = descriptions[i]
            obj = objects[i]
            object_name = obj['object_name']
            object_id = obj['object_id']

            # Skip processing if object_name contains an underscore
            if '_' in object_name:
                continue

            # Prepare the content to send to the API
            response = client.chat.completions.create(
                model="glm-4",  # Specify the model to use
                messages=[
                    {"role": "system", "content": "You are an AI assistant proficient in both Chinese and English dialogue. Please strictly follow the user's required format for generating data."},
                    {"role": "user", 
                     "content": f"I am giving you a scene description and object information related to it. Please generate one concise question and one answer about this scene. "
                                f"The answer should only provide location, no subject or verb is needed, and object names should not have articles. "
                                f"The answer must start with a lowercase letter and everything must be in English. The response must follow this format: "
                                f'[{{"question": "question_content", "answers": ["answer_content"], "object_names": ["object_names"]}}]. '
                                f"Here is the scene description: {description}, and the object information is: {object_name}."}
                ],
                temperature=0.2,
                max_tokens=128
            )

            # Extract the response content
            response_text = response.choices[0].message.content
            start_index = response_text.find('[')
            end_index = response_text.rfind(']')
            # Extract the JSON-like substring from the response
            json_str = response_text[start_index:end_index + 1].replace('”', '"').replace('“', '"').replace('，', ',')

            if json_str:
                try:
                    # Parse the extracted JSON string
                    data = json.loads(json_str)

                    # Check if scene_id exists in max_question_ids, if not set it to 0
                    if scene_id not in max_question_ids:
                        max_question_ids[scene_id] = 0

                    # Add additional fields like question_id, scene_id, object_name, and object_id
                    data_with_ids = add_ids(
                        data,
                        f'train-{scene_id.split("_")[0]}-{max_question_ids[scene_id] + 1}',
                        scene_id,
                        object_name,
                        object_id
                    )

                    # Increment the max_question_id for this scene
                    max_question_ids[scene_id] += 1

                except JSONDecodeError as e:
                    print(f"JSON decoding error: {e}. Skipping this iteration.")
                    continue  # Skip to the next iteration in case of a JSON parsing error

                # Convert the modified data back to JSON format with proper indentation
                json_data = json.dumps(data_with_ids, indent=2)
                json_data_without_brackets = json_data[1:-1]

                # Append the data to the output file
                with open('result_qa_data_description.json', 'a') as f:
                    # Add a comma at the end of each object for proper separation
                    json_data_without_brackets += ','
                    f.write(json_data_without_brackets)
                
                print(f"Q&A pairs for scene {scene_id} have been generated.")


# Generate questions and answers for all scene files
def captioning_generation():
    # Get a list of all unique scan names from the 'scannet_data' directory
    all_scan_names = list(
        set(
            [
                os.path.basename(x)[:12]  # Extract the first 12 characters of each filename
                for x in os.listdir(os.path.join("..", "scannet", "scannet_data"))  # List files in the directory
                if x.startswith("scene")  # Only process files that start with "scene"
            ]
        )
    )

    # Iterate through each scan name and generate Q&A pairs using get_zhipuai_response_4
    for scan_name in all_scan_names:
        get_zhipuai_response_4(scan_name)

    print("All data has been successfully appended to the file 'result_qa_data_description.json'.")


if __name__ == '__main__':
    get_zhipuai_response()
    #add_question_id()

