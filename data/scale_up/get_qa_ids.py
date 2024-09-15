import json
import random

# Retrieve all question-answer pairs from ScanQA_v1.0_train.json
def get_qa():
    # Read the JSON file
    with open('ScanQA_v1.0_train.json', 'r') as f:
        data = json.load(f)
    # Dictionary to store data organized by scene_id
    organized_data = {}

    # Iterate over each JSON object
    for item in data:
        # Extract scene_id
        scene_id = item["scene_id"]
        # If scene_id is not in organized_data, create a new list for this scene
        if scene_id not in organized_data:
            organized_data[scene_id] = []
        # Append the current object to the corresponding scene_id list
        organized_data[scene_id].append({
            "answers": item["answers"],
            "object_names": item["object_names"],
            "question": item["question"],
            "question_id": item["question_id"],
            "object_ids": item["object_ids"]
        })

    # Convert organized_data to the specified format
    result = []
    for scene_id, data in organized_data.items():
        result.append({
            scene_id: [
                {
                    "answers": d["answers"],
                    "object_names": d["object_names"],
                    "question": d["question"],
                    "question_id": d["question_id"],
                    "object_ids": d["object_ids"]
                } for d in data
            ]
        })

    # Uncomment to print the result
    # print(json.dumps(result, indent=4))

    # Uncomment to save the result to a file
    # with open('organized_data.json', 'w') as file:
    #     json.dump(result, file, indent=4)

    return result

# Randomly select three question-answer pairs from train.json
def get_random_qa():
    json_data = get_qa()
    # Randomly select three QA objects for each scene_id
    random_data = []
    for scene_data in json_data:
        for scene_id, qa_list in scene_data.items():
            # If the number of QA objects is less than or equal to 10, select all
            if len(qa_list) <= 10:
                selected_qa = qa_list
            else:
                selected_qa = random.sample(qa_list, 10)  # Randomly select 10 QA objects
            random_data.append({scene_id: selected_qa})

    # Uncomment to save the randomly selected data to another file
    # with open("qa_random_data.json", "w") as outfile:
    #     json.dump(random_data, outfile, indent=2)

    print("Random selection and saving successful!")
    # Uncomment to print the first 4 entries of random_data
    # print(random_data[:4])
    # Uncomment to print the length of the QA list for a specific scene
    # print(len(random_data[2]['scene0002_00']))

    return random_data

if __name__ == '__main__':
    get_random_qa()
