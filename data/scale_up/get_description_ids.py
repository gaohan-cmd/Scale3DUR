import json
import random


def get_descriptions():
    # Read the JSON file
    with open('ScanRefer_filtered_train.json', 'r') as f:
        data = json.load(f)

    # Create an empty dictionary to store descriptions and object info for each scene
    scene_data = {}

    # Iterate over the data
    for item in data:
        scene_id = item['scene_id']
        description = item['description']

        object_info = {
            'object_id': item['object_id'],
            'object_name': item['object_name']
        }
        # Replace object name with formatted string if it exists in the description
        if object_info['object_name'] in description:
            description = description.replace(object_info['object_name'],
                                              object_info['object_name'] + "(object_id=" + object_info['object_id'] + ")")

        # If the scene ID already exists, append the description and object info to the corresponding lists;
        # otherwise, create a new entry
        if scene_id in scene_data:
            scene_data[scene_id]['descriptions'].append(description)
            scene_data[scene_id]['object_info'].append(object_info)
        else:
            scene_data[scene_id] = {
                'descriptions': [description],
                'object_info': [object_info]
            }
    # Output to JSON format
    # with open('json_result.json', 'w') as file:
    #     json.dump(scene_data, file, indent=4)
    return scene_data


def get_random_description():
    data = get_descriptions()
    # List to store the final result
    final_result = []

    # Process each scene
    for scene_key, scene_value in data.items():
        scene_result = {"scene": scene_key, "descriptions": [], "object_info": []}

        # Get the list of original descriptions and their indices
        original_descriptions = scene_value["descriptions"]
        original_object_info = scene_value["object_info"]
        description_indices = list(range(len(original_descriptions)))

        # Shuffle the description indices
        random.shuffle(description_indices)

        # Select the first 4 shuffled indices
        selected_indices = description_indices[:4]

        # Store the selected descriptions
        index_item = 1
        for index in selected_indices:
            description = original_descriptions[index]
            object_info = original_object_info[index]
            scene_result["descriptions"].append(str(index_item) + ":" + description)
            index_item += 1
            scene_result["object_info"].append(object_info)

        # Append the scene result to the final list
        final_result.append(scene_result)

    # Merge into a JSON object
    merged_json = {}
    for json_data in final_result:
        scene_key = json_data["scene"]
        merged_json[scene_key] = json_data

    # Convert to JSON string format
    merged_json_str = json.dumps(merged_json, indent=4)
    # Write the final result to a JSON file
    with open("json_random_result.json", "w") as json_file:
        json_file.write(merged_json_str)
    return merged_json_str


# Get all descriptions and objects for each scene - focus on dense subtitles issue
def get_all_scene():
    # Read the JSON file
    with open('ScanRefer_filtered_train.json', 'r') as f:
        data = json.load(f)

    # Create an empty dictionary to store descriptions and object info for each scene
    scene_data = {}

    # Iterate over the data
    for item in data:
        scene_id = item['scene_id']
        description = item['description']

        object_info = {
            'object_id': item['object_id'],
            'object_name': item['object_name']
        }
        descriptionsItem = {'descriptions': description, 'scene_id': scene_id, 'object_id': item['object_id'],
                            'object_name': item['object_name'], 'ann_id': item['ann_id']}
        # If the scene ID already exists, append the description and object info to the corresponding list;
        # otherwise, create a new entry
        if scene_id in scene_data:
            scene_data[scene_id].append(descriptionsItem)
        else:
            scene_data[scene_id] = [descriptionsItem]

    # Output to JSON format
    with open('json_result_per_scene_descriptions.json', 'w') as file:
        json.dump(scene_data, file, indent=4)


# Randomly select objects from each scene obtained in get_random_scene
def get_random_scene():
    data = json.load(open('json_result_per_scene_descriptions.json', 'r'))
    # Create a dictionary to store selected objects from each scene
    selected_objects = {}

    # Iterate over each scene
    for scene_id, objects in data[0].items():
        # Randomly select up to 15 objects
        selected = random.sample(objects, min(len(objects), 15))
        # Store the selected objects
        selected_objects[scene_id] = selected

    # Save the selected objects to a JSON file
    with open('json_result_per_scene_random_15.json', 'w') as file:
        file.write(json.dumps(selected_objects, indent=4))


# Randomly select 8 descriptions from previously generated data
def get_selected_scene():
    data = json.load(open('new_description_data_random_15_scanRefer.json', 'r'))
    # Group JSON objects by scene_id
    grouped_data = {}
    for obj in data:
        scene_id = obj['scene_id']
        if scene_id in grouped_data:
            grouped_data[scene_id].append(obj)
        else:
            grouped_data[scene_id] = [obj]

    # Select 8 JSON objects from each scene
    selected_data = []
    for scene_id, objects in grouped_data.items():
        selected_objects = random.sample(objects, min(len(objects), 8))
        selected_data.extend(selected_objects)

    # Write the new JSON data to a file
    with open('new_description_data_random_8_scanRefer.json', 'w') as f:
        json.dump(selected_data, f, indent=4)


if __name__ == '__main__':
    # get_random_description()
    # get_all_scene()
    # get_random_scene()
    get_selected_scene()
