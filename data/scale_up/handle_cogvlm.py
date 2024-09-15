import json
import re


# Filters and processes JSON data by adjusting object fields
def filter_and_process_json(input_file, output_file):
    try:
        # Step 1: Read the original JSON file
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Step 2: Filter objects where 'object_ids' is not empty and 'object_names' is a list
        filtered_data = [
            obj for obj in data
            if obj.get('object_ids') and isinstance(obj.get('object_names'), list)
        ]

        # Step 3: Process each object by ensuring 'answers' is a list
        for obj in filtered_data:
            if not isinstance(obj['answers'], list):
                obj['answers'] = [obj['answers']]

        # Step 4: Write the processed data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        print(f"Filtered and processed data successfully saved to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_file}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Flattens nested lists into a single list
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


# Handles and processes generated QA data
def handle_data():
    object_name_to_id = {
        "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5, "sofa": 6, "table": 7,
        "door": 8, "window": 9, "bookshelf": 10, "picture": 11, "counter": 12, "blinds": 13,
        "desk": 14, "shelves": 15, "curtain": 16, "dresser": 17, "pillow": 18, "mirror": 19,
        "floor mat": 20, "clothes": 21, "ceiling": 22, "books": 23, "refrigerator": 24,
        "television": 25, "paper": 26, "towel": 27, "shower curtain": 28, "box": 29,
        "whiteboard": 30, "person": 31, "nightstand": 32, "toilet": 33, "sink": 34, "lamp": 35,
        "bathtub": 36, "bag": 37, "otherstructure": 38, "otherfurniture": 39, "otherprop": 40
    }

    # Read the maximum question_id for each scene
    with open('ScanQA_v1.0_train.json', 'r') as f:
        data = json.load(f)
    
    # Dictionary to store the maximum question_id for each scene_id
    max_question_ids = {}

    # Iterate over JSON data
    for item in data:
        scene_id = item['scene_id']
        question_id = int(item['question_id'].split('-')[-1])

        # Update the maximum question_id for the scene
        max_question_ids[scene_id] = max(max_question_ids.get(scene_id, -1), question_id)

    # Load filtered data for processing
    with open('filtered_data.json', 'r') as f:
        cogdata = json.load(f)

    for item in cogdata:
        scene_name = list(item.keys())[0]
        scene_id = scene_name.split('_')[0]

        if scene_name not in max_question_ids:
            max_question_ids[scene_name] = -1

        for obj in item[scene_name]:
            obj['scene_id'] = scene_name
            obj['question_id'] = f"train-{scene_id}-{max_question_ids[scene_name] + 1}"
            max_question_ids[scene_name] += 1

            # Process object names and IDs
            new_object_ids = []
            object_names = []
            for obj_item in obj["object_names"]:
                if isinstance(obj_item, list):
                    new_object_ids.append([object_name_to_id.get(name, 39) for name in obj_item])
                    object_names.append([name for name in obj["object_names"]])
                else:
                    new_object_ids.append(object_name_to_id.get(obj_item, 39))
                    object_names.append(obj_item)

            obj["object_ids"] = flatten(new_object_ids)
            obj["object_names"] = flatten(object_names)

            # Write processed data to the new JSON file
            with open('result_cogvlm_data-new.json', 'a') as f:
                json.dump(obj, f, indent=2)
                f.write(',\n')


# Handles and processes generated descriptive data
def handle_data_des():
    object_name_to_id = {
        "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5, "sofa": 6, "table": 7,
        "door": 8, "window": 9, "bookshelf": 10, "picture": 11, "counter": 12, "blinds": 13,
        "desk": 14, "shelves": 15, "curtain": 16, "dresser": 17, "pillow": 18, "mirror": 19,
        "floor mat": 20, "clothes": 21, "ceiling": 22, "books": 23, "refrigerator": 24,
        "television": 25, "paper": 26, "towel": 27, "shower curtain": 28, "box": 29,
        "whiteboard": 30, "person": 31, "nightstand": 32, "toilet": 33, "sink": 34, "lamp": 35,
        "bathtub": 36, "bag": 37, "otherstructure": 38, "otherfurniture": 39, "otherprop": 40
    }

    # Read descriptive data
    with open('filtered_data_des.json', 'r') as f:
        cogdata = json.load(f)

    for item in cogdata:
        scene_name = list(item.keys())[0]

        for obj in item[scene_name]:
            obj['scene_id'] = scene_name

            # Process object names and IDs
            new_object_ids = []
            object_names = []
            for obj_item in obj["object_name"]:
                if isinstance(obj_item, list):
                    new_object_ids.append([object_name_to_id.get(name, 39) for name in obj_item])
                    object_names.append([name for name in obj["object_name"]])
                else:
                    new_object_ids.append(object_name_to_id.get(obj_item, 39))
                    object_names.append(obj_item)

            obj["object_id"] = flatten(new_object_ids)[0]
            obj["object_name"] = flatten(object_names)[0]

            # Tokenize and extract description
            obj["token"] = re.findall(r'\w+|[^\w\s]', obj["description"][0])
            obj['description'] = obj['description'][0]

            # Write processed data to the new JSON file
            with open('result_cogvlm_data_des.json', 'a') as f:
                json.dump(obj, f, indent=2)
                f.write(',\n')


if __name__ == '__main__':
    # filter_and_process_json('description.json', 'filtered_data.json')
    handle_data_des()
