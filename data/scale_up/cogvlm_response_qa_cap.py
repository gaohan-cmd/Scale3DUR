import os
import random

import requests
import json
import base64

base_url = "http://127.0.0.1:8000"


# base_url = "http://10.3.242.146:8000"


def create_chat_completion(model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):

    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        if use_stream:
            # Handling streaming responses
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # Handling non-streaming responses
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")

            print(content)
            return content
    else:
        print("Error:", response.status_code)
        return None


def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def simple_image_chat(use_stream=True, img_path=None):
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The image displays a wooden boardwalk extending through a vibrant green grassy wetland. The sky is partly cloudy with soft, wispy clouds, indicating nice weather. Vegetation is seen on either side of the boardwalk, and trees are present in the background, suggesting that this area might be a natural reserve or park designed for ecological preservation and outdoor recreation. The boardwalk allows visitors to explore the area without disturbing the natural habitat.",
        },
        {
            "role": "user",
            "content": "Do you think this is a spring or winter photo?"
        },
    ]
    create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)


def generate_qa(use_stream=True, img_path=None):
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    text = f"""
    For a question and answer on this picture, the question is to ask the spatial relationship between objects or the properties of objects. The answer language must be concise and conform to the content of the picture. 
    Next, a question and answer is performed on the image, only one question and one answer are generated, and the output content is strictly in the following format {{"answers": ["answer content"], "object_names": ["object name"], "question": "question content"}}
    
    All objects in the picture object_id correspond to the following:
    
    1 wall
    2nd floor
    3 cabinets
    4 beds
    5 chairs
    6 sofas
    7 tables
    8 doors
    9 windows
    10 bookshelves
    11 pictures
    12 counters
    13 blinds
    14 desks
    15 shelves
    16 curtains
    17 dressers
    18 pillows
    19 mirrors
    20 floor mats
    21 clothes
    22 ceilings
    23 books
    24 refridgerator
    25 television
    26 papers
    27 towels
    28 shower curtains
    29 boxes
    30 whiteboards
    31 persons
    32 nightstands
    33 toilets
    34 sinks
    35 lamps
    36 bathtubs
    37 bags
    38 otherstructures
    39 otherfurniture
    40 otherprops
    
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        }
    ]
    return create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)


def generate_description(use_stream=True, img_path=None):
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    text = f"""
    You need to provide a summary of the scene. The summary should be about object types, object properties, relative positions between objects. Also describe the scene about common sense, for example, how objects are used by humans and human activities in the scene. The description should conform to the given scene information. You do not need to describe every object in the scene, choose some objects in the scene to summarize. You can also summarize the function, style and comfort of the room based on the arrangement and color of objects in the room. Your summary should not exceed 110 words.

    Examples are as follows:
    User input: a picture; model output: {{"description": "The television is positioned over the cupboard. Its color is dark.", "object_name": "tv"}}
    
    All objects in the picture object_id correspond to the following:

    1 wall
    2nd floor
    3 cabinets
    4 beds
    5 chairs
    6 sofas
    7 tables
    8 doors
    9 windows
    10 bookshelves
    11 pictures
    12 counters
    13 blinds
    14 desks
    15 shelves
    16 curtains
    17 dressers
    18 pillows
    19 mirrors
    20 floor mats
    21 clothes
    22 ceilings
    23 books
    24 refridgerator
    25 television
    26 papers
    27 towels
    28 shower curtains
    29 boxes
    30 whiteboards
    31 persons
    32 nightstands
    33 toilets
    34 sinks
    35 lamps
    36 bathtubs
    37 bags
    38 otherstructures
    39 otherfurniture
    40 otherprops

    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        }
    ]
    return create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)


if __name__ == "__main__":
    # Get a list of all unique scene names by extracting the first 12 characters of each filename
    data_dir = "/data/scannet/scannet_data"
    scanqa_frames_dir = "/data/ScanQA/frames_square"
    
    all_scan_names = list(
        set(
            os.path.basename(x)[:12]
            for x in os.listdir(data_dir)
            if x.startswith("scene")
        )
    )

    # Process each scan name
    for scan_name in all_scan_names:
        # Get the folder path for images associated with the current scene
        image_folder = os.path.join(scanqa_frames_dir, scan_name, "color")
        
        # Get all image paths from the folder that end with .jpg or .png
        image_paths = [
            os.path.join(image_folder, file) 
            for file in os.listdir(image_folder) 
            if file.endswith(('.jpg', '.png'))
        ]
        
        if not image_paths:
            print(f"No images found for scene {scan_name}. Skipping this scene.")
            continue  # Skip to the next scene if no images are found

        # Randomly select 4 images from the folder
        random_images = random.sample(image_paths, min(4, len(image_paths)))  # In case there are less than 4 images

        # Dictionary to store the results for the current scene
        result = []
        scene_result = {scan_name: []}

        # Generate descriptions for the selected images
        for image_path in random_images:
            print(f"Processing image: {image_path}")
            description = generate_description(use_stream=False, img_path=image_path)
            result.append(description)

        # Add descriptions to the scene result dictionary
        scene_result[scan_name] = result

        # Write the scene results to the description.json file
        with open("description.json", "a") as f:
            json.dump(scene_result, f, indent=2)
            f.write(",\n")

        print(f"Descriptions for scene {scan_name} have been written to 'description.json'.")


