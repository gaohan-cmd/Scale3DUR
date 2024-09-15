import json

from openai import OpenAI
import time

from get_description_ids import get_random_description


# Generate questions and answers for a given scene
def get_kimi_response(scene_id):
    # Initialize OpenAI client
    client = OpenAI(
        api_key="sk-xxxxxxxxxxx",
        base_url="https://api.moonshot.cn/v1",
    )

    # Load the scene description and object info
    scene_all = json.loads(get_random_description()).get(scene_id)
    if not scene_all:
        return None

    descriptions = scene_all.get('descriptions')
    objects = scene_all.get('object_info')

    # Construct the user prompt
    user_prompt = (
        "I will provide you with several descriptions of a scene and a list of objects along with their IDs. "
        "Please generate 3 questions and answers about the scene, ensuring they are logical, in English, and align with the scene's actual description. "
        "The response format must strictly follow: [{“question”: “question content”, “answers”: “answer content”, “object_ids”: [object_id], “object_names”: “object_names”}], "
        "where object_id is an integer (e.g., [1]). My scene descriptions are: "
        f"{descriptions}, and the object labels are: {objects}."
    )

    # API call to generate questions and answers
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Kimi, an AI assistant provided by Moonshot AI. You excel in both English and Chinese conversations. "
                    "You provide safe, helpful, and accurate responses, and refuse to answer any questions related to terrorism, "
                    "racial discrimination, pornography, or violence. The term 'Moonshot AI' is a proper noun and should not be translated."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
    )

    # Extract the relevant content from the API response
    response_content = completion.choices[0].message.content
    start_index = response_content.find('[')
    end_index = response_content.rfind(']')
    
    if start_index != -1 and end_index != -1:
        result = response_content[start_index:end_index + 1]
        time.sleep(4)  # Optional delay for rate-limiting or safety
        return result

    return None

if __name__ == '__main__':
    get_kimi_response("scene0000_00")
