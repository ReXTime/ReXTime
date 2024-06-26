import os, json, time
import numpy as np
import cv2
import base64

from datasets import load_dataset
from openai import OpenAI

def decode_video(video_path, total_frames=50):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    # Get duration and fps
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    video.release()
    
    selected_ids = np.linspace(0, len(base64Frames) - 1, total_frames, dtype=np.int32)
    selected_frames = [base64Frames[id] for id in selected_ids]
    
    return selected_frames, duration

def extract_json_from_string(str):
    json_str = str.split('```json')[1].strip().split('```')[0].strip()
    clean_json_string = '\n'.join([line.split('//')[0].strip() for line in json_str.splitlines()])
    output = json.loads(clean_json_string)
    return output

def replace_all(text, replace_dict):
    for key, value in replace_dict.items():
        text = text.replace(key, value)
    return text

# Load the dataset
dataset_path = "<Path to the Huggingface dataset repo>"
anet_vid_dir = '<Path to the ActivityNet raw video directory>'
qvh_vid_dir = '<Path to the QVHighlights raw video directory>'

rextime_data = load_dataset(dataset_path, split="validation")
input_data = rextime_data[0]

# Configure the GPT chatbot
organization = "<Your organization ID>"
api_key = "<Your API key>"
client = OpenAI(api_key=api_key, organization=organization)

# Decode the video frames
if input_data['source'] == "qvhighlights_val":
    video_path = os.path.join(qvh_vid_dir, input_data['vid'] + '.mp4')
else:
    video_path = os.path.join(anet_vid_dir, input_data['vid'] + '.mp4')
frames, duration = decode_video(video_path)
print("Total duration {}, {} frames selected".format(duration, len(frames)))

# Prepare the prompt
prompt = '''
You are tasked to complete the following logical reasoning task.
1. Watch and breifly summarize the video.
2. Given 50 video frames and a question, choose the most correct answer from the options.
3. Find the time span in seconds that support your answer. The time span should be consistent with the option you choose.

Total Video Duration: <DURATION> seconds
Question: <QUESTION>
Options:
A. <Option A>
B. <Option B>
C. <Option C>
D. <Option D>

<Provide your video summarize here.>
<Provide your explaination to the question here.>
Finish the following json structure strictly with your answer and supporting time span according your explaination.
```json{
    "relevant_span": <[start_time, end_time], in seconds, e.g. [3.0, 10.5]>,
    "answer": <Option you choice from "A", "B", "C", "D".>
}```
'''

replace_dict = {
    "<QUESTION>": input_data['question'],
    "<Option A>": input_data['options'][0],
    "<Option B>": input_data['options'][1],
    "<Option C>": input_data['options'][2],
    "<Option D>": input_data['options'][3],
    "<DURATION>": str(input_data['duration'])
}

prompt = replace_all(prompt, replace_dict)

# Prepare the chatbot parameters
PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            prompt,
            *map(lambda x: {"image": x, "resize": 534}, frames),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 1000,
    "temperature": 0.7,
}

# Send the request to the chatbot
Done, invalid = False, False
while not Done:
    try:
        result = client.chat.completions.create(**params)
        Done = True
    except Exception as e:
        print(f"Waiting 1 second...")
        print(e, type(e))
        if "invalid_request_error" in str(e):
            print("Invalid request error, skipping...")
            invalid = True
            break
        time.sleep(0.01)

# Extract the answer and price
answer = result.choices[0].message.content
answer_json = extract_json_from_string(answer)
ans = answer_json['answer']
prompt_tokens = result.usage.prompt_tokens
completion_tokens = result.usage.completion_tokens
price = (prompt_tokens * 0.005 + completion_tokens * 0.015) / 1000
print("Predicted Answer: ", ans)
print(
    f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_price={price}"
)
