import os
import numpy as np
import argparse
import av
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def extract_time(response):
    # Extract the time spans
    # _, times = response.split(": ", 1)
    start_time_str, end_time_str = response.split(" to ")

    # Define the time format
    time_format = "%H:%M:%S"

    # Parse the time strings into datetime objects
    start_time = datetime.strptime(start_time_str, time_format)
    end_time = datetime.strptime(end_time_str, time_format)

    # Convert to seconds
    start_time_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    end_time_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
    return [start_time_seconds, end_time_seconds]

# Arguments parsing
parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
parser.add_argument("--dataset_path", type=str, help="Path to the Huggingface dataset repo")
parser.add_argument("--anet_vid_dir", type=str, help="Path to the ActivityNet raw video directory")
parser.add_argument("--qvh_vid_dir", type=str, help="Path to the QVHighlights raw video directory")
args = parser.parse_args()

# Load the dataset
rextime_data = load_dataset(args.dataset_path, split="validation")
input_data = rextime_data[0]

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Input video configuration
if input_data['source'] == "qvhighlights_val":
    video_path = os.path.join(args.qvh_vid_dir, input_data['vid'] + '.mp4')
else:
    video_path = os.path.join(args.anet_vid_dir, input_data['vid'] + '.mp4')
container = av.open(video_path)

# Sample uniformly 8 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

### Moment retrieval
### We can use the model to predict the time span for the answer.
prompt = f"USER: <video>Please find a relevant span to answer in seconds to answer the question in the following format: From <start_time> to <end_time>. Quesetion: {input_data['question']} ASSISTANT: "
inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(device)

out = model.generate(**inputs, max_new_tokens=40)
repsonse = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print("Response: ", repsonse[0])
pred_time = extract_time(repsonse[0].split("ASSISTANT: ")[1])
print("Predicted Time: ", pred_time)

### Question Answering
### We can use the model to predict the answer to the question from the 4 options.
prompt = f"USER: <video>{input_data['question']} ASSISTANT: "
prompts = []
for option in input_data['options']:
    # You can replace the <s0>, <e0> with the predicted values if available to evaluation grounding VQA.
    sentence = input_data['answer'].replace("<s0>", str(input_data['span'][0]))
    sentence = sentence.replace("<e0>", str(input_data['span'][1]))
    sentence = sentence.replace("<option>", option)
    prompts.append(prompt + sentence)

# Preprocess the text and video
clips = [clip for _ in range(len(prompts))]
inputs = processor(text=prompts, videos=clips, return_tensors="pt", padding=True).to(device)

# Inference to get the logits
with torch.no_grad():
    output = model(**inputs)

# Alternatively, you can also feed the labels to the model to compute the loss as scores.
preds = []
for opt in range(len(prompts)):
    # 6 is the length of input_ids which is ahead of the image tokens, here we only consider the scores after the image tokens.
    sentence_length = inputs['input_ids'][opt].shape[0] - 6
    scores = tuple([i.unsqueeze(0) for i in output['logits'][opt]][-sentence_length-1:-1])
    output_scores = model.compute_transition_scores(inputs['input_ids'][opt].unsqueeze(0)[:, 6:], scores, normalize_logits=True)
    score = sum(output_scores[0].cpu().tolist()) / len(output_scores[0])
    preds.append(score)

# Output the result
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
pred_ans  = mapping[preds.index(max(preds))]
print("Predicted Answer: ", pred_ans)

output = {
    "qid": input_data['qid'],
    "pred_relevant_windows": [pred_time],
    "ans": pred_ans
}
print(output)
