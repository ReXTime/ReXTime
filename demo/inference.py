import os
import numpy as np
import av
import torch
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

# Load the dataset
dataset_path = "<Path to the Huggingface dataset repo>"
anet_vid_dir = '<Path to the ActivityNet raw video directory>'
qvh_vid_dir = '<Path to the QVHighlights raw video directory>'

rextime_data = load_dataset(dataset_path, split="validation")
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

# Input configuration
prompt = f"USER: <video>{input_data['question']} ASSISTANT: "
if input_data['source'] == "qvhighlights_val":
    video_path = os.path.join(qvh_vid_dir, input_data['vid'] + '.mp4')
else:
    video_path = os.path.join(anet_vid_dir, input_data['vid'] + '.mp4')
container = av.open(video_path)

prompts = []
for option in input_data['options']:
    sentence = input_data['answer'].replace("<s0>", str(input_data['span'][0]))
    sentence = sentence.replace("<e0>", str(input_data['span'][1]))
    sentence = sentence.replace("<option>", option)
    prompts.append(prompt + sentence)

# Sample uniformly 8 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

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
