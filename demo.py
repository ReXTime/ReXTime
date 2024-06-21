import os, json
from PIL import Image
import requests
import numpy as np
import av
import torch
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from transformers import BitsAndBytesConfig

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

data_file = '/home/shinji106/ntu/PrivateEval/rextime/ReXTime/data/rextime_val.json'
anet_vid_dir = '/home/shinji106/Data/ActivityNet/Anet_videos_15fps_short256'
qvh_vid_dir = '/data/ntu/videos'

# TODO: Replace with Huggingface dataset
with open(data_file, 'r') as f:
    data = json.load(f)
input_data = data[0]
# dataset = load_dataset("ReXTime", split="test")
# breakpoint()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  # Choose quantization type, nf4 or fp4
    bnb_4bit_use_double_quant=True,  # Optional: improves performance but requires more memory
    bnb_4bit_compute_dtype=torch.float16  # Set compute dtype to float16 for faster computation
)

# Load model and processor
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Input configuration
# prompt = "USER: <video>Why is this video funny? ASSISTANT:"
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
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

# inputs = processor(text=prompt, videos=clip, return_tensors="pt", padding=True).to(device)
clips = [clip for _ in range(len(prompts))]
inputs = processor(text=prompts, videos=clips, return_tensors="pt", padding=True).to(device)

# Inference to get the logits
output = model(**inputs)

# Alternatively, you can also feed the labels to the model to compute the loss as scores.
preds = []
for opt in range(len(prompts)):
    # breakpoint()
    # 6 is the length of input_ids which is ahead of the image tokens, here we only consider the scores after the image tokens.
    sentence_length = inputs['input_ids'][opt].shape[0] - 6
    scores = tuple([i.unsqueeze(0) for i in output['logits'][opt]][-sentence_length-1:-1])
    output_scores = model.compute_transition_scores(inputs['input_ids'][opt].unsqueeze(0)[:, 6:], scores, normalize_logits=True)
    score = sum(output_scores[0].cpu().tolist()) / len(output_scores[0])
    preds.append(score)

# generate_ids = model.generate(**inputs, max_length=80)
# output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Output the result
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
pred_ans  = mapping[preds.index(max(preds))]
print(pred_ans == input_data['ans'])
