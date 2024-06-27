# ReXTime: A Benchmark Suite for Reasoning-Across-Time in Videos

ReXTime is designed to test AI models' temporal reasoning within video events, focusing on understanding cause-and-effect across different video segments, with 921 validation samples and 2,143 test samples.

|[**Project Page**](https://rextime.github.io/) | [**Github**](https://github.com/ReXTime/ReXTime) | [**🏆Leaderboard**](https://eval.ai/web/challenges/challenge-page/2326/overview) | [**📖Paper**]() |

![Teaser](./images/teaser_v5.png)

## Table of Contents

* [Getting Started](#getting-started)
    * [Clone this repo](#clone-this-repo)
    * [Clone dataset from Huggingface](#clone-dataset-from-huggingface)
    * [Source video downloading](#source-video-downloading)
    * [Directory structure](#directory-structure)
    * [Install dependencies](#install-dependencies)
* [Inference Demo](#inference-demo)
* [Evaluation](#evaluation)
* [Acknowledgement](#acknowledgement)
* [LICENSE](#license)

## Getting Started 

### Clone this repo

```
git clone https://github.com/ReXTime/ReXTime.git
cd ReXTime
```

### Clone dataset from Huggingface
```
git clone https://huggingface.co/datasets/ReXTime/ReXTime
```

### Source video downloading

1. ActivityNet

Download the raw video data from the [Download page](http://activity-net.org/download.html) at ActivityNet official website. You need to fill in their request form to have a 7-day-access to download the videos from the drive folders. You can find the [form](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) here.

2. QVHighlights

Download raw video data from the [link]((https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz)) provided by [Moment-DETR](https://github.com/jayleicn/moment_detr). Extract the file.
```
tar -xvzf qvhilights_videos.tar.gz
```

### Directory structure

```
.
├── videos/                                     # Path to the QVHighlights raw videos, can be anywhere.
│   ├── 9c_w8HU3hqc_210.0_360.0.mp4             # Video 1
│   └── efCSWDWjm6g_360.0_510.0.mp4             # Video 2
├── Anet_videos_15fps_short256/                 # Path to the ActivityNet raw videos, can be anywhere.
│   ├── v_5R3h6lxne90.mp4                       # Video 1
│   └── v_aQ-F9wr0HQ4.mp4                       # Video 2
├── ReXTime/                                    # Code repo
│   ├── ReXTime/                                # Huggingface dataset repo
│   ├── evaluation/                             # Evaluation code
│   ├── demo/                                   # Inference demo script
│   └── requirements.txt                        # Packages for environment
...
```

### Install dependencies

```
conda create --name=rextime python=3.10 -y
conda activate rextime
pip install -r requirements.txt
```

## Inference Demo
Here we provide open source model evaluation demo and proprietary models evaluation demo. You need to modify the path to the dataset repo and paths to the directory of two source raw videos in the following scripts. For proprietary models evaluation, you need to fill in your API key.

Open source MLLM demo:
```
python ./demo/inference.py \
    --dataset_path ./ReXTime \
    --anet_vid_dir <Path to the AcrivityNet video directory> \
    --qvh_vid_dir <Paht to the QVHighlights video directory>
```

Proprietary MLLM demo:
```
OPENAI_API_KEY="sk-***********************************" python ./demo/request.py \
    --dataset_path ./ReXTime \
    --anet_vid_dir <Path to the AcrivityNet video directory> \
    --qvh_vid_dir <Paht to the QVHighlights video directory>
```


## Evaluation

This is an example of output/submission file in .jsonl format. For the assessment of moment grounding, you only need to provide "qid" and "pred_relevant_windows". For the assessment of multi-choice VQA, you only need to provide "qid" and "ans". For the assessment of grounding VQA, you need to provide "qid" "pred_relevant_windows" and "ans" in your submission file. For grounding VQA evaluation, the predicted answer should be conditioned on the predicted time span.
```
{"qid": "anet_val384", "pred_relevant_windows": [[0.0, 15.8304]], "ans": "A"}
{"qid": "qvh_val114", "pred_relevant_windows": [[0.0, 25.50]], "ans": "A"}
...
```

Modify the file paths in the following and run:

```
python ./evaluation/rextime_eval.py \
    --submission_path ${submission_path} \
    --gt_path ${gt_path} \
    --save_path ${save_path}
```

Here we only provide the ground truth file of validation set in 'data/rextime_val.jsonl'. To access on the test set, please submit the predicted file to [ReXTime Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/19544?secret_key=4ac51fd2-1349-45c3-900e-9144217413a1).

## Acknowledgement
* The evaluation code is build from [Moment-detr](https://github.com/jayleicn/moment_detr). 

## License
The annotation files are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, see [./data/LICENSE](data/LICENSE). All the code are under [MIT](https://opensource.org/licenses/MIT) license, see [LICENSE](./LICENSE).

<!--
**GTR-Benchmark/GTR-Benchmark** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
