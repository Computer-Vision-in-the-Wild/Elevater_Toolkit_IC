# ELEVATER Image Classification Toolkit

## Introduction

The toolkit for image classification in the benchmark: Evaluation of Language-augmented Visual Task-level Transfer [[ELEVATER]](https://computer-vision-in-the-wild.github.io/ELEVATER/).


### Contents
Please follow the steps below to use this codebase to reproduce the results in the paper, and onboard your own checkpoints & methods.

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Getting Started](#getting-started)
4. [Evaluation](#evaluation)
    1. [Zero-shot](#zero-shot-evaluation)
    2. [Linear probe / Fine-tuning (Few-shot & Full-shot)](#linear-probe-and-fine-tuning)
5. [Submit your results to vision leaderboard](#submit-your-results-to-vision-leaderboard)


## Installation

Our code base is developed and tested with PyTorch 1.7.0, TorchVision 0.8.0, CUDA 11.0, and Python 3.7.

```Shell
conda create -n elevater python=3.7 -y
conda activate elevater

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
pip install -e .
```

## Datasets

We support the downstream evaluation of image classification on 20 datasets: `Caltech101`, `CIFAR10`, `CIFAR100`, `Country211`, `DTD`, `EuroSat`, `FER2013`, `FGVCAircraft`, `Food101`, `GTSRB`, `HatefulMemes`, `KittiDistance`, `MNIST`, `Flowers102`, `OxfordPets`, `PatchCamelyon`, `SST2`, `RESISC45`, `StanfordCars`, `VOC2007`.

To evaluate on these datasets, our toolkit *automatically* downloads these datasets once with [`vision-datasets`](https://github.com/microsoft/vision-datasets) and store them locally for the future usage.  You do **NOT** need to explicitly download any datasets. However, if you are interested in downloading all data before running experiments, please refer to [[Data Download]](https://github.com/Computer-Vision-in-the-Wild/DataDownload).


## Getting Started

ELEVATER benchmark supports three types of the evaluation: zeroshot, linear probe, and finetuning.  We have embodied all three types of the evaluation into a unified launch script: [`run.sh`](run.sh). By specifying different arguments, you may enable different settings, including: 

#### Few-shot
- `num_shots=5`: the number of images in few-shot learning;  default=5. {5, 20, 50} for few shot, and -1 for full-shot
- `random_seed=0`: it specifies the subset of dataset samples used in few-shot; default=0. We conisder [0,1,2] in our benchmark.

#### Language-augmented model adaptation method 
- `init_head_with_text_encoder=True`: whether or not to init the linear head with the proposed language-augmented method, eg, text encoder output
- `merge_encoder_and_proj=False` whether or not to merge the encoder projection and the linear head

#### Unitilization of external knowledge sources

- `use_wordnet_hierachy=False`: WordNet hierachy knowledge is used or not.
- `use_wordnet_definition=False`: WordNet definition knowledge is used or not.
- `use_wiktionary_definition=False`: Wiktionary definition knowledge is used or not.
- `use_gpt3=False`: GPT3 knowledge is used or not.
- `use_gpt3_count=0`: the number of GPT3 knowledge items used: [1,2,3,4,5]

To run the benchmark toolkit, please refer to the instructions in `run.sh` and modify accordingly.  By default, `./run.sh` will run the zeroshot evaluation of the CLIP ViT/B-32 checkpoint on Caltech-101 dataset.

### Launch Multiple Experiments

You may need to launch multiple experiments in batch as ELEVATER benchmark contains 20 datasets.  We provide an example script [`run_multi.sh`](run_multi.sh) where you can specify different configurations from command line directly without modifying the shell script.

```Shell
DATASET=caltech101 \
OUTPUT_DIR=./output/experiment \
bash run_multi.sh
```

You can refer to [`run_multi.sh`](run_multi.sh) to add other customizable configurations.  Examples are `dataset` and `output_dir`.

## Evaluation

### Zero-shot Evaluation

Our implementation and prompts are from OpenAI repo: [[Notebook]](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb) [[Prompt]](https://github.com/openai/CLIP/blob/main/data/prompts.md).


For zero-shot evaluation, we support both the model from the [CLIP](https://github.com/openai/CLIP) repo and customized models.

- [CLIP](https://github.com/openai/CLIP) model, example configuration file can be found here: resources/model/vitb32_CLIP.yaml
- Customized models

To evaluate *customized model* for __zeroshot__ evaluation, you need to:

- Put your model class in folder `vision_benchmark/models`, and register it in [`vision_benchmark/models/__init__.py`](vision_benchmark/models/__init__.py).
- Prefix the file of model class definition with `clip_`, see the example [`vision_benchmark/models/clip_example.py`](vision_benchmark/models/clip_example.py).
- Define method `encode_image()`, which will be used to extract image features.
- Define method `encode_text()`, which will be used to extract text features.
- Define static method `get_zeroshot_model(config)`, which is used to create the model.
- Configure model hyperparameters and specify model parameter file in configuration file. See an example here: [`resources/model/clip_example.yaml`](resources/model/clip_example.yaml)
- Re-run the installation command as mentioned in the beginning.

### Linear Probe and Fine-tuning

We use automatic hyperparameter tuning for linear probe and finetuning evaluation.  For details, please refer to Appendix Sec. D of [our paper](https://arxiv.org/abs/2204.08790).

Models evaluated here can be models from:

- [Pytorch pre-trained model](https://pytorch.org/vision/stable/models.html): ResNet50, ResNet101, etc
- [Timm](https://github.com/rwightman/pytorch-image-models#models): efficientnet_b0, vit_base_patch16_224 (correspond to Vit-B/16 in CLIP), etc
- [CLIP](https://github.com/openai/CLIP): ViT-B/32 (correspond to CLIP-Vit-B/16 in CLIP), etc
- Customized models

To evaluate *customized model*, you need to:

- Put your model class in folder `vision_benchmark/models`, and register it in `vision_benchmark/models/__init__.py`.
- Prefix the file of model class definition with `cls_`, see the example `vision_benchmark/models/cls_example.py`.
- Define method `forward_features()`, which will be used to extract features.
- Define static method `get_cls_model(config)`, which is used to create the model.
- Configure model hyperparameters and specify model parameter file in configuration file. See an example here: `resources/model/example.yaml`
- Re-run the installation command as mentioned in the beginning.

## Submit to Leaderboard

Leaderboard submission are supported via [EvalAI](https://eval.ai/web/challenges/challenge-page/1832/overview).  Please first generate the prediction files locally, and then submit the results to Eval AI.  Details are documented as below.

### Generate Prediction Files

**You need to evaluate and generate prediction files for all 20 datasets before submitting to the leaderboard**. However, to test that the pipeline is working correctly, you *can* submit partial evaluation results, but it will **NOT** appear on the leaderboard and display a **FAIL** status.  The partially evaluated results can be found from the link under "Stderr file" column.

To generate the prediction files, follow the steps below:

1. Verify that prediction file submission is supported.  Prediction file generation is only supported after commit `2c7a53c3`.  Please make sure that your local copy of our code base is up-to-date.

2. Generate prediction files for all datasets separately.  Please make sure to modify output folder accordingly so that 20 prediction files for the same configuration will appear within the same folder.

```
# Modify these two lines accordingly in run.sh

DATASET=caltech101 \
OUTPUT_DIR=./output/exp_1_submit \
  bash run_multi.sh
```

3. Combine all prediction files to a single zip file. Assume `/path_to_predictions` contains all 20 JSON prediction files.  The combined prediction file will be located at `/path_to_predictions/all_predictions.zip`

```Shell
python commands/prepare_submit.py \
  --combine_path /path_to_predictions
```

### Submit to EvalAI

- Go to our [challenge page](https://eval.ai/web/challenges/challenge-page/1832/overview) on EvalAI, and register a team
- Navigate to ["Submit"](https://eval.ai/web/challenges/challenge-page/1832/submission) tab, and select the corresponding phase (track) for your submission.  Explanation and examples of different phases can be found [here](https://eval.ai/web/challenges/challenge-page/1832/evaluation), and the limitations of the pretrained data for different phases can be found [here](https://eval.ai/web/challenges/challenge-page/1832/phases).
- Use "Upload file" for "Select submission type" and choose the **combined** prediction file (a **zip** file).
- Name your submission following the format in the corresponding leaderboard, finish the required Q&A, and click submit.
- After the submission is uploaded, navigate to [My Submissions](https://eval.ai/web/challenges/challenge-page/1832/my-submission), and choose the phase that you just submitted to.  Your submissions will be evaluated by our remote worker.  Typically the evaluation time for IC will be less than 3 minutes.  If it shows an error, click link under "Stderr file" column to see error messages.  If the evaluation does not start or does not finish 10 minutes after your submission, please contact us.
- If everything looks correct, click "Show on leaderboard" to make your results appear on the leaderboard and contribute to the community.  Before this, your results will **NOT** be publicly visible.

### View Leaderboard

Navigate to [Leaderboard](https://eval.ai/web/challenges/challenge-page/1832/leaderboard) tab to view all baseline results and results from the community.

## Citation

Please cite our paper as below if you use the ELEVATER benchmark or our toolkit.

```bibtex
@article{li2022elevater,
    title={ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models},
    author={Li, Chunyuan and Liu, Haotian and Li, Liunian Harold and Zhang, Pengchuan and Aneja, Jyoti and Yang, Jianwei and Jin, Ping and Lee, Yong Jae and Hu, Houdong and Liu, Zicheng and Gao, Jianfeng},
    journal={arXiv preprint arXiv:2204.08790},
    year={2022}
}
```
