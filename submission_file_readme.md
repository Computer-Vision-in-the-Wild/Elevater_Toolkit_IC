
## Submssion File Format for Zero-shot

#### Final submission file
The final submission file is a ``json`` file (or its zip format), which is the merged result from the prediction files from all 20 datasets (or 21 datasets if ImageNet is considered in submission). The submission json file is contains a dictionary. You can check out [``all_predictions.zip``](https://cvinthewildeus.blob.core.windows.net/datasets/submission_files/all_predictions_klite_gpt3.zip?sp=r&st=2023-08-28T01:41:20Z&se=3023-08-28T09:41:20Z&sv=2022-11-02&sr=c&sig=Msoq5dIl%2Fve6F01edGr8jgcZUt7rtsuJ896xvstSNfM%3D) as one submission example that successfully passes the evaluation pipeline on the server. We illustrate the dictionary format using the pseudo example below.
It consists of prediction results from 21 datasets (including ImageNet-1K).
```json
{"data": 
  [
    {"model_name": "CLIP-clip_swin", "dataset_name": "caltech-101", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 6084, 102)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "cifar-10", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 10000, 10)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "cifar-100", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 10000, 100)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "country211", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 21100, 211)]"},
    {"model_name": "CLIP-clip_swin", "dataset_name": "dtd", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 1880, 47)]"},
    {"model_name": "CLIP-clip_swin", "dataset_name": "eurosat_clip", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 5000, 10)]"},
    {"model_name": "CLIP-clip_swin", "dataset_name": "fer-2013", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 3589, 7)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "fgvc-aircraft-2013b-variants102", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 3333, 100)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "food-101", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 25250, 101)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "gtsrb", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 12630, 43)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "hateful-memes", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 500, 2)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "imagenet-1k", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 50000, 1000)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "kitti-distance", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 711, 4)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "mnist", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 10000, 10)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "oxford-flower-102", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 6149, 102)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "oxford-iiit-pets", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 3669, 37)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "patch-camelyon", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 32768, 2)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "rendered-sst2", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 1821, 2)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "resisc45_clip", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 25200, 45)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "stanford-cars", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 8041, 196)]"}, 
    {"model_name": "CLIP-clip_swin", "dataset_name": "voc-2007-classification", "num_trainable_params": 0.0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 4952, 20)]"}
  ]
}

```

#### Per-dataset predition file

The above ``all_predictions.zip`` is merged from 21 per-dataset predition results, using [``prepare_submit.py``](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC/blob/main/vision_benchmark/commands/prepare_submit.py). 
For each dataset, we first produce a prediction json file, by activating ``--save-predictions`` in [``zeroshot.py``](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC/blob/main/vision_benchmark/commands/zeroshot.py). 
If we use CIFAR-10 dataset for example, the generated prediction file is [``cifar-10.json``](https://cvinthewildeus.blob.core.windows.net/datasets/submission_files/cifar-10.json?sp=r&st=2023-08-28T01:41:20Z&se=3023-08-28T09:41:20Z&sv=2022-11-02&sr=c&sig=Msoq5dIl%2Fve6F01edGr8jgcZUt7rtsuJ896xvstSNfM%3D). We illustrate the dictionary format using the pseudo example below.

```json
{"model_name": "CLIP-clip_swin", "dataset_name": "cifar-10", "num_trainable_params": 0, "num_params": 150695609, "num_visual_params": 86743224, "num_backbone_params": 150695609, "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 10000, 10)]"}
```

where the key-value pairs in the prediction dictionary are :

- `model_name` specify the model name. It will not appear on the leaderboard, the user can customize the model name when filling the survey in submission.
- `dataset_name`: the current dataset for evaluation.
- `num_trainable_params`: the number of trainable parameters in model adaptation stage (type=int); For zero-shot learning, the model is frozen, and thus num_trainable_params = 0.
- `num_params`: `num_backbone_params` + `num_average_head_params` (type=int), where num_average_head_params is the number of parameters in prediction head.  Note that `num_average_head_params = head_dim * average(#cls_ds1, #cls_ds2, ..., #cls_ds20)`.
- `num_visual_params`: the number of parameters in the image encoder of the pre-trained model (type=int).
- `num_backbone_params`: the number of all parameters in the pre-trained model (type=int).
- `n_shot`: the number of training images per class in model adaptation stage (type=int); For zero-shot learning, no training images are used, and thus n_shot = 0.
- `rnd_seeds`: list of random seeds to determine the subset in few-shot evaluation (type=list of int); For zero-shot learning, rnd_seeds is not really, the default [0] is used.
- `predictions`: the prediction probability tensor [size: (1, #images in test set, #classes)] (type=float tensor); For CIFAR-10, #images in test set=10000 and #classes=10.

## Submssion File Format for Few-shot

The basic format is similar to the zero-shot setting. One should pay attention to the following key-value pairs in generating the submission file.

- `num_trainable_params`: the number of trainable parameters in model adaptation stage (type=int); For few-shot learning, a different number of the model weights can be updated, depending on the adaptation method, eg linear probing, prompting, fine-tuing.
- `n_shot`: Only 5-shot learning setting is consdiered to represent few-shot learning, n_shot = 5.
- `rnd_seeds`:  3 subsets of training images are generated via `rnd_seeds=[0,1,2]`, respectively.
- `predictions`: the prediction probability tensor [size: (3, #images in test set, #classes)] (type=float tensor); Note 3 prediction results are considered per datasets, each of which is for a random seed.

## Submssion File Format for Full-shot

The basic format is similar to the zero-shot setting. One should pay attention to the following key-value pairs in generating the submission file.

- `num_trainable_params`: the number of trainable parameters in model adaptation stage (type=int); For few-shot learning, a different number of the model weights can be updated, depending on the adaptation method, eg linear probing, prompting, fine-tuing.
- `n_shot`: All training images are used, n_shot = -1.
- `rnd_seeds`:  No need to generate subsets of training images, the default [0] is used.
- `predictions`: the prediction probability tensor [size: (1, #images in test set, #classes)] (type=float tensor).
