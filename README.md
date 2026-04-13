# NTIRE 2026 Robust AI-Generated Image Detection in the Wild

[![Page](https://img.shields.io/badge/Challenge-Page-blue)](https://www.codabench.org/competitions/12761/)

<img width="2637" height="1358" alt="NTIRE_LOGO_WIDE" src="https://i.ibb.co/Y4qsmxHL/header-compr2.png" />

## Challenge overview

Text-to-image (T2I) models have made synthetic images nearly indistinguishable from real photos in many cases, which creates serious challenges for trust, authenticity, forensics, and content safety. At the same time, real-world images are routinely transformed (cropped, resized, compressed, blurred), and detectors must remain reliable under such post-processing and distribution shifts.

In this challenge, we introduce a dataset of real and AI-generated images, with additional “in-the-wild” style transformations, to benchmark detection methods that are accurate, robust, and generalize to unseen generators.

### Important dates
* **15 January - 31 January: Intro Phase**. We present the challenge, invite participants, and introduce the subject. 
* **31 January - 10 March: Validation Phase**. We accept submissions for automatic evaluation on validation data. Submissions are limited per day (subject to change). At the start of this phase, the starting kit, training data, and validation data will be released.
* **10 March - 15 March: Testing Phase**. We accept submissions for evaluation on test data. 
* **15 March - 17 March: Reproducibility window**. Teams submit code or executables for verification.
* **19 March: Final results announced**. Final rankings are confirmed after reproducibility checks.
* **22 March: Challenge-paper submission deadline**. Teams submit the papers describing their approaches to NTIRE 2026.
* **24 March: Paper acceptance decision notification**
* **8 April: Camera ready submission**

## Evaluation metrics
* Primary metric – Robust ROC AUC
  * Compute a single ROC AUC across all images **after** transformations using label_num (0/1) and the submitted score. This measures the detector’s global discriminative ability regardless of threshold and robustness to postprocessing.
* Secondary metric -- Clean ROC AUC
  * Compute a single ROC AUC across all images **without** transformations using label_num (0/1) and the submitted score. This measures the detector’s global discriminative ability regardless of threshold and **does not** assess robustness to postprocessing.
 
## Data

| **Split** | **# of Images** | **Real/Fake Ratio** | **Labels Provided** | **Generator Models** | **Transformations** |
|---|---|---|---|---|---|
| **Train** | ~277K | ~1:1.77 | Yes | 20 Models<br> | 12 Transformations |
| **Validation** | 10K | 1:1 | No | 9 Models | 19 Transformations |
| **Validation Hard** | 2.5K | 1:1 | No | 7 Models | 19 Transformations |
| **Test (Public)** | 2.5K | 1:1 | No | 10 Models | 22 Transformations |
| **Test (Private)** | 2.5K | ~1:1 | No | 10 Models | 24 Transformations |

### Models and Transformations

**Train Models:** YOSO PixArt-512, PixArt-α, PixArt-Σ, Kandinsky 2, Kandinsky 3, Kolors, OmniGen, OmniGen 2, Stable Diffusion 1.4, Stable Diffusion 1.5, Stable Diffusion 2.1, Stable Diffusion XL 1.0, SDXL Lightning, SDXL Turbo, Janus Pro 7B, Infinity 2B, Infinity 8B, Ovis Image, DeepFloyd IF, FLUX.1 Kontext Dev 

**Train Transformations:** (Provided as distortion pipeline) Gaussian Blur, Lens Blur, Color Shift, Color Saturation, JPEG Compression, White Noise, Impulse Noise, Brightness Increase, Brightness Decrease, Color Jitter, Color Quantization, Linear Contrast Change

 **Validation Models:** FLUX.1 Kontext Dev, SDXL Turbo, FLUX.1 Dev, Playground v2.5, Lumina Image 2.0, Qwen Image, Stable Diffusion 3 Medium, Ideogram v3 Turbo<sup>†</sup>, ImageGen-4 Fast<sup>†</sup> 
 
 **Validation Transformations:** Gaussian Blur, Lens Blur, Color Shift, Color Saturation, JPEG Compression, White Noise, Impulse Noise, Brightness Increase, Brightness Decrease, Color Jitter, Color Quantization, Linear Contrast Change, Motion Blur, Multiplicative Noise, Pixelation, RGB Channel Shift, Random Crop, Random Aspect Crop, Downscale 
 
**Validation Hard Models:** Playground v2.5, SDXL Turbo, HiDream, FLUX.1 Schnell, Stable Diffusion 3.5 Large Turbo, Nano Banana<sup>†</sup>, Seedream 4<sup>†</sup> 

 **Validation Hard Transformations:**  Gaussian Blur, Lens Blur, JPEG Compression, White Noise, Impulse Noise, Color Quantization, Multiplicative Noise, RGB Channel Shift, Random Crop, Random Aspect Crop, Neural Image Compression (JPEG AI), Random Tone Curve, CLAHE, ISO Noise, Perspective Transform, Multiple Compressions (JPEG), Multiple Compressions (JPEG + JPEG AI), Watermark Attack (Adv. Embedding, CLIP/ResNet), Downscale 
 
**Test (Public) Models:** HiDream, FLUX.1 Schnell, Stable Diffusion 3.5 Large, FLUX Krea, Z-Image Turbo, Nano Banana Pro<sup>†</sup>, FLUX-2 Max<sup>†</sup>, ImageGen-4 Ultra<sup>†</sup>, Seedream 5 Lite<sup>†</sup>, Groq Imagine Image<sup>†</sup> 

**Test (Public) Transformations:** Color Saturation, Brightness Increase, Lens Blur, JPEG Compression, Impulse Noise, RGB Channel Shift, Random Crop, Random Aspect Crop, Neural Image Compression (JPEG AI), Random Tone Curve, CLAHE, ISO Noise, Perspective Transform, Multiple Compressions (JPEG), Multiple Compressions (JPEG + JPEG AI), Watermark Attack (Adv. Embedding, CLIP/ResNet), JPEG 2000, Watermark Attack (WMForger), Neural Image Compression (Cheng2020), Shot Noise, Downscale, Invisible Watermark Insertion (1 of 6 algorithms) 

**Test (Private) Models:** HiDream, Stable Diffusion 3.5 Large Turbo, FLUX.1 Dev SRPO, Z-Image Turbo, Kandinsky 5, Nano Banana 2<sup>†</sup>, GPT Image 1.5<sup>†</sup>, ImageGen-4 Ultra<sup>†</sup>, Seedream 5 Lite<sup>†</sup>, Groq Imagine Image<sup>†</sup> 


### Test Dataset

The Public Test Dataset contains 2,500 images in total: 1,250 clean and 1,250 distorted. No labels are provided, meaning there is no indication of whether an image is Real vs. AI-generated or Clean vs. Distorted.

The Public Test Dataset can be downloaded [here](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-test). If you have problems downloading it please feel free to write a post on forum or send us an email.

For the submission format please refer to *Final test submission* tab.

### Transformations Script
The basic distortion pipeline is now  [available](https://drive.google.com/file/d/1oGr--PUOd11xy0ayYB6p2Mgg67n6eJPc/view?usp=sharing).

### Toy Dataset
[Toy dataset](https://drive.google.com/file/d/1d5m9tBDaiZ6rYuv7ZnR139D4eQPSfwYR/view?usp=drive_link) 

The toy dataset is intended to help participants get familiar with the data structure and the submission format, to verify their pipelines and does not reflect the distribution of the training data.

### Train Dataset
Train dataset consists of ~277k images split into 6 shards of size 50k (except for the last one). Shards can be downloaded from https://calypso.gml-team.ru:5001/sharing/oLxhMpcLY or https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train. All shards have similar data distribution, and can be used separately if you prefer to train/test the model on a smaller set. Shards have following structure:

```
./shard_*i*/
 images/
  *image_name*.jpg
  …
 labels.csv
 ```

image_name is a unique random combination of 20 symbols. labels.csv contains a map between image names and their respective labels (0 corresponds to a real image, and 1 to a generated one).

Following PyTorch Dataset class can be used to work with the train set:

```
import pandas as pd 
import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image 
import torch
class AIGenDetDataset(Dataset):
    def __init__(self, shard_dir, shard_nums=None):
        """ 
        Example dataset class for NTIRE 2026 Robust AI-generated Image Detection In The Wild challenge.

        Arguments:
            - shard_dir (str): base directory where shards are stored 
            - shard_nums (None or list[int]): a list of specific shard numbers (0 to 6) to use. 
                Uses all shards found in shard_dir by default.
        """
        self.shard_root_dir = shard_dir
        if shard_nums is None:
            self.shard_dirs = [os.path.join(shard_dir, f'shard_{i}') for i in range(0,6)]
        else:
            self.shard_dirs = [os.path.join(shard_dir, f'shard_{i}') for i in shard_nums]
        self.shard_dirs = [x for x in self.shard_dirs if os.path.isdir(x)]
        label_dfs = [pd.read_csv(os.path.join(x, 'labels.csv'), index_col=0) for x in self.shard_dirs]
        self.label_df = pd.DataFrame(columns=['image_name', 'label', 'shard_name'])
        for idx, ldf in enumerate(label_dfs):
            ldf['shard_name'] = Path(self.shard_dirs[idx]).name
            self.label_df = pd.concat([self.label_df, ldf], ignore_index=True)
        print(f'Found {len(self.shard_dirs)} shards, {len(self.label_df)} images in total.')

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.shard_root_dir,
                                self.label_df.loc[idx, 'shard_name'], 'images', self.label_df.loc[idx, 'image_name'])
        image = Image.open(img_path)
        label = self.label_df.loc[idx, 'label']
        sample = {'image': image, 'label': label}
        return sample
```

### Validation Dataset

The Validation Dataset contains 10,000 images in total: 5,000 clean and 5,000 distorted. No labels are provided, meaning there is no indication of whether an image is Real vs. AI-generated or Clean vs. Distorted.

The Validation Dataset can be downloaded [here](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-val). If you have problems downloading it please feel free to write a post on forum or send us an email.

For the submission format please refer to *Evaluation* tab.

**Hard part of the Validation Dataset** is available [here](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-val). Download *val_images_hard.zip* to use it. This second part is more similar to the Test Dataset and will help participants to adjust their solutions.
**Test (Private) Transformations:** Color Saturation, Brightness Increase, Lens Blur, JPEG Compression, Impulse Noise, RGB Channel Shift, Random Crop, Random Aspect Crop, Neural Image Compression (JPEG AI), Random Tone Curve, CLAHE, ISO Noise, Perspective Transform, Multiple Compressions (JPEG), Multiple Compressions (JPEG + JPEG AI), Multiple Compressions (JPEG + JPEG 2000), Watermark Attack (Adv. Embedding, CLIP/ResNet), JPEG 2000, Watermark Attack (WMForger), Neural Image Compression (Cheng2020), Shot Noise, Glass Blur, Downscale, Invisible Watermark Insertion (1 of 7 algorithms)

<sup>†</sup> Proprietary model.

## Organizers

The NTIRE challenge on Robust AI‑Generated Image Detection in the Wild is organized jointly with MSU, and the NTIRE 2026 workshop The results of the challenge will be published at NTIRE 2026 workshop and in the CVPR 2026 Workshops proceedings.

Organizers
- Aleksandr Gushchin¹ (alexanterg@gmail.com <-- primary contact)
- Khaled Abud¹
- Georgii Bychkov¹
- Ekaterina Shumitskaya¹
- Anastasia Antsiferova¹
- Dmitriy Vatolin¹
- Radu Timofte² (Radu.Timofte@uni-wuerzburg.de)
- Changsheng Chen³
- Shunquan Tan³

¹ -- MSU

² -- University of Würzburg, Germany

³ -- Shenzhen MSU-BIT University, SMBU

[The Graphics & Media Lab (GML)](https://videoprocessing.ai/) at the MSU is a leading research laboratory in the field of Artificial Intelligence, particularly in the quality assessment, processing, and compression of multimedia data. GML has contributed over a hundred articles to prestigious journals and conferences, including ECCV, NIPS, CVPR, AAAI, ICML.

[Computer Vision Laboratory and University of Würzburg](https://www.informatik.uni-wuerzburg.de/computervision/) in general are an exciting environment for research, for independent thinking. Our team is highly international, with people from about 12 countries, and the members have already won awards at top conferences (ICCV, CVPR, ICRA, NeurIPS, ...), founded successful spinoffs, and/or collaborated with industry.

The AI Security Lab in SMBU has been continuously engaged in multimedia forensics and information security research since 2004, and is among the earliest groups in China to systematically explore this field. In recent years, the team has established a stable research framework focusing on image and video forensics, AI-generated content detection, tampering localization, identity authentication, and privacy protection. This team received the Natural Science Award from the China Society of Image and Graphics (2024) for its research on Multimedia forensics.

## Citation

If you use this code for your research, please cite our paper.

```
@inproceedings{ntire26aigendet, 
title={{    NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild    }}, 
author={    Gushchin, Aleksandr and  Abud, Khaled and  Shumitskaya, Ekaterina and  Filippov, Artem and  Bychkov, Georgii and  Lavrushkin, Sergey and  Erofeev, Mikhail and  Antsiferova, Anastasia and  Chen, Changsheng and  Tan, Shunquan and  Timofte, Radu and  Vatolin, Dmitriy and others    },
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},  
year = {2026} 
}
```
