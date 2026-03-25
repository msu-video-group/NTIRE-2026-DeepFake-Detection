# NTIRE 2026 Robust AI-Generated Image Detection in the Wild

[![Page](https://img.shields.io/badge/Challenge-Page-blue)](https://www.codabench.org/competitions/12761/)

<img width="2637" height="1358" alt="NTIRE_LOGO_WIDE" src="https://i.ibb.co/Y4qsmxHL/header-compr2.png" />

### Challenge overview

Text-to-image (T2I) models have made synthetic images nearly indistinguishable from real photos in many cases, which creates serious challenges for trust, authenticity, forensics, and content safety. At the same time, real-world images are routinely transformed (cropped, resized, compressed, blurred), and detectors must remain reliable under such post-processing and distribution shifts.

In this challenge, we introduce a dataset of real and AI-generated images, with additional “in-the-wild” style transformations, to benchmark detection methods that are accurate, robust, and generalize to unseen generators.

### Evaluation metrics
* Primary metric – Robust ROC AUC
  * Compute a single ROC AUC across all images **after** transformations using label_num (0/1) and the submitted score. This measures the detector’s global discriminative ability regardless of threshold and robustness to postprocessing.
* Secondary metric -- Clean ROC AUC
  * Compute a single ROC AUC across all images **without** transformations using label_num (0/1) and the submitted score. This measures the detector’s global discriminative ability regardless of threshold and **does not** assess robustness to postprocessing.
 
### Data

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

**Test (Private) Transformations:** Color Saturation, Brightness Increase, Lens Blur, JPEG Compression, Impulse Noise, RGB Channel Shift, Random Crop, Random Aspect Crop, Neural Image Compression (JPEG AI), Random Tone Curve, CLAHE, ISO Noise, Perspective Transform, Multiple Compressions (JPEG), Multiple Compressions (JPEG + JPEG AI), Multiple Compressions (JPEG + JPEG 2000), Watermark Attack (Adv. Embedding, CLIP/ResNet), JPEG 2000, Watermark Attack (WMForger), Neural Image Compression (Cheng2020), Shot Noise, Glass Blur, Downscale, Invisible Watermark Insertion (1 of 7 algorithms) |

<sup>†</sup> Proprietary model.
