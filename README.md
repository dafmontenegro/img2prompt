# img2prompt

**img2prompt** is a local Python tool that transforms images into detailed text descriptions (prompts) using the [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) image captioning model. Its goal is to evaluate the impact of different decoding configurations to generate more diverse, detailed, and context-rich prompts—especially useful for downstream applications like image-to-image generation or semantic analysis.

## Theoretical Framework
This project explores **image captioning** via **Vision-Language Transformers**, specifically the BLIP (Bootstrapped Language Image Pretraining) architecture. BLIP employs a multimodal transformer model that has been pre-trained for tasks like image-text matching and conditional generation. We test caption generation across varying decoding strategies, evaluating:

* **Minimum sequence length**
* **Beam search with penalties for repetition**
* **Score and token analysis per inference step**

This allows us to quantify **caption diversity**, **semantic variance**, and **token confidence evolution** across different configurations.

## Tools Used

* Python 3
* [Transformers (Hugging Face)](https://huggingface.co/docs/transformers)
* PyTorch with MPS backend (for macOS acceleration)
* Pillow (PIL)
* Standard libraries: `json`, `platform`, `os`, `time`

## Development

### 1. Model and Image Setup

* Loads the `Salesforce/blip-image-captioning-large` model and processor.
* Uses `torch.float16` on macOS/Apple Sillicon GPUs via MPS when available.
* Reads and processes a local image (`test.jpg`).

### 2. Baseline Inference

* Performs a first pass with default settings to generate a **base caption** for later comparison.

### 3. Grid Configuration and Experiments

Three decoding configurations are tested:

```python
[
  {"min_length": 10},
  {"min_length": 30},
  {"min_length": 50}
]
```

Each configuration applies:

* `beam search` with 7 beams
* `length_penalty = 1.2`
* `repetition_penalty = 1.2`
* `no_repeat_ngram_size = 3`

### 4. Output Analysis

For each configuration, the system records:

* **Final caption**
* **Generation score**
* **Difference vs base caption** (in word set)
* **Token-level scores per step** (top 5 candidates per position)
* **Full JSON reports saved in `./resultados/`**

Example output structure:

```json
{
  "caption_base": "...",
  "results": [
    {
      "caption": "...",
      "score": 12.33,
      "delta_vs_base": 9,
      "tokens": [...],
      "inference_time": 3.5
    }
  ],
  "step_scores": [
    {
      "step": 1,
      "top_tokens": [["a", 2.31], ["the", 2.21], ...]
    }
  ]
}
```

## Conclusions

* BLIP generates high-quality captions out-of-the-box, but customizing decoding strategies can unlock more nuanced or artistic outputs.
* This tool is suitable for developers aiming to **extract detailed prompts** from images to use in generative art or caption-based indexing.
* Local execution ensures full control over inference, performance profiling, and offline experimentation.
