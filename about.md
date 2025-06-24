---
layout: article
titles:
  # @start locale config
  en      : &EN       About LightLLM
  en-GB   : *EN
  en-US   : *EN
  en-CA   : *EN
  en-AU   : *EN
  zh-Hans : &ZH_HANS  关于LightLLM
  zh      : *ZH_HANS
  zh-CN   : *ZH_HANS
  zh-SG   : *ZH_HANS
  zh-Hant : &ZH_HANT  關於LightLLM
  zh-TW   : *ZH_HANT
  zh-HK   : *ZH_HANT
  # @end locale config
key: page-about
---

<p style="text-align:center">
  <img src="./assets/images/logo/lightllm-logo.webp" alt="Lightllm" width="100%">
</p>

<p style="text-align:center">
  <strong>A lightweight, high-performance large language model serving framework</strong>
</p>

<p style="text-align:center">
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <a class="github-button" href="https://github.com/ModelTC/lightllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
  <a class="github-button" href="https://github.com/ModelTC/lightllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
  <a class="github-button" href="https://github.com/ModelTC/lightllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

**Welcome to LightLLM!** LightLLM is a large language model inference and serving framework developed purely in Python, featuring a lightweight design, easy extensibility, and high performance. LightLLM integrates the advantages of many open-source solutions, including but not limited to FasterTransformer, TGI, vLLM, and FlashAttention.

## Key Features

* **Multi-process Collaboration**: Tokenization, language model inference, vision model inference, and other tasks are performed asynchronously, significantly improving GPU utilization.
* **Zero-padding**: Provides support for nopad-Attention computation across multiple models to efficiently handle requests with large length differences.
* **Dynamic Batching**: Capable of dynamic batch scheduling for requests.
* **FlashAttention**: Integrates FlashAttention to improve speed and reduce GPU memory usage during inference.
* **Tensor Parallelism**: Utilizes multiple GPUs for tensor parallelism to accelerate inference speed.
* **Token Attention**: Implements a token-level KV cache memory management mechanism, achieving zero memory waste during inference.
* **High-performance Routing**: Combined with Token Attention, it precisely manages GPU memory at the token level, optimizing system throughput.
* **INT8 KV Cache**: This feature can double the maximum number of tokens. Currently, it only supports Llama-architecture models.

## Supported Models

* [BLOOM](https://huggingface.co/bigscience/bloom)
* [LLaMA](https://github.com/facebookresearch/llama)
* [LLaMA V2](https://huggingface.co/meta-llama)
* [StarCoder](https://github.com/bigcode-project/starcoder)
* [Qwen-7b](https://github.com/QwenLM/Qwen-7B)
* [ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B)
* [Baichuan-7b](https://github.com/baichuan-inc/Baichuan-7B)
* [Baichuan2-7b](https://github.com/baichuan-inc/Baichuan2)
* [Baichuan2-13b](https://github.com/baichuan-inc/Baichuan2)
* [Baichuan-13b](https://github.com/baichuan-inc/Baichuan-13B)
* [InternLM-7b](https://github.com/InternLM/InternLM)
* [Yi-34b](https://huggingface.co/01-ai/Yi-34B)
* [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL)
* [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
* [Llava-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
* [Llava-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)
* [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
* [Stablelm](https://huggingface.co/stabilityai/stablelm-2-1_6b)
* [MiniCPM](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
* [Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
* [CohereForAI](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
* [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
* [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)
