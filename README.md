# Fork AutoAWQ

<p align="center">
| <a href="https://github.com/casper-hansen/AutoAWQ/issues/32"><b>Roadmap</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/tree/main/examples"><b>Examples</b></a> | <a href="https://github.com/casper-hansen/AutoAWQ/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22"><b>Issues: Help Wanted</b></a> |
</p>

<p align="center" style="margin-bottom: 0px;">
    <a href="https://huggingface.co/models?search=awq">
        <img alt="Huggingface - Models" src="https://img.shields.io/badge/🤗_1000+_models_available-8A2BE2">
    </a>
    <a href="https://github.com/casper-hansen/AutoAWQ/releases">
        <img alt="GitHub - Releases" src="https://img.shields.io/github/release/casper-hansen/AutoAWQ.svg">
    </a>
    <a href="https://pypi.org/project/autoawq/">
        <img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/autoawq/month">
    </a>
</p>

<div align="center" style="color: white;">
    <p>Supported by</p>
    <a href="https://runpod.io/?utm_source=referral&utm_medium=autoAWQ">
    <img src="https://github.com/aadil-runpod/rp-logo/assets/164108768/a8fc546d-cbab-44c4-9a5a-dfb6c400ad24" alt="RunPod Logo" width="100" border="0">  </a>
</div>
Fork AutoAWQ  
  
1. 支持 DeepSeek-R1(fp8) w4a16 量化  
2. 改进 ep 量化为多卡并行，在 8*H800 上，量化从 10h 降低到 2h+ 
3. 改进 gemm , 将低效晦涩的 for 循环改进为高效易懂的 torch.tensor 运算
4. 重构了量化相关代码，并去除了一些 module

## Prerequisites  

pip install transformers==4.49.0

## Install 

python setup.py install --force

## Usage

Under examples, you can find examples of how to quantize, run inference, and benchmark AutoAWQ models.

### INT4 GEMM vs INT4 GEMV vs FP16

There are two versions of AWQ: GEMM and GEMV. Both names relate to how matrix multiplication runs under the hood. We suggest the following:

- GEMV (quantized): 20% faster than GEMM, only batch size 1 (not good for large context).
- GEMM (quantized): Much faster than FP16 at batch sizes below 8 (good with large contexts).
- FP16 (non-quantized): Recommended for highest throughput: [vLLM](https://github.com/vllm-project/vllm).

#### Compute-bound vs Memory-bound

At small batch sizes with small 7B models, we are memory-bound. This means we are bound by the bandwidth our GPU has to push around the weights in memory, and this is essentially what limits how many tokens per second we can generate. Being memory-bound is what makes quantized models faster because your weights are 3x smaller and can therefore be pushed around in memory much faster. This is different from being compute-bound where the main time spent during generation is doing matrix multiplication. 

In the scenario of being compute-bound, which happens at higher batch sizes, you will not gain a speed-up using a W4A16 quantized model because the overhead of dequantization will slow down the overall generation. This happens because AWQ quantized models only store the weights in INT4 but perform FP16 operations during inference, so we are essentially converting INT4 -> FP16 during inference.

### Fused modules

Fused modules are a large part of the speedup you get from AutoAWQ. The idea is to combine multiple layers into a single operation, thus becoming more efficient. Fused modules represent a set of custom modules that work separately from Huggingface models. They are compatible with `model.generate()` and other Huggingface methods, which comes with some inflexibility in how you can use your model if you activate fused modules:

- Fused modules are activated when you use `fuse_layers=True`.
- A custom cache is implemented. It preallocates based on batch size and sequence length.
    - You cannot change the sequence length after you have created your model.
    - Reference: `AutoAWQForCausalLM.from_quantized(max_seq_len=seq_len, batch_size=batch_size)`
- The main accelerator in the fused modules comes from FasterTransformer, which is only compatible with Linux.
- The `past_key_values` from `model.generate()` are only dummy values, so they cannot be used after generation.

### Examples

More examples can be found in the [examples directory](examples).

<details>

<summary>Quantization</summary>

Expect this to take 10-15 minutes on smaller 7B models, and around 1 hour for 70B models.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

</details>

<details>

<summary>Inference</summary>

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
from awq.utils.utils import get_best_device

device = get_best_device()

quant_path = "TheBloke/zephyr-7B-beta-AWQ"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>"""

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.to(device)

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_seq_len=512
)
```

</details>

## Benchmarks

These benchmarks showcase the speed and memory usage of processing context (prefill) and generating tokens (decoding). The results include speed at various batch sizes and different versions of AWQ kernels. We have aimed to test models fairly using the same benchmarking tool that you can use to reproduce the results. Do note that speed may vary not only between GPUs but also between CPUs. What matters most is a GPU with high memory bandwidth and a CPU with high single core clock speed.

- Tested with AutoAWQ version 0.1.6
- GPU: RTX 4090 (AMD Ryzen 9 7950X)
- Command: `python examples/benchmark.py --model_path <hf_model> --batch_size 1`
- 🟢 for GEMV, 🔵 for GEMM, 🔴 for avoid using

| Model Name | Size | Version | Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory (VRAM)     |
| ---------- | ---- | ------- | ---------- | -------------- | ------------- | ---------------- | --------------- | ----------------- |
| Vicuna     | 7B   | 🟢GEMV   | 1          | 64             | 64            | 639.65           | 198.848         | 4.50 GB (19.05%)  |
| Vicuna     | 7B   | 🟢GEMV   | 1          | 2048           | 2048          | 1123.63          | 133.191         | 6.15 GB (26.02%)  |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| Mistral    | 7B   | 🔵GEMM   | 1          | 64             | 64            | 1093.35          | 156.317         | 4.35 GB (18.41%)  |
| Mistral    | 7B   | 🔵GEMM   | 1          | 2048           | 2048          | 3897.02          | 114.355         | 5.55 GB (23.48%)  |
| Mistral    | 7B   | 🔵GEMM   | 8          | 64             | 64            | 4199.18          | 1185.25         | 4.35 GB (18.41%)  |
| Mistral    | 7B   | 🔵GEMM   | 8          | 2048           | 2048          | 3661.46          | 829.754         | 16.82 GB (71.12%) |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| Mistral    | 7B   | 🟢GEMV   | 1          | 64             | 64            | 531.99           | 188.29          | 4.28 GB (18.08%)  |
| Mistral    | 7B   | 🟢GEMV   | 1          | 2048           | 2048          | 903.83           | 130.66          | 5.55 GB (23.48%)  |
| Mistral    | 7B   | 🔴GEMV   | 8          | 64             | 64            | 897.87           | 486.46          | 4.33 GB (18.31%)  |
| Mistral    | 7B   | 🔴GEMV   | 8          | 2048           | 2048          | 884.22           | 411.893         | 16.82 GB (71.12%) |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| TinyLlama  | 1B   | 🟢GEMV   | 1          | 64             | 64            | 1088.63          | 548.993         | 0.86 GB (3.62%)   |
| TinyLlama  | 1B   | 🟢GEMV   | 1          | 2048           | 2048          | 5178.98          | 431.468         | 2.10 GB (8.89%)   |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| Llama 2    | 13B  | 🔵GEMM   | 1          | 64             | 64            | 820.34           | 96.74           | 8.47 GB (35.83%)  |
| Llama 2    | 13B  | 🔵GEMM   | 1          | 2048           | 2048          | 2279.41          | 73.8213         | 10.28 GB (43.46%) |
| Llama 2    | 13B  | 🔵GEMM   | 3          | 64             | 64            | 1593.88          | 286.249         | 8.57 GB (36.24%)  |
| Llama 2    | 13B  | 🔵GEMM   | 3          | 2048           | 2048          | 2226.7           | 189.573         | 16.90 GB (71.47%) |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| MPT        | 7B   | 🔵GEMM   | 1          | 64             | 64            | 1079.06          | 161.344         | 3.67 GB (15.51%)  |
| MPT        | 7B   | 🔵GEMM   | 1          | 2048           | 2048          | 4069.78          | 114.982         | 5.87 GB (24.82%)  |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| Falcon     | 7B   | 🔵GEMM   | 1          | 64             | 64            | 1139.93          | 133.585         | 4.47 GB (18.92%)  |
| Falcon     | 7B   | 🔵GEMM   | 1          | 2048           | 2048          | 2850.97          | 115.73          | 6.83 GB (28.88%)  |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| CodeLlama  | 34B  | 🔵GEMM   | 1          | 64             | 64            | 681.74           | 41.01           | 19.05 GB (80.57%) |
| CodeLlama  | 34B  | 🔵GEMM   | 1          | 2048           | 2048          | 1072.36          | 35.8316         | 20.26 GB (85.68%) |
| ...        | ...  | ...     | ...        | ...            | ...           | ...              | ...             | ...               |
| DeepSeek   | 33B  | 🔵GEMM   | 1          | 64             | 64            | 1160.18          | 40.29           | 18.92 GB (80.00%) |
| DeepSeek   | 33B  | 🔵GEMM   | 1          | 2048           | 2048          | 1012.1           | 34.0093         | 19.87 GB (84.02%) |

### Multi-GPU

GPU: 2x NVIDIA GeForce RTX 4090

| Model | Size    | Version       |   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)     |
|--------:|------:|--------------:|-------------:|-----------------:|----------------:|-------------------:|------------------:|:------------------|
| Mixtral | 46.7B | 🔵GEMM        |            1 |               32 |              32 |            149.742 |           93.406  | 25.28 GB (53.44%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |               64 |              64 |           1489.64  |           93.184  | 25.32 GB (53.53%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |              128 |             128 |           2082.95  |           92.9444 | 25.33 GB (53.55%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |              256 |             256 |           2428.59  |           91.5187 | 25.35 GB (53.59%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |              512 |             512 |           2633.11  |           89.1457 | 25.39 GB (53.67%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |             1024 |            1024 |           2598.95  |           84.6753 | 25.75 GB (54.44%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |             2048 |            2048 |           2446.15  |           77.0516 | 27.98 GB (59.15%) |
| Mixtral | 46.7B | 🔵GEMM        |            1 |             4096 |            4096 |           1985.78  |           77.5689 | 34.65 GB (73.26%) |

### CPU

- CPU: 48 cores SPR (Intel 4th Gen Xeon CPU)
- Command: `python examples/benchmark.py --model_path <hf_model> --batch_size 1 --generator hf`

| Model | Version | Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory |
|-------|---------|------------|----------------|---------------|-------------------|------------------|---------------|
| TinyLlama 1B | gemm | 1 | 32 | 32 | 817.86 | 70.93 | 1.94 GB (0.00%) |
| TinyLlama 1B | gemm | 1 | 2048 | 2048 | 5279.15 | 36.83 | 2.31 GB (0.00%) |
| Falcon 7B | gemm | 1 | 32 | 32 | 337.51 | 26.41 | 9.57 GB (0.01%) |
| Falcon 7B | gemm | 1 | 2048 | 2048 | 546.71 | 18.8 | 13.46 GB (0.01%) |
| Mistral 7B | gemm | 1 | 32 | 32 | 343.08 | 28.46 | 9.74 GB (0.01%) |
| Mistral 7B | gemm | 1 | 2048 | 2048 | 1135.23 | 13.23 | 10.35 GB (0.01%) |
| Vicuna 7B | gemm | 1 | 32 | 32 | 340.73 | 28.86 | 9.59 GB (0.01%) |
| Vicuna 7B | gemm | 1 | 2048 | 2048 | 1143.19 | 11.14 | 10.98 GB (0.01%) |
| Llama 2 13B | gemm | 1 | 32 | 32 | 220.79 | 18.14 | 17.46 GB (0.02%) |
| Llama 2 13B | gemm | 1 | 2048 | 2048 | 650.94 | 6.54 | 19.84 GB (0.02%) |
| DeepSeek Coder 33B | gemm | 1 | 32 | 32 | 101.61 | 8.58 | 40.80 GB (0.04%) |
| DeepSeek Coder 33B | gemm | 1 | 2048 | 2048 | 245.02 | 3.48 | 41.72 GB (0.04%) |
| Phind CodeLlama 34B | gemm | 1 | 32 | 32 | 102.47 | 9.04 | 41.70 GB (0.04%) |
| Phind CodeLlama 34B | gemm | 1 | 2048 | 2048 | 237.57 | 3.48 | 42.47 GB (0.04%) |

## Reference

If you find AWQ useful or relevant to your research, you can cite their [paper](https://arxiv.org/abs/2306.00978):

```
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
```
