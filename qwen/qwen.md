
## Model Support List

- [x] [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)


### Qwen/Qwen2.5-0.5B-Instruct


1. download model
```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /path/to/Qwen2.5-0.5B-Instruct
```

2. install required packages
```bash
conda create -n llm2onnx python=3.10
pip install requirements.txt
```

3. run script to export onnx for qwen
```bash
python qwen/qwen2onnx.py -m /path/to/model/Qwen2.5-0.5B-Instruct
```



