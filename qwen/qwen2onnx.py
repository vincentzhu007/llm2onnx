import os
import logging
import argparse
import numpy as np
import torch
import transformers
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen2Onnx:
    def __init__(self):
        pass

    def load(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.eval()
        logging.info(f'model loaded: {model_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info(f'tokenizer loaded: {model_path}')

    def run(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        self.input_ids = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **self.input_ids,
            max_new_tokens=256
            )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def export(self, model_path):
        self.onnx_path = model_path
        torch.onnx.export(
            self.model,
            (torch.zeros(1, 256, dtype=torch.long),),
            model_path,
            input_names=['input_ids'],
            output_names=['logits']
        )

    def run_onnx(self, onnx_path):
        self.ort_session = ort.InferenceSession(onnx_path)
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name

        input_data = np.zeros((1, 256), dtype=np.int64)
        input_data.fill(self.tokenizer.eos_token_id)

        real_input = self.input_ids['input_ids'].cpu().numpy()
        input_data[0, :real_input.shape[1]] = real_input[0]

        ort_inputs = {input_name: input_data}
        ort_outputs = self.ort_session.run([output_name], ort_inputs)

        id = np.argmax(ort_outputs[0][0][0])

        return self.tokenizer.decode(id, skip_special_tokens=True)
        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='export llm to various onnx.')
    
    parser.add_argument('-m', '--model', type=str, required=True, help='model path')
    parser.add_argument('-o', '--onnx', type=str, required=False, default='qwen.onnx', help='onnx path')
    parser.add_argument('--force_export', action='store_true', help='force export onnx model, not use cache')

    args = parser.parse_args()

    logging.info(f'use model: {args.model}')

    qwen2onnx = Qwen2Onnx()
    qwen2onnx.load(args.model)

    logging.info('step 1: run torch model...\n')
    text = qwen2onnx.run('简要介绍3种杭州美食.')
    logging.info(f'output: {text}')
    logging.info('run torch model done.\n')

    if os.path.exists(args.onnx) and not args.force_export:
        logging.info(f'onnx model exists: {args.onnx}, skip export.')
    else:
        logging.info('step 2: export onnx model...\n')
        qwen2onnx.export(args.onnx)
        logging.info(f'export onnx model done: {args.onnx}')

    logging.info('step 3: run onnx model...\n')
    text = qwen2onnx.run_onnx(args.onnx)
    logging.info(f'output: {text}')
    logging.info(f'run onnx model done.')    