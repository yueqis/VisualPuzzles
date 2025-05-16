from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import datetime
from PIL import Image
import random
import numpy as np
import re
from io import BytesIO
import base64
import json
import logging

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

start=950
end=1200
prompt = 'MULTI_CHOICE_DIRECT_PROMPT' #Change to other prompts

logging.info(f"qvq - start: {start}; end: {end}; prompt: {prompt}")

out_file = 'qvq-direct.jsonl'
image_dir = '/'
model_name = 'Qwen/QVQ-72B-Preview'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

MULTI_CHOICE_DIRECT_PROMPT = "Answer the question with the option's letter from the given choices directly."
MULTI_CHOICE_PROMPT = "Answer the question with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options."
COT_PROMPT = "Solve the multiple-choice question and then answer with the option letter from the given choices. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
PROMPTS = {'MULTI_CHOICE_DIRECT_PROMPT': MULTI_CHOICE_DIRECT_PROMPT, 'MULTI_CHOICE_PROMPT': MULTI_CHOICE_PROMPT, 'COT_PROMPT': COT_PROMPT}

def reasoning_doc_to_visual(doc):
    image_path = doc['image_path']
    image_path = image_dir + image_path
    image = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode("utf-8")
    return base64_string

def reasoning_doc_to_text(doc, prompt):
    question = 'Question: ' + doc["question"].strip()
    options = doc['options']
    if options != None: question += '\nOptions:\n(A) ' + options[0] + '\n(B) ' + options[1] + '\n(C) ' + options[2] + '\n(D) ' + options[3]
    else: question += '\nOptions: Choose from (A) (B) (C) (D) in the image.'
    question += '\n' + PROMPTS[prompt]
    return question

def get_messages(doc, prompt):
    image = reasoning_doc_to_visual(doc)
    question = reasoning_doc_to_text(doc, prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{image}",
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages

with open('data.json') as f: data = json.load(f)[start:end]
with open(out_file) as f: 
    f = f.readlines()
    ids = [json.loads(row)['id'] for row in f]
for row in data:
    if row['id'] in ids: continue
    try:
        logging.info(row['id'])
        messages = get_messages(row, prompt)
        # Change the following code to that of any model's generation code
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        with open(out_file, 'a') as f: f.write(json.dumps({'id': row['id'], 'resps': output_text, 'target': row['answer']}) + '\n')
    except Exception as e: logging.info(f"error: {e}")

