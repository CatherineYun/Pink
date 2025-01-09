from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria, CLIPModel
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import os
import sys
sys.path.append("./")
from pink import *
import base64
import pandas as pd
import numpy as np
import torch
import json

import io
from PIL import Image
import random
import math
from pink.datasets.Templates import ChoiceQuestionAnswer
from pink.conversation import conv_llava_v1, conv_simple_v1_mmbench, conv_llama2, conv_simple_v1
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re

# In[3]:


model_name = "/data/Katherine/data/Pink/base_fgvc_2025_01_07_2"
config = AutoConfig.from_pretrained(model_name, use_cache=True)
config.llama_path = ""
if config.llama_path != model_name:
    # need to merge parameters
    llama_path = '/data/Katherine/data/Llama-2-7b-chat-hf_2'
    weight_map_index = json.load(open(os.path.join("/data/Katherine/data/Llama-2-7b-chat-hf_2", "pytorch_model.bin.index.json"), "r"))
    shard_files = list(set(weight_map_index["weight_map"].values()))
    loaded_keys = weight_map_index["weight_map"].keys()
    state_dict = {}
    for index, shard_file in enumerate(shard_files):
        state_dict.update(torch.load(os.path.join(llama_path, shard_file), map_location="cpu"))
    peft_parameters = torch.load(os.path.join(model_name, "saved_parameters.pth"), map_location="cpu")
    for k, v in peft_parameters.items():
        state_dict[k] = v
else:
    state_dict = None

model = AutoModelForCausalLM.from_pretrained(None, config=config, state_dict=state_dict)
for name, param in model.model.named_parameters():
    if not ("adapter_" in name or "lora_" in name):
        param.data = param.data.half()
model.lm_head.to(torch.float16)
model = model.cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')


# In[4]:


from pink.conversation import conv_llama2
image_processor = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
image_token_len = model.config.num_patches

model.eval()
conv = conv_llama2.copy()
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
BEGIN_LOC = "<loc>"
END_LOC = "</loc>"
BEGIN_CLS = "<cls>"
END_CLS = "</cls>"
BEGIN_RELATION = "<rel>"
END_RELATION = "</rel>"
BEGIN_QUESTION = "<qes>"
END_QUESTION = "</qes>"
DEFAULT_EOS_TOKEN = "</s>"
# Set the question for inference
question = "Please just tell me the model of this plane."

# Define the image directory
image_dir = "/data/Katherine/data/FGVC-Aircraft/fgvc-aircraft-2013b/data/images/"
# Results will be stored in this list
results = []
image_files = [
    "0800390", "1320565", "0923499", "1298910",
    "0523227", "0973160", "0104604", "0198448",
    "1616362", "1296580", "1051041", "0906979",
    "0457162", "2158992", "0198446", "0944176",
    "1176955", "0784557", "0197892", "1695906",
    "1544178", "0445606", "1885835", "0784479",
    "0725715", "1430022", "0995387", "0487393",
    "0383400", "1117062", "0329381", "0523192",
    "0810303"
]

image_files2 = [
    "0829813", "0523278", "1590457", "0523139",
    "1140416", "1297059", "1083282", "0677638",
    "0974359", "0063293", "1061208", "1351441",
    "0923714", "0522967", "1691610", "0836257",
    "0894205", "0958066", "0094396", "0719053",
    "0738982", "0522975", "0693457", "0324987",
    "0504826", "1594717", "0522969", "0659112",
    "1619881", "0066418", "0551251", "0195009",
    "0845036"]
image_files3=["1940728", "0066405", "1412918",
    "0582362", "0808929", "0907404", "0523052",
    "0903662", "0064200", "0879893", "0836159",
    "0983420", "1204567", "0337964", "1298949",
    "0184897", "1177961", "0229256", "0523189",
    "1415738", "1388984", "1273293", "0391828",
    "1481131", "2007881", "0255359", "0384438",
    "0979615", "0984879", "0522973", "0066406",
    "0936117", "1185509", "1633282"
]
# Iterate over all images in the directory
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file+".jpg")
    
    loc_pattern = re.compile(r"(\[[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9],[0-9].[0-9][0-9][0-9]\])")
    bbox_pattern = re.compile(r"[0-9].[0-9][0-9][0-9]")

    with open(image_path, "rb") as f:
        image = Image.open(io.BytesIO(f.read())).convert('RGB')
    width, height = image.size

    copy_image1 = image.copy()

    image_tensor = image_processor(image)
    images = image_tensor.unsqueeze(0).cuda()
    conv.messages = []
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    conv.set_system(PREFIX_IMAGE + image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
    cur_prompt = conv.get_prompt()
    print(cur_prompt)

    tokenized_output = tokenizer(
        [cur_prompt],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
    attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            has_images=[True],
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            temperature=0.7,
            max_new_tokens=1024,
        )

    for input_id, output_id in zip(input_ids, output_ids):
        input_token_len = input_id.shape[0]
        n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
        output = output.strip()
        print(output)
        result_entry = {
            "image_name": image_file,
            "output": output
        }
        results.append(result_entry)

# Save results to JSON file
with open('/data/Katherine/Pink/out/results_of_1_07_1_2.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Inference results saved to results.json.")
