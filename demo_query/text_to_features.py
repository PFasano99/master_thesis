import json
import pathlib
from dataclasses import dataclass
from typing import List, Union

import matplotlib
import numpy as np
import open_clip
import torch
import tyro
from typing_extensions import Literal

import os

# Path to saved pointcloud to visualize
load_path: str

device: str = "cpu"

# Similarity computation and visualization params
viz_type: Literal["topk", "thresh"] = "thresh"
similarity_thresh: float = 0.6
topk: int = 10000

# CLIP model config
open_clip_model = "ViT-H-14"
open_clip_pretrained_dataset = "laion2b_s32b_b79k"


if __name__ == "__main__":

    print(
        f"Initializing OpenCLIP model: {open_clip_model}"
        f" pre-trained on {open_clip_pretrained_dataset}..."
    )
    model, _, _ = open_clip.create_model_and_transforms(
        open_clip_model, open_clip_pretrained_dataset
    )
    model.cpu()
    model.eval()

    tokenizer = open_clip.get_tokenizer(open_clip_model)

    prompt_texts = ["table","show me a table", "where can i sit", "where is the kitchen?"]

    save_path = "./resources/dataset/cnr_c60/prompt_feat"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'Folder created at: {save_path}')

    i=0
    for prompt_text in prompt_texts[:1]:
        text = tokenizer([prompt_text])
        textfeat = model.encode_text(text)
        textfeat = torch.nn.functional.normalize(textfeat.float(), dim=-1)
        #textfeat = textfeat.unsqueeze(0)

        print(textfeat.shape)

        # Convert the tensor to a NumPy array
        numpy_array = textfeat.detach().numpy()


        file_name = save_path + "/prompt_"+str(i)+".bin"
        # Save the NumPy array to a binary file
        with open(file_name, 'wb') as f:
            numpy_array.tofile(f)
        
        i+=1

        

