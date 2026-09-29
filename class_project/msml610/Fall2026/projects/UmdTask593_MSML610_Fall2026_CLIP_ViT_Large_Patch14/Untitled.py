# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
os.environ["HF_HOME"] = "cache/hf"

from transformers import CLIPModel, CLIPProcessor
model_id = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id).eval()
processor = CLIPProcessor.from_pretrained(model_id)
print(model.config.projection_dim)
