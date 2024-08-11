import json
from openai import OpenAI  
import numpy as np
import os
import pandas as pd

client = OpenAI(base_url="https://yanlp.zeabur.app/v1", api_key=os.environ["OPENAI_API_KEY"])

def get_embedding(text, model="text-embedding-3-large"):
   return client.embeddings.create(input = [text], model=model).data[0].embedding

print(get_embedding("hello world"))