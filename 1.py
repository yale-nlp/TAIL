import json

with open("data/raw.json", "r") as f:
    data = json.load(f)

for item in data:
    print(len(item["text"]))