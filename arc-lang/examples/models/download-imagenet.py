import requests
import json

url = 'https://storage.googleapis.com/download.tensorflow.org' \
    '/data/imagenet_class_index.json'
r = requests.get(url, allow_redirects=True)
data = json.loads(r.content)
result = []

for i in range(len(data)):
    key = str(i)
    if key in data:
        result.append(data[key][1])
    else:
        print(f"Key {key} does not exist.")

with open("imagenet_class_index.json", "w") as f:
    json.dump(result, f)
