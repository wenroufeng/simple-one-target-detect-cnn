
import fiftyone as fo
import sys
import os
import json


shape = sys.argv[1]

dataset = fo.Dataset.from_images_dir(f'data/{shape}s/')

with open(f'data/{shape}s.json', 'rt') as f:
    json_content = json.load(f)
    for s in dataset.iter_samples(autosave=True):
        img_basename = os.path.basename(s.filepath)
        location = json_content[img_basename]
        detections = fo.Detection(label=shape, bounding_box=(
            location['left'] / 30, location['top'] / 30, 
            (location['right'] - location['left']) / 30, 
            (location['bottom'] - location['top']) / 30, 
            ))
        s['ground_truth'] = fo.Detections(detections=[detections])

session = fo.launch_app(dataset, address="10.158.99.23", port=8000) # type: ignore
session.wait()

# while True:
#     a = input('>>>>>>')
#     if a:
#         if not session.view:
#             print("请确定过滤")
#             continue
#         with open('del.txt', 'wt') as f:
#             for s in session.view:
#                 f.write(s.filepath + '\n')
#                 print(s.filepath)
