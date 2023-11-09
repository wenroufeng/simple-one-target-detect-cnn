import os
import json
import sys


def mkd(shape, val):
    for i, img in enumerate(os.listdir(f'data/{shape}s/')):
        location = gt_content[img]
        left = location['left']
        top = location['top']
        right = location['right']
        bottom = location['bottom']
        if i < 500:
            os.system(f"cp data/{shape}s/{img} dataset/train/")
            ft.write(f'''dataset/train/{img} {val} {left} {top} {right} {bottom}\n''')
        elif i < 750:
            os.system(f"cp data/{shape}s/{img} dataset/val/")
            fv.write(f'dataset/val/{img} {val} {left} {top} {right} {bottom}\n')


if __name__ == '__main__':
    shape = sys.argv[1]
    os.makedirs(f'dataset/train/', exist_ok=True)
    os.makedirs(f'dataset/val/', exist_ok=True)

    with open(f'data/{shape}s.json', 'rt') as fr:
        gt_content = json.load(fr)

    ft = open(f'dataset/train.txt', 'at')
    fv = open(f'dataset/val.txt', 'at')

    from shape_map import cls_map
    # for shape in cls_map:
    
    mkd(shape, cls_map[shape])

    ft.close()
    fv.close()

