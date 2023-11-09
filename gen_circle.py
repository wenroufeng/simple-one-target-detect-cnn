import random
from PIL import Image, ImageDraw
import os
import json
import math


def create_circle():
    os.makedirs('data/circles/', exist_ok=True)
    with open('data/circles.json', 'wt') as f:
        json_content = {}
        for i in range(1000):
            # 创建一张白色背景的图像，大小为300x300像素
            width, height = 300, 300
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)
            x1, y1 = random.randint(0, 100), random.randint(0, 100)
            diameter = random.randint(30, 200)
            x2, y2 = x1 + diameter, y1 + diameter

            draw.ellipse([(x1, y1), (x2, y2)], outline='black', fill=None)

            # 保存生成的图像
            image.save(f'data/circles/{i}_circle.png')
            print((x1, y1), (x2, y2))
            json_content[f'{i}_circle.png'] = {
                'left': round(x1 / 15), 'top': round(y1 / 15), 
                'right': round(x2 / 15), 'bottom': round(y2 / 15)
            }
            # break
        json.dump(json_content, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    create_circle()
