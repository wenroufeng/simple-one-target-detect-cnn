import random
from PIL import Image, ImageDraw
import os
import json


def create_triangle():
    os.makedirs('data/triangles/', exist_ok=True)
    json_content = {}
    with open('data/triangles.json', 'wt') as f:
        for i in range(1000):
            # 创建一张白色背景的图像，大小为300x300像素
            width, height = 300, 300
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)
            x1, y1 = random.randint(0, 100), random.randint(0, 100)
            # diameter = random.randint(30, 200)
            # x2, y2 = x1 + diameter, y1 + diameter

            x1, y1 = random.randint(0, width // 2 - 100), random.randint(100, 200)
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            x3, y3 = x2 + random.randint(10, width // 2 - 50), y1 - random.randint(10, height // 2 - 50)

            # 绘制锐角三角形，可以修改颜色
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], outline='black', fill=None)

            json_content[f'{i}_triangle.png'] = {
                'left': round(min(x1, x2, x3) / 15), 'top': round(min(y1, y2, y3) / 15), 
                'right': round(max(x1, x2, x3) / 15), 'bottom': round(max(y1, y2, y3) / 15)
            }
            # 保存生成的图像
            image.save(f'data/triangles/{i}_triangle.png')
        json.dump(json_content, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    create_triangle()
