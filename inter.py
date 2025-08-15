import torch
import numpy as np
# import os
# import argparse
from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from improved_model import ImprovedIdentify

# transform my data in the same way as training
val_transform = Compose([
    Resize(300, 300),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 复用训练时的归一化参数
    ToTensorV2()
])


def load_single_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # 确保RGB格式
            img_np = np.array(img)  # 转为numpy数组（HWC格式）

        augmented = val_transform(image=img_np)
        img_tensor = augmented['image']  # 经过Resize、归一化、转为CHW张量

        # [3, 300, 300] -> [1, 3, 300, 300]
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_path', type=str, required=True, help='单张测试图片的路径')
#     return parser.parse_args()


def identify(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class_names = ["paper", "scissors", "rock"]
    num_classes = len(class_names)

    # 加载单张图片
    img_tensor = load_single_image(image_path)
    if img_tensor is None:
        return

    try:
        model = ImprovedIdentify(num_classes=num_classes)
        state_dict = torch.load(r"C:\Users\72472\Desktop\Cambridge\improved_model-63%.pth", map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 执行预测
    try:
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, y_pred = torch.max(outputs, 1)

        predicted_label = y_pred.item()
        return class_names[predicted_label]

    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        return None


# if __name__ == "__main__":
#     result = main()
#     if result:
#         print(result)