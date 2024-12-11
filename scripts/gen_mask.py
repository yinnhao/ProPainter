import numpy as np
import cv2

def create_mask_with_bbox(image_size, bbox):
    """
    创建带有指定bounding box的mask图片
    
    参数:
    image_size: tuple (height, width) - 图片尺寸
    bbox: tuple (x, y, w, h) - bounding box坐标和尺寸
    
    返回:
    mask图片 (黑色背景，白色bbox)
    """
    # 创建黑色背景
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # 解析bbox坐标
    x, y, w, h = bbox
    
    # 在mask上绘制白色矩形
    mask[y:y+h, x:x+w] = 255
    
    return mask

# 使用示例
if __name__ == "__main__":
    # 设置图片尺寸 (高度, 宽度)
    h, w = 1024, 576
    x1 = (125, 139)
    x2 = (179,  450)
    img_size = (h, w)
    
    # 设置bounding box (x, y, 宽度, 高度)
    bbox = (x1[1], x1[0], x2[1]-x1[1], x2[0]-x1[0])
    
    # 创建mask
    mask = create_mask_with_bbox(img_size, bbox)
    
    # 显示结果
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 保存mask
    cv2.imwrite('./inputs/video_completion/shuijiaoxian_mask.png', mask)