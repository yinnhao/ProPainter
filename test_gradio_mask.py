import gradio as gr
import cv2
import numpy as np

def create_mask(image_and_mask):
    # 从gradio的sketch输入中分离图片和mask
    if image_and_mask is None:
        return None
    
    image, mask = image_and_mask["image"], image_and_mask["mask"]
    
    # 如果没有mask，返回全黑图像
    if mask is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 将涂抹的mask转换为二值图像
    mask = (mask > 128).astype(np.uint8) * 255
    # 确保mask是2D数组
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    return mask

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 图片涂抹工具")
    
    with gr.Row():
        input_image = gr.Image(label="上传图片并在图片上涂抹选择区域", tool="sketch", source="upload")
        output_image = gr.Image(label="生成的Mask")
    
    # 使用change事件来处理涂抹的结果
    input_image.change(
        fn=create_mask,
        inputs=input_image,
        outputs=output_image
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=8080,
    debug=True
)