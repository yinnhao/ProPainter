import gradio as gr
import cv2
import numpy as np

def create_mask(image, x1: int, y1: int, x2: int, y2: int):
    # 创建全黑的mask
    mask = np.zeros_like(image)
    
    # 确保坐标为整数
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # 在选定区域填充白色
    mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 255
    
    return mask

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 图片区域选择工具")
    
    with gr.Row():
        input_image = gr.Image(label="上传图片并在图片上拖动选择区域", tool="select")
        output_image = gr.Image(label="生成的Mask")
    
    # 在 Gradio 3.50.0 中，select 事件会直接返回选择框的坐标
    input_image.select(
        fn=create_mask,
        inputs=[input_image],
        outputs=output_image
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=8080,
    debug=True
)