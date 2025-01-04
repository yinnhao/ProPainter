import gradio as gr
import cv2
import numpy as np

def create_mask(image, evt: gr.SelectData):
    # 获取选择框的坐标
    x1, y1 = evt.index[0]  # 起始点
    x2, y2 = evt.index[1]  # 结束点
    
    # 创建全黑的mask
    mask = np.zeros_like(image)
    
    # 在选定区域填充白色
    mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 255
    
    return mask

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 图片区域选择工具")
    
    with gr.Row():
        input_image = gr.Image(label="上传图片并在图片上拖动选择区域")
        output_image = gr.Image(label="生成的Mask")
    
    # 当在输入图片上进行选择时触发create_mask函数
    input_image.select(create_mask, input_image, output_image)

demo.launch(server_name="0.0.0.0",
        server_port=8080,
        debug=True)