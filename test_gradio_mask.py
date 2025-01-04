import gradio as gr
import numpy as np

def on_image_upload(image_dict):
    # 当上传图片时，返回一个全黑的mask
    if image_dict is None or not isinstance(image_dict, dict):
        return None
    
    image = image_dict.get("image")
    if image is None:
        return None
    
    # 返回全黑mask，并确保它是3通道的
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return mask

def on_image_draw(image_and_mask):
    # 处理涂抹事件
    if image_and_mask is None or not isinstance(image_and_mask, dict):
        return None
    
    image = image_and_mask.get("image")
    mask = image_and_mask.get("mask")
    
    if image is None:
        return None
        
    if mask is None:
        return np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # 将涂抹的mask转换为二值图像
    mask = (mask > 128).astype(np.uint8) * 255
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    # 转换为3通道
    mask = np.stack([mask, mask, mask], axis=2)
    return mask

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## 图片涂抹工具")
    
    with gr.Row():
        input_image = gr.Image(
            label="上传图片并在图片上涂抹选择区域",
            tool="sketch",
            source="upload",
            type="numpy"
        )
        output_image = gr.Image(
            label="生成的Mask",
            type="numpy"
        )
    
    # 分别处理上传和涂抹事件
    input_image.upload(
        fn=on_image_upload,
        inputs=input_image,
        outputs=output_image
    )
    
    input_image.edit(
        fn=on_image_draw,
        inputs=input_image,
        outputs=output_image
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=8080,
    debug=True
)