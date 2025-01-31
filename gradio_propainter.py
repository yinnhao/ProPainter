import os
import gradio as gr
import torch
import tempfile
import json
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

from inference_propainter_ffmpeg import (
    video_infer_propainter,
    load_file_from_url,
    RAFT_bi,
    RecurrentFlowCompleteNet,
    InpaintGenerator,
    get_device
)

# 全局变量
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
device = get_device()

def load_models():
    """加载所需的模型"""
    # 加载RAFT
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'raft-things.pth'),
        model_dir='weights',
        progress=True
    )
    fix_raft = RAFT_bi(ckpt_path, device)
    
    # 加载flow completion模型
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
        model_dir='weights',
        progress=True
    )
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    # 加载ProPainter模型
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
        model_dir='weights',
        progress=True
    )
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()
    
    return fix_raft, fix_flow_complete, model

class Args:
    """参数类"""
    def __init__(self, **kwargs):
        self.ref_stride = kwargs.get('ref_stride', 10)
        self.neighbor_length = kwargs.get('neighbor_length', 10)
        self.subvideo_length = kwargs.get('subvideo_length', 80)
        self.raft_iter = kwargs.get('raft_iter', 20)
        self.save_frames = kwargs.get('save_frames', False)

def process_video(video_path, mask_path, resize_ratio=1.0, mask_dilation=4, 
                 ref_stride=10, neighbor_length=10, subvideo_length=80,
                 raft_iter=20, use_fp16=False, progress=gr.Progress()):
    """处理视频的主函数"""
    progress(0, desc="Loading models...")
    
    # 加载模型
    fix_raft, fix_flow_complete, model = load_models()
    
    # 创建临时输出文件
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_output.name
    
    # 设置参数
    args = Args(
        ref_stride=ref_stride,
        neighbor_length=neighbor_length,
        subvideo_length=subvideo_length,
        raft_iter=raft_iter,
        save_frames=False
    )
    
    # 读取mask信息
    if mask_path.endswith('.json'):
        with open(mask_path, 'r') as f:
            mask_info = json.load(f)
    else:
        mask_info = mask_path
        
    progress(0.2, desc="Processing video...")
    
    # 设置编码参数
    encode_params = ("libx264", "x264opts", "qp=24:bframes=3")
    
    # 创建video_infer实例
    video_infer = video_infer_propainter(
        video_path,
        output_path,
        encode_params,
        model=model,
        scale=resize_ratio,
        mask_info=mask_info,
        fix_raft=fix_raft,
        fix_flow_complete=fix_flow_complete,
        args=args,
        use_half=use_fp16,
        mask_dilation=mask_dilation,
        frames_len=15  # N_in 参数
    )
    
    # 执行处理
    progress(0.4, desc="Inpainting in progress...")
    video_infer.infer_multi_frames_propainter(15)
    
    progress(1.0, desc="Done!")
    return output_path

def extract_frame(video_path, frame_number=0):
    """从视频中提取指定帧"""
    # 创建一个默认的空白图像
    empty_frame = np.zeros((500, 800, 3), dtype=np.uint8)
    
    if not video_path:
        return empty_frame
        
    if hasattr(video_path, 'name'):
        video_path = video_path.name
        
    try:
        cap = cv2.VideoCapture(video_path)
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 确保frame_number在有效范围内
        frame_number = max(0, min(frame_number, total_frames - 1))
        
        # 设置读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            return empty_frame
            
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return empty_frame

def create_mask_from_rect(image, rect_data):
    """根据矩形坐标创建mask"""
    if image is None or rect_data is None:
        return None
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, [
        rect_data['x1'], rect_data['y1'],
        rect_data['x2'], rect_data['y2']
    ])
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def save_mask(mask):
    """保存mask为图片文件"""
    if mask is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_save_folder = "./gradio_mask"
    mask_path = f"{mask_save_folder}/temp_mask_{timestamp}.png"
    cv2.imwrite(mask_path, mask)
    return mask_path

def on_image_draw(image_and_mask):
    """处理涂抹事件，生成mask"""
    if image_and_mask is None or not isinstance(image_and_mask, dict):
        return None, None
    
    image = image_and_mask.get("image")
    mask = image_and_mask.get("mask")
    
    if image is None:
        return None, None
        
    if mask is None:
        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        return mask, None
    
    # 将涂抹的mask转换为二值图像
    mask = (mask > 128).astype(np.uint8) * 255
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    # 保存mask并返回路径
    mask_path = save_mask(mask)
    # 转换为3通道显示
    mask_display = np.stack([mask, mask, mask], axis=2)
    return mask_display, mask_path

def clear_mask():
    """清除mask的函数"""
    # 返回一个空白的黑色图像作为mask预览，和None作为mask路径
    empty_mask = np.zeros((500, 800, 3), dtype=np.uint8)  # 创建空白图像
    return empty_mask, None

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="Draw Mask and Video Inpainting based on ProPainter") as app:
        gr.Markdown("# ProPainter Video Inpainting")
        gr.Markdown("Upload a video and draw mask to remove unwanted objects")
        
        with gr.Row():
            with gr.Column():
                # 设置最大宽度为480，高度设为None以自动计算
                video_input = gr.Video(label="Input Video", width=480, height=None)

                with gr.Row():
                    extract_btn = gr.Button("Extract Frame")
                    frame_number = gr.Number(
                        label="Frame Number", 
                        value=0, 
                        minimum=0, 
                        step=1,
                        precision=0
                    )
                
                first_frame = gr.Image(
                    label="Selected Frame (Draw mask here)",
                    type="numpy",
                    tool="sketch",
                    width=480,
                    height=None
                )
                # mask预览也使用相同的尺寸
                mask_preview = gr.Image(
                    label="Mask Preview",
                    type="numpy",
                    interactive=False,
                    width=480,
                    height=None
                )
                
                
                mask_path = gr.State(None)
                
                with gr.Row():
                    clear_mask_btn = gr.Button("Clear Mask")
                
                with gr.Row():
                    resize_ratio = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, 
                                          label="Resize Ratio")
                    mask_dilation = gr.Slider(minimum=0, maximum=20, value=4, step=1,
                                           label="Mask Dilation")
                
                with gr.Row():
                    ref_stride = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                        label="Reference Stride")
                    neighbor_length = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                             label="Neighbor Length")
                
                with gr.Row():
                    subvideo_length = gr.Slider(minimum=10, maximum=200, value=80, step=10,
                                             label="Subvideo Length")
                    raft_iter = gr.Slider(minimum=5, maximum=50, value=20, step=5,
                                       label="RAFT Iterations")
                
                use_fp16 = gr.Checkbox(label="Use FP16 (Half Precision)", value=False)
                
                process_btn = gr.Button("Process Video")
            
            with gr.Column():
                video_output = gr.Video(label="Output Video")
        
        # 事件处理
        extract_btn.click(
            fn=extract_frame,
            inputs=[video_input, frame_number],
            outputs=[first_frame]
        )
        
        # mask绘制事件（在首帧上涂抹）
        first_frame.edit(
            fn=on_image_draw,
            inputs=[first_frame],
            outputs=[mask_preview, mask_path]  # 更新mask预览和保存路径
        )
        
        clear_mask_btn.click(
            fn=lambda: (None, None),
            inputs=[],
            outputs=[mask_preview, mask_path]
        )
        
        # 修改process_video的调用
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                mask_path,  # 使用生成的mask路径
                resize_ratio,
                mask_dilation,
                ref_stride,
                neighbor_length,
                subvideo_length,
                raft_iter,
                use_fp16
            ],
            outputs=video_output
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.queue()  # 先启用队列
    app.launch(
        server_name="0.0.0.0",
        server_port=8080,
        debug=True
    )
