import os
import gradio as gr
import torch
import tempfile
import json
from PIL import Image
import numpy as np

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

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="ProPainter Video Inpainting") as app:
        gr.Markdown("# ProPainter Video Inpainting")
        gr.Markdown("Upload a video and mask to remove unwanted objects")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")
                mask_input = gr.File(label="Mask (JSON or Image)")
                
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
                
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                mask_input,
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
    app.launch(server_name="0.0.0.0",
        server_port=8080,
        debug=True)
