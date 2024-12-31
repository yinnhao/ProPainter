```shell
# 输入mask是一张图片
python inference_propainter_ffmpeg.py --video /root/paddlejob/workspace/ProPainter/inputs/video_completion/zhongguojie.mp4 --mask /root/paddlejob/workspace/ProPainter/inputs/video_completion/zhongguojie_mask.png --height 1280 --width 720
# 输入mask是一个json文件
python inference_propainter_ffmpeg.py --video ./dy_logo_case/videos/video_3.mp4 --mask ./dy_logo_case/videos/video_3.json
```