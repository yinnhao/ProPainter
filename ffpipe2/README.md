## ffpipe

### 简介
python调用ffmpeg进行视频处理，分析，模型推理，功能类似于ffmpeg filter；
- 基于python可快速验证模型效果
- 解耦视频推理框架与图像处理算法 


本地模块的发布
```
python setup.py sdist
python setup.py install
```

### 使用
1. 继承video_infer类，重写forward函数
```
from ffpipe import video_infer
```
2. video_infer需要设定的参数

```python
def __init__(self, file_name, save_name, encode_params, model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p10le", **decode_param_dict):
```

* `file_name`: 输入文件名
* `save_name`: 输出保存的文件名
* `encode_params`: 输出文件编码参数，以下格式：

sdr x264: 
```
encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
```
sdr x265: 
```
encode_params = ("libx265", "x265-params", "qp=12:bframes=3")
```
hdr x265:
```
encode_params = ("libx265", "x265-params", "hrd=1:aud=1:no-info=1:sar='1:1':colorprim='bt2020':transfer='smpte2084':colormatrix='bt2020nc':master-display='G(8500,39850)B(6500,2300)R(35400,14600)WP(15635,16450)L(0,0)':max-cll='0,0':no-open-gop=1:qp=12:bframes=3")
```

* `model`: 如果是使用模型推理进行视频增强，要传入模型
* `scale`: 尺度参数
* `in_pix_fmt`: 输入的pix_fmt，也就是ffmpeg解码返回的帧的格式，一般就是设成yuv444
* `out_pix_fmt`: 输出的pix_fmt
* `**decode_param_dict`: 解码的参数字典

```python
decode_parmas_dict = {"r":"{}".format(str(decode_fps))} 
gray = gray_video_infer(in_path, out_path, encode_params, out_pix_fmt="yuv444p", **decode_parmas_dict)
```

3.使用案例

```python

from ffpipe import video_infer
class gray_video_infer(video_infer):
    def __init__(self, in_path, out_path, encode_params,  model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p10le", **decode_param):
        super(gray_video_infer, self).__init__(in_path, out_path, encode_params,  model, scale, in_pix_fmt, out_pix_fmt, **decode_param)

    def forward(self, x):
        y = x.copy()
        y[1, :, :] = 128
        y[2, :, :] = 128
        return y

in_path = "/data/yh/video/gongxun_1_15s.mov"
out_path = "/data/yh/video/gongxun_1_15s_gray_3.mp4"
encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
decode_fps = 1
decode_parmas_dict = {"r":"{}".format(str(decode_fps))} 
gray = gray_video_infer(in_path, out_path, encode_params, out_pix_fmt="yuv444p", **decode_parmas_dict)
gray.infer()
```

### V1.1更新
```
在解码的进程增加多一些参数，比如-r等，用于视频分析不需要每帧都分析的情况，通过修改以下函数实现：
start_ffmpeg_process1

调用时新增decode解码参数
decode_parmas_dict = {"r":"{}".format(str(decode_fps))} 
# no decode_parmas_dict
# gray = gray_video_infer(in_path, out_path, encode_params, out_pix_fmt="yuv444p")
# decode_parmas_dict
gray = gray_video_infer(in_path, out_path, encode_params, out_pix_fmt="yuv444p", **decode_parmas_dict)
```