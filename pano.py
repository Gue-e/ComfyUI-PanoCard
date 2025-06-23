
import math
import re
import numpy as np
import comfy
import nodes

from .misc import *
from .distort import *

from PIL import ImageDraw as ImageW
from PIL import Image as ImageF

import torch
import torchvision.transforms.functional as TF

import sys
from os import path

# 添加父目录到模块搜索路径
sys.path.insert(0, path.dirname(__file__))
from folder_paths import get_save_image_path, get_output_directory

# 获取当前文件所在目录的父目录（即 custom_nodes 目录）
parent_dir = path.abspath(path.join(path.dirname(__file__), '..'))

# 手动添加 Impact Pack 的路径
impact_pack_path = path.join(parent_dir, 'comfyui-impact-pack')
if impact_pack_path not in sys.path:
    sys.path.append(impact_pack_path)

# 导入DetailerHook
from modules.impact.hooks import DetailerHook

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

# 用于将tensor图像扩展
def transpose_tensor(image, left, top, right, bottom, method):
    tensor = image.clone().detach()
    tensor_pad = TF.pad(
        tensor.permute(2, 0, 1), 
        [left, top, right, bottom], 
        padding_mode=method
        ).permute(1, 2, 0)
    return tensor_pad

# 调整图像大小的辅助函数。
def resize_tensor(image, H, W):
    image = image.permute(0, 3, 1, 2)
    image = TF.resize(image, (H, W))
    image = image.permute(0, 2, 3, 1)
    return image

# 获取拼接前的6个张量
def get_original_tensors(tensors):
    num_tensors = 6
    total_length = tensors.shape[0]
    tensor_size = total_length // num_tensors
    
    # 计算每个张量的起始和结束索引
    start_indices = [i * tensor_size for i in range(num_tensors)]
    end_indices = [(i + 1) * tensor_size for i in range(num_tensors)]
    
    # 使用 zip 将 start_indices 和 end_indices 打包成元组列表
    zipped_indices = list(zip(start_indices, end_indices))
    
    # 使用列表推导式从 tensors 中提取原始张量
    original_tensors = [tensors[start_index:end_index] for start_index, end_index in zipped_indices]
    
    return original_tensors

# 定义居中位置，支持不同的背景尺寸和对齐方式
def center_position(h, w, bg_h, bg_w, align='center'):
    if align == 'center':
        center_h = (bg_h - h) / 2
        center_w = (bg_w - w) / 2
    elif align == 'top_left':
        center_h, center_w = 0, 0
    elif align == 'top_right':
        center_h, center_w = 0, bg_w - w
    elif align == 'bottom_left':
        center_h, center_w = bg_h - h, 0
    elif align == 'bottom_right':
        center_h, center_w = bg_h - h, bg_w - w
    else:
        raise ValueError("Unsupported alignment option. Choose from 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'.")
    
    return int(center_h), int(center_w)

# 将每张图片居中叠加到背景张量上,images 是一个列表，每个元素都是一个张量
def overlay_images(images, bg_h, bg_w, align='center'):
    # 初始化背景张量
    stacked_images = torch.zeros(len(images), bg_h, bg_w, 4, dtype=torch.float32)
    
    for i, image_i in enumerate(images):
        if image_i is None:
            print("None image:", i)
            continue
        
        # 获取图片的高度和宽度
        h, w = image_i.shape[-3], image_i.shape[-2]
        
        # 计算居中位置
        center_h, center_w = center_position(h, w, bg_h, bg_w, align)
        
        # 将图片居中叠加到背景张量上
        stacked_images[i, center_h:center_h+h, center_w:center_w+w, :] = image_i
    
    return stacked_images

def DepackClip(cond_face):
    res = []
    [b_conds, b_pooleds_dict] = cond_face[0]
    if "pooled_output" in b_pooleds_dict:
        b_pooleds = b_pooleds_dict["pooled_output"]
    else:
        raise Exception("no pooled_output")
    
    conds = get_original_tensors(b_conds)
    if len(conds) != 6:
        raise Exception("必须包含6个条件")
    
    pooleds = get_original_tensors(b_pooleds)

    key_pooleds = 0.0
    if "guidance" in b_pooleds_dict:
        key_pooleds = b_pooleds_dict["guidance"]
    else:
        print("no guidance")

    cnet_list = None
    if 'pano_control' in b_pooleds_dict:
        cnet_list = b_pooleds_dict["pano_control"]
        print("has pano_control number:",len(cnet_list))
    else:
        print("no pano_control")

    for i in range(6):
        if key_pooleds > 0.0:
            cond_dict = {"pooled_output": pooleds[i], "guidance": key_pooleds}
        else:
            cond_dict = {"pooled_output": pooleds[i]}

        if cnet_list is not None:
            if 'control' in cnet_list[i] and 'control_apply_to_uncond' in cnet_list[i]:
                cond_dict['control'] = cnet_list[i]['control']
                cond_dict['control_apply_to_uncond'] = cnet_list[i]['control_apply_to_uncond']

        res.append([(conds[i], cond_dict)])
    return res


def pad_conditions(conds, num_tokens=0):
    if len(conds) == 0:
        return []

    # 找到最大的 token 数量
    if num_tokens > 0:
        max_num_tokens = num_tokens
    else:
        max_num_tokens = max(tensor.shape[1] for tensor in conds)

    # 创建一个新的列表来存储填充后的张量
    padded_conds = []

    for cond in conds:
        current_num_tokens = cond.shape[1]
        if current_num_tokens < max_num_tokens:
            # 填充较短的张量
            padding_size = max_num_tokens - current_num_tokens
            # 创建填充张量
            padding = torch.zeros(
                (cond.shape[0], padding_size, cond.shape[2]),  # [batch_size, padding_size, token_dim]
                dtype=cond.dtype,
                device=cond.device
            )
            # 拼接原始张量和填充张量
            padded_cond = torch.cat((cond, padding), dim=1)
        elif current_num_tokens > max_num_tokens:
            # 截断较长的张量
            padded_cond = cond[:, :max_num_tokens, :]
        else:
            # 如果长度相同，直接使用原始张量
            padded_cond = cond
        
        padded_conds.append(padded_cond)

    return padded_conds

class PanoViewer:
    @classmethod
    # 定义输入类型
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_image": ("IMAGE",),
            }
        }

    def __init__(self):
        self.saved_pano = []
        # 初始化保存路径相关信息
        self.full_output_folder, self.filename, self.counter, self.subfolder, self.filename_prefix = get_save_image_path("panoimg", get_output_directory())

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process_inputs"
    CATEGORY = "PanoCard"

    def process_inputs(self, pano_image):
        # 清空已保存的HDR图像列表
        self.saved_pano.clear()
        # 处理输入图像，将其转换为NumPy数组
        image = pano_image[0].detach().cpu().numpy()
        # 将数组转换为PIL图像，并确保其为RGB格式
        image = ImageF.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8)).convert('RGB')

        # 返回处理后的图像
        return self.display(image)

    def display(self, pano_image):
        # 生成文件名
        filename_with_counter = f"{self.filename}_{self.counter:05}.png"
        # 拼接完整路径
        image_file = path.join(self.full_output_folder, filename_with_counter)
        # 保存图像
        pano_image.save(image_file)

        # 记录保存的信息
        self.saved_pano.append({
            "filename": filename_with_counter,
            "subfolder": self.subfolder,
            "type": "output",
        })

        # 更新计数器
        self.counter += 1

        # 返回UI展示信息
        return {"ui": {"pano_image": self.saved_pano}}  

class PANO_PIPE:
    def __init__(self, sides, scale, fov, angle, depolar, width, height, scale_up, scale_down, blue, alpha):
        self.sides = sides
        self.scale = scale
        self.fov = fov
        self.angle = angle
        self.depolar = depolar
        self.width = width
        self.height = height
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.blue = blue
        self.alpha = alpha

class PanoImagePipe:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/pad"   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sides": ("INT", {"default": 4, "min": 1, "max": 10}),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "step": 8
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "step": 8
                }),
                "fov": ("FLOAT", {
                    "default": 0.0,
                    "max": 90.0,
                    "min": -90.0,
                    "step": 0.1
                }),
                "depolar": ("FLOAT", {
                    "default": 1.2,
                    "max": 20.0,
                    "min": 0.5,
                    "step": 0.01
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "max": 1.0,
                    "min": 0.1,
                    "step": 0.01
                }),
                "scale_up": ("FLOAT", {
                    "default": 1.0,
                    "max": 1.0,
                    "min": 0.1,
                    "step": 0.01
                }),
                "scale_down": ("FLOAT", {
                    "default": 1.0,
                    "max": 1.0,
                    "min": 0.1,
                    "step": 0.01
                }),
            }
        }
    RETURN_TYPES = ("PANO_PIPE","INT")
    RETURN_NAMES = ("pano_pipe","sides")
    FUNCTION = "switch"

    def switch(self, 
               sides, height, width,
               fov, depolar, scale, 
               scale_up, scale_down):
  
        #单个卡片透视到等距圆柱体上的扭曲幅度
        cfov = (90.0, 85.0, 80.0, 70.0, 63.0, 52.0, 49.0, 45.0, 40.0, 35.0)

        if width == 0:
            width = 1024

        if height == 0:
            height = 1024

        #保证宽高为8的整数倍
        width = MUT8(width)
        height = MUT8(height)

        #全景投影fov使用默认扭曲参数
        
        fov += cfov[sides - 1]

        if fov <= 0.01:
            fov = 0.0
        
        #极轴扭曲depolar使用默认扭曲参数
        if depolar <= 0.01:
            depolar = 1.2

        pano_pipe = {
            "sides": sides,
            "scale": scale,
            "fov": fov,
            "angle": 0.0,
            "depolar": depolar,
            "width": width,
            "height": height,
            "scale_up": scale_up,
            "scale_down": scale_down,
            "blue": 0.0,
            "alpha": 0.0,
        }
        return (pano_pipe, sides)
def deal_image(image, width, height, scale):
    # 格式转换
    img = []
    
    if image is None:
        img = ImageContainer(width, height, 0, 0, 0, 0)
    else:
        img = image
        if image.shape[-1] == 3:
            img = add_alpha_channel(img)

        # 缩放比例
        if scale:
            img = ImageTransformResizeClip(img, width, height, 64, 64, method='lanczos')
    return img

class PanoImageHeightPad:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/pad"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_pipe": ("PANO_PIPE",),
            },
            "optional": {
                "image_pad": ("IMAGE",),
                "image_up": ("IMAGE",),
                "image_down": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("PANO_PIPE", "IMAGE", "MASK")
    RETURN_NAMES = ("pano_pipe", "equ", "mask")
    FUNCTION = "node"

    def node(self, 
             pano_pipe,
             image_pad=None,
             image_up=None,
             image_down=None
             ):
        
        depolar = pano_pipe.get('depolar')
        height = pano_pipe.get('height')
        width = pano_pipe.get('width')
        sides = pano_pipe.get('sides')
        fov = pano_pipe.get('fov')
        scale_up = pano_pipe.get('scale_up')
        scale_down = pano_pipe.get('scale_down')

        # 遮罩从中间填充指定高度的遮罩
        def modify_mask(mask, up_len, down_len):
            B, H, W = mask.shape
            half_height = H // 2
            new_mask = torch.zeros_like(mask)
            # 处理 up_len 为 0 的情况
            if up_len == 0:
                up_mask = mask[:, :half_height, :]
                new_mask[:, :half_height, :] = up_mask
            else:
                if up_len > 0:
                    up_mask = mask[:, up_len:half_height, :]
                    new_mask[:, :half_height-up_len, :] = up_mask
                    new_mask[:, half_height-up_len:half_height, :] = 1.0
                else:
                    up_mask = mask[:, :half_height+up_len, :] 
                    new_mask[:, -up_len:half_height, :] = up_mask

            # 处理 down_len 为 0 的情况
            if down_len == 0:
                down_mask = mask[:, half_height:, :]
                new_mask[:, half_height:, :] = down_mask
            else:
                if down_len > 0:
                    down_mask = mask[:, half_height:H-down_len, :]
                    new_mask[:, half_height+down_len:, :] = down_mask
                    new_mask[:, half_height:half_height+down_len, :] = 1.0
                else:
                    down_mask = mask[:, half_height-down_len:, :]
                    new_mask[:, half_height:H+down_len, :] = down_mask
            return new_mask
        
        #保证depolar极轴变换从多边形端点开始展开
        angle = 180/sides

        #根据输入的高度重新计算愣住体的高度，也是愣住体顶部外接正方形边长
        size = MUT8(width * 2)

        img_a = deal_image(image_up, size, size, True)
        img_b = deal_image(image_down, size, size, True)

        # 图像上下扩展
        img_c = []
        if image_pad is None:
            img_c = torch.zeros((1, size + size, size,  4), dtype=torch.float32)
            img_c[:, :size, :, :] = img_a
            img_c[:, size:, :, :] = img_b    
        else:
            # 图像上下扩展量
            B, h_pad, w_pad, C = image_pad.shape
            if C == 3:
                image_pad = add_alpha_channel(image_pad)

            # 扩展相同的高度
            add_height = (w_pad//2 - height) // 2 
            if add_height < 0:
                w_pad = height * 3
                add_height = height//2
                image_pad = overlay_images(image_pad, h_pad, w_pad)
                
            h_out = h_pad + add_height
            pad_out = h_out + add_height

            img_c = torch.zeros((B, pad_out, w_pad, 4), dtype=torch.float32)
            img_src = img_c.clone()

            # 生成mask
            add_len = MUT8(add_height * (1.5-fov/360))
            add_len_up = MUT8(add_len * scale_up)
            add_len_down = MUT8(add_len * scale_down)

            # 向上扩展图像
            img_a = depolar_transform(
                input_tensor=img_a,
                output_size=(w_pad, add_len_up),
                angle=angle,
                exponent = depolar,
            )
            img_src[:, :add_len_up, :, :] = img_a

            # 向下扩展图像
            img_b = depolar_transform(
                input_tensor=img_b,
                output_size=(w_pad, add_len_down),
                angle=angle,
                exponent = depolar,
            )
            img_b = ImageTransformTranspose(img_b, method='flip_vertically')
            img_src[:, pad_out-add_len_down:, :, :] = img_b 

            # 图像遮罩复合
            img_c[:, add_height:h_out, :, :] = image_pad

            # 遮罩合成
            mask = img_c[:, :, :, -1]
            mask = modify_mask(mask, add_len - add_len_up, add_len - add_len_down)
            mask = 1.0 - mask

            img_c = image_mask_composite(img_c, img_src, 0, 0, mask)
            mask = img_c[:, :, :, -1]

        return (pano_pipe, img_c, mask)

class PanoImageWidthPad:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/pad"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_pipe": ("PANO_PIPE",),
                "fit": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
            },
            "optional": {
                "image_pad": ("IMAGE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("PANO_PIPE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pano_pipe", "image_pad", "image")
    FUNCTION = "node"

    def node(self, 
             pano_pipe,
             fit,
             image_pad = None,
             image = None
             ):
        
        height = pano_pipe.get('height')
        width = pano_pipe.get('width')
        scale = pano_pipe.get('scale')
        fov = pano_pipe.get('fov')
        
        # 格式转换
        img_a = deal_image(image, width, height, False)
        
        # 单面透视变换
        if fov > 1.0:
            img_a = plane_to_cylinder(input_image=img_a, fov=fov,output_size=(height, width))
            #img_a = crop_transparent_area(img_a)

        # 图片叠加居中
        if fit:
            img_a = ImageTransformResizeAbsolute(img_a, width, height, method='lanczos')

        if scale < 0.99:
            img_a = ImageTransformResizeRelative(img_a, scale, scale, method='lanczos')

        img_b = overlay_images(img_a, height, width, align='center')

        # 声明
        img_c = []

        # 图像右扩
        if image_pad is not None:     
            # 图像向右扩展量
            B, h_pad, w_pad, C = image_pad.shape
            if C == 3:
                image_pad = add_alpha_channel(image_pad)

            if h_pad != height:
                image_pad = ImageTransformResizeClip(image_pad, w_pad, height, 64, 64, method='lanczos')
                
            # 设置输出图像的尺寸
            B, h_pad, w_pad, C = image_pad.shape
            w_out = w_pad + width
            img_c = torch.zeros((B, h_pad, w_out, 4), dtype=torch.float32)
       
            # 图像右扩
            img_c[:, :, :w_pad, :] = image_pad
            img_c[:, :, w_pad:, :] = img_b

        else:
            img_c = img_b

        return (pano_pipe, img_c, img_b)


class PanoImagePad:
    def __init__(self):
        pass
    
    CATEGORY = "PanoCard/pad"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "add_width": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "add_height": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "ratio_width": ("FLOAT", {
                    "default": 0.5,
                    "max": 1.0,
                    "min": 0,
                    "step": 0.01
                }),
                "ratio_height": ("FLOAT", {
                    "default": 0.5,
                    "max": 1.0,
                    "min": 0,
                    "step": 0.01
                }),
                "method": (["edge", "reflect", "constant", "symmetric"],),
                "resize": (["Default", "Panorama_Ratio", "Original_Ratio","None"],),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ("image","mask",)
    FUNCTION = "node"

    def node(self, images, add_width, add_height, ratio_width, ratio_height, method, resize):
        def create_mask(image, left, top, right, bottom):
            B, H, W, C = image.shape
            mask = torch.ones((B, H + top + bottom, W + left + right, 1), dtype=torch.float32)
            mask[:, top:H+top, left:W+left, :] = 0
            return mask
        
        def adjust_dimensions(H, W):
            max_dimension = max(2 * H, W)
            if max_dimension == W:
                H = MUT8(W // 2)
            else:
                H = MUT8(H)
            W = 2 * H
            return H, W
        
        Original_H = images.shape[1]
        Original_W = images.shape[2]

        if add_width == 0 and add_height == 0:
            B, H, W, C = images.shape  
            if resize == "Panorama_Ratio": 
                H1, W1 = adjust_dimensions(H + H//4, W + W//4)
            else:
                H1, W1 = adjust_dimensions(H, W)
            add_width = W1 - W
            add_height = H1 - H
            print(f"{H}x{W} -> {H1}x{W1}")

        right = int(add_width * ratio_width)
        left = add_width - right
        top = int(add_height * ratio_height)
        bottom = add_height - top

        image = torch.stack([transpose_tensor(images[i], left, top, right, bottom, method) for i in range(len(images))])

        # 创建遮罩
        mask = create_mask(images[0].unsqueeze(0), left, top, right, bottom)  # 确保输入的图像形状为 (B, H, W, C)

        if resize != "None":
            B, H, W, C = image.shape
            if resize == "Panorama_Ratio":
                H = MUT8(H)
                W = MUT8(W)
                if add_width == 0 and add_height:
                    H, W = W//2, W
                elif add_height == 0 and add_width:
                    H, W = H, H + H  
                else:
                    H, W = adjust_dimensions(H, W)
            elif resize == "Original_Ratio":
                H, W = Original_H, Original_W
            
            elif resize == "Default":
                H = MUT8(H)
                W = MUT8(W)

            # 调整图像大小
            image = resize_tensor(image, H, W)

            # 调整遮罩大小
            mask = resize_tensor(mask, H, W)

        mask = mask.squeeze(-1) 
        return (image, mask)


class PanoImageRoll:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/adjust"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "tiles": ("INT", {"default": 2, "min": 2}),
            "offset": ("INT", {"default": 1, "min": 1}),
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE","MASK")
    FUNCTION = "node"

    def node(self, image, tiles, offset, mask = None):
        if tiles <= offset:
            offset = offset % tiles

        if offset == 0:
            return (image, mask)

        width = image.shape[2]
        tile_width = width // tiles

        if mask is None:
            if image.shape[3] == 4:
                mask = image[:, :, :, 3]
            else:
                mask = image[:, :, :, 2]
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        output_image = torch.zeros_like(image)
        output_mask = torch.zeros_like(mask)  # 创建一个与 mask 形状相同的输出 mask

        start_w = offset * tile_width
        tile_image = image[:, :, start_w:, :]
        tile_mask = mask[:, :, start_w:]  # 提取 mask 的相应部分

        output_image[:, :, :width - start_w, :] = tile_image
        output_mask[:, :, :width - start_w] = tile_mask  # 将 tile_mask 放置在输出 mask 的相应位置

        tile_image = image[:, :, :start_w, :]
        tile_mask = mask[:, :, :start_w]  # 提取 mask 的相应部分

        output_image[:, :, width - start_w:, :] = tile_image
        output_mask[:, :, width - start_w:] = tile_mask  # 将 tile_mask 放置在输出 mask 的相应位置

        return (output_image, output_mask)


class PanoImageOutClamp:
    def ImageHeightAdjust(self, images, ratio_front, ratio_right, ratio_back, ratio_left, ratio_up, ratio_down):
        # [b,h,w,c]
        print("images",images.shape)

        num_images = len(images)

        images_list = []
        for i in range(6):
            if i < num_images:
                image = images[i]
                if image.shape[2] == 3:
                    # [H, W, C] 添加 alpha 通道
                    H, W, _ = image.shape
                    alpha_channel = torch.ones((H, W, 1), dtype=image.dtype, device=image.device)
                    image = torch.cat((image, alpha_channel), dim=2)
                images_list.append(image)
            else:
                # 如果图像数量不足6张，添加默认图像
                images_list.append(torch.zeros((64, 64, 4)))

        pad_ratio = [ratio_front, ratio_right, ratio_back, ratio_left, ratio_up, ratio_down]

        for i in range(6):
            if pad_ratio[i] > 0.01 or pad_ratio[i] < -0.01:
                H, W, _ = images_list[i].shape
                if pad_ratio[i] > 0.01:
                    top = int(H * pad_ratio[i])
                    images_list[i] = transpose_tensor(images_list[i], 0, top, 0, 0, "constant").unsqueeze(0)
                else:
                    bottom = -int(H * pad_ratio[i])
                    images_list[i] = transpose_tensor(images_list[i], 0, 0, 0, bottom, "constant").unsqueeze(0)
                images_list[i] = resize_tensor(images_list[i], H, W)
            else:
                images_list[i] = images_list[i].unsqueeze(0)

        # 返回结果
        return (images_list[0],images_list[1],images_list[2],images_list[3],images_list[4],images_list[5])


class PanoImageAdjust:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/adjust"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (["none", "alpha_remove", 
                            "scale", "scale_height", "scale_width", 
                            "pad_height","pad_width","pad_edge",
                            "stretch", "stretch_pano","stretch_arc",
                            "crop_edge", "crop_width", "crop_height", "crop_up_down", "crop_left_right"],),
                "keep_size": ("BOOLEAN", {"default": False,"label_on": "ture", "label_off": "false"}),
                "ratio_front": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),     
                "ratio_right": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ratio_back": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ratio_left": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ratio_up": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ratio_down": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "front": ("IMAGE",), 
            },
            "optional": {    
                "front_mask": ("MASK",),    
                "right": ("IMAGE",),
                "right_mask": ("MASK",), 
                "back": ("IMAGE",),
                "back_mask": ("MASK",), 
                "left": ("IMAGE",),
                "left_mask": ("MASK",), 
                "up": ("IMAGE",),  
                "up_mask": ("MASK",),           
                "down": ("IMAGE",),
                "down_mask": ("MASK",), 
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("front","right","back","left","up","down")
    FUNCTION = "node"
    def node(self, method, keep_size,
             ratio_front, ratio_right, ratio_back, ratio_left, ratio_up, ratio_down,
             front, front_mask=None,
             right=None, right_mask=None,
             back=None, back_mask=None,
             left=None, left_mask=None,
             up=None, up_mask=None,
             down=None, down_mask=None
             ):

        # 将输入的图像转换为列表并克隆
        images = [front.clone() if front is not None else None,
                  right.clone() if right is not None else None,
                  back.clone() if back is not None else None,
                  left.clone() if left is not None else None,
                  up.clone() if up is not None else None,
                  down.clone() if down is not None else None]
        
        masks = [front_mask, right_mask, back_mask, left_mask, up_mask, down_mask]
        pad_ratio = [ratio_front, ratio_right, ratio_back, ratio_left, ratio_up, ratio_down]

        # 缩放比例
        for i in range(6):   
            if images[i] is None:
                continue
            
            ratio = pad_ratio[i]
            _, H, W, C = images[i].shape

            if masks[i] is not None:
                images[i] = join_image_with_alpha(images[i], masks[i])
            elif C == 3:
                images[i] = add_alpha_channel(images[i])

            if method == "alpha_remove":
                rgb = images[i][:, :, :, :3]
                if keep_size == False:
                    alpha = images[i][:, :, :, 3:]
                    mask = (alpha == 0).expand_as(rgb)
                    rgb[mask] = ratio
                images[i] = rgb
                print("alpha_remove", images[i].shape)
                continue

            if ratio < 0.01 and ratio > -0.01:
                continue

            image = images[i]

            if method == "pad_height" or method == "pad_width" or method == "pad_edge":
                # 扩展
                edge_w = int(W * ratio)   
                edge_h = int(H * ratio)        
                if ratio < 0:
                    edge_w = -edge_w
                    edge_h = -edge_h   

                if method == "pad_height":
                    if ratio > 0:
                        images[i] = transpose_tensor(image[0], 0, edge_h, 0, 0, "constant").unsqueeze(0)
                    else:
                        images[i] = transpose_tensor(image[0], 0, 0, 0, edge_h, "constant").unsqueeze(0)
                elif method == "pad_width":
                    if ratio > 0:
                        images[i] = transpose_tensor(image[0], edge_w, 0, 0, 0, "constant").unsqueeze(0)
                    else:
                        images[i] = transpose_tensor(image[0], 0, 0, edge_w, 0, "constant").unsqueeze(0)
                elif method == "pad_edge":    
                    images[i] = transpose_tensor(image[0], edge_w, edge_h, edge_w, edge_h, "constant").unsqueeze(0)

                if keep_size:
                    images[i] = resize_tensor(images[i], H, W)

            elif method == "stretch":
                # 拉伸
                new_ratio = ratio
                if ratio < 0:
                    new_ratio = 1.0 + ratio * 0.5
                else:
                    new_ratio = 1.0 + ratio
                stretch_ratio = new_ratio * H / W

                width = MUT8(math.sqrt(H*W / stretch_ratio))
                height = MUT8(width * stretch_ratio)

                if keep_size:
                    if width > W:
                        width = W
                        height = int(width * stretch_ratio)
                    else:
                        height = H
                        width = int(height / stretch_ratio)
                    images[i] = ImageTransformResizeAbsolute(image, width, height, "lanczos") 
                    images[i] = overlay_images(images[i], H, W, align='center')
                else:
                    images[i] = ImageTransformResizeAbsolute(image, width, height, "lanczos") 
            elif method == "stretch_pano" or method == "stretch_arc":
                # 拉伸
                if method == "stretch_pano":
                    fov = 179 * ratio
                    images[i] = plane_to_cylinder(input_image=image, fov=fov)
                else:
                    images[i] = arc_distortion(image, ratio)

                if keep_size:
                    images[i] = resize_tensor(images[i], H, W)

            elif method == "scale_height" or method == "scale_width" or method == "scale":
                # 缩放
                new_ratio = ratio
                if ratio < 0:
                    new_ratio = -ratio

                scale_width = 1.0
                scale_height = 1.0
                if method == "scale_height":
                    scale_height = new_ratio
                elif method == "scale_width":
                    scale_width = new_ratio
                else:
                    scale_width = new_ratio
                    scale_height = new_ratio

                images[i] = ImageTransformResizeRelative(image, scale_width, scale_height, "lanczos") 
                if keep_size:
                    images[i] = overlay_images(images[i], H, W, align='center')

            elif method == "crop_edge" or method == "crop_width" or method == "crop_height" or method == "crop_left_right"or method == "crop_up_down":
                # 裁剪 
                crop_ratio = ratio
                if ratio < 0:
                    crop_ratio = -ratio
                # 计算需要清零的边缘宽度和高度
                edge_w = int(W * crop_ratio / 2)
                edge_h = int(H * crop_ratio / 2)
                
                if edge_w > W//2:
                    edge_w = W//2 - 1
                if edge_h > H//2:
                    edge_h = H//2 - 1

                # 处理边缘部分
                if method == "crop_edge":
                    if keep_size:
                        image[:, :edge_h, :, -1] = 0  # 上边缘
                        image[:, -edge_h:, :, -1] = 0  # 下边缘
                        image[:, :, :edge_w, -1] = 0  # 左边缘
                        image[:, :, -edge_w:, -1] = 0  # 右边缘
                    else:
                        images[i] = image[:, edge_h:H-edge_h, edge_w:W-edge_w, :]
                elif method == "crop_width":
                    if keep_size:
                        image[:, :, :edge_w, -1] = 0  # 左边缘
                        image[:, :, -edge_w:, -1] = 0  # 右边缘
                    else:
                        images[i] = image[:, :, edge_w:W-edge_w, :]
                elif method == "crop_height":
                    if keep_size:
                        image[:, :edge_h, :, -1] = 0  # 上边缘
                        image[:, -edge_h:, :, -1] = 0  # 下边缘
                    else:
                        images[i] = image[:, edge_h:H-edge_h, :, :]
                elif method == "crop_left_right":
                    edge_w += edge_w
                    if keep_size:
                        if ratio > 0:
                            image[:, :, -edge_w:, -1] = 0  # 右边缘
                        else:
                            image[:, :, :edge_w, -1] = 0  # 左边缘
                    else:
                        if ratio > 0:
                            images[i] = image[:, :, :W-edge_w, :]
                        else:
                            images[i] = image[:, :, edge_w:, :]
                elif method == "crop_up_down":
                    edge_h += edge_h
                    if keep_size:
                        if ratio > 0:
                            image[:, :edge_h, :, -1] = 0  # 上边缘
                        else:
                            image[:, -edge_h:, :, -1] = 0  # 下边缘
                    else:
                        if ratio > 0:
                            images[i] = image[:, :H-edge_h, :, :]
                        else:
                            images[i] = image[:, edge_h:, :, :]

        # 返回结果
        return (images[0],images[1],images[2],images[3],images[4],images[5])


class PanoImageClamp:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_scale": ("FLOAT", {
                    "default": 1.0,
                    "max": 5.0,
                    "min": 1.0,
                    "step": 0.01
                }),
                "method": (["none", "fit", "full"],),   
            },
            "optional": {  
                "front": ("IMAGE",),       
                "right": ("IMAGE",),
                "back": ("IMAGE",),
                "left": ("IMAGE",),
                "up": ("IMAGE",),            
                "down": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "node"
    def node(self, canvas_scale, method, front=None, right=None, back=None, left=None, up=None, down=None):
        # 将输入的图像转换为 PyTorch 的 Tensor
        images = [front, right, back, left, up, down]

        # 使用列表推导式获取所有高度和宽度
        heights = [64] * 6
        widths = [64] * 6

        for i in range(6):
            if images[i] is not None:
                heights[i] = images[i].shape[1]
                widths[i] = images[i].shape[2]

                if images[i].shape[-1] == 3:
                    images[i] = add_alpha_channel(images[i])

        # 找到最大高度和宽度
        max_height = max(heights)
        max_width = max(widths)

        size = max(max_height, max_width)
        size = MUT8(size)

        # 缩放比例
        for i in range(6):  
            if method == "none":
                break
            if images[i] is None:
                continue
            if method == "fit":           
                width = widths[i]
                height = heights[i]
                ratio = width / height
                # 根据 size 调整宽度和高度
                if width > height:
                    width = size
                    height = int(width / ratio)
                else:
                    height = size
                    width = int(height * ratio) 
                images[i] = ImageTransformResizeAbsolute(images[i], width, height, "lanczos")

            elif method == "full":
                images[i] = ImageTransformResizeAbsolute(images[i], size, size, "lanczos")
                
        if canvas_scale > 1.0:
            size = MUT8(size * canvas_scale)
        stacked_images = overlay_images(images, size, size, align='center').unsqueeze(0)

        # 返回结果
        return (stacked_images,)


class PanoFaceToLong:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face": ("IMAGE",),            
            },
        }

    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ("long","long_mask")
    FUNCTION = "node"
    def node(self, face):
        B, H, W, C = face.shape
        if W != H:
            raise ValueError("输入的宽高必须相等")
        
        if B != 6:
            raise ValueError("输入的图像必须为包含6张")

        if C == 3:
            face = add_alpha_channel(face)

        long_image = torch.zeros(1, H, H*6, 4, dtype=face.dtype, device=face.device)

        for i in range(6):
            long_image[0, :, i*H:(i+1)*H, :] = face[i]

        mask = long_image[:, :, :, -1]
        
        # 返回结果
        return (long_image, mask)

class PanoMaskOutClamp:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),            
            },
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK")
    RETURN_NAMES = ("front","right","back","left","up","down")
    FUNCTION = "node"
    def node(self, masks):

        print("masks",masks.shape)
        assert len(masks) == 6
        masks = [mask.unsqueeze(0) if len(mask.shape) < 3 else mask for mask in masks]
        
        # 返回结果
        return (masks[0],masks[1],masks[2],masks[3],masks[4],masks[5])


class PanoMaskOutFaceClamp:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {   
                "long_mask": ("MASK",),  
                "ratio": ("FLOAT", {"default": 1.0,
                            "max": 1.0,
                            "min": 0.1,
                            "step": 0.01
                        }),
                "invert": ("BOOLEAN", {"default": False, "label_on": "true", "label_off": "false"}),     
            },
        }

    RETURN_TYPES = ("MASK","MASK")
    RETURN_NAMES = ("mask","masks")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "node"
    def node(self, long_mask, ratio, invert):

        B, H, W = long_mask.shape
        if W != H * 6:
            raise ValueError("Mask shape error.")
        
        S = int(H * ratio)
        T = (H - S) // 2

        mask = torch.zeros_like(long_mask)
        face_mask = [mask.clone() for _ in range(6)]
        for i in range(6):
            temp = long_mask[:, :, i*H:(i+1)*H]
            temp2 = resize_mask(temp, (S, S))
            temp = torch.zeros_like(temp)
            temp[:, T:T+S, T:T+S] = temp2
            if invert:
                temp = 1 - temp
            mask[:, :, i*H:i*H+H] = temp
            face_mask[i][:, :, i*H:i*H+H] = temp

        face_mask = torch.stack(face_mask, dim=0).squeeze(1).unsqueeze(0)

        # 返回结果
        return (mask, face_mask)

class PanoImageOutClamp:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),   
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("front","right","back","left","up","down")
    FUNCTION = "node"
    def node(self, images):
        num_images = len(images)
        images_list = []
        for i in range(6):
            if i < num_images:
                images_list.append(images[i].unsqueeze(0))
            else:
                # 如果图像数量不足6张，添加默认图像
                images_list.append(torch.zeros((1, 64, 64, 3)))

        # 返回结果
        return (images_list[0],images_list[1],images_list[2],images_list[3],images_list[4],images_list[5])




class PanoMaskCondBatch:
    CATEGORY = "PanoCard/conditioning"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ), 
                "strength_mask": ("FLOAT",{"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},),
                "combine_up_down": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "total": ("STRING", {"multiline": True}),
                "prefix": ("STRING", {"multiline": True}),
                "front": ("STRING", {"multiline": True}),
                "right": ("STRING", {"multiline": True}),
                "back": ("STRING", {"multiline": True}),
                "left": ("STRING", {"multiline": True}),
                "up": ("STRING", {"multiline": True}),
                "down": ("STRING", {"multiline": True}),
            },
            "optional": {
                "masks": ("MASK",),
                "cond_face": ("CONDITIONING",),
            }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","MASK")
    RETURN_NAMES = ("cond_mix","cond_face","cond_total","cond_mask","masks")
    FUNCTION = "encode"

    def encode(self, 
               clip, 
               strength_mask,
               combine_up_down,
               total, 
               prefix, 
               front, right, back, left, up, down,
               masks = None, cond_face = None
               ):

        #调整局部和整体的强度
        strength_mask  = strength_mask
        strength_total = 1.0 - strength_mask

        #全局编码
        tokens = clip.tokenize(total + prefix)
        t_cond, t_pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning_total = [(t_cond, {"pooled_output": t_pooled})]
        num_tokens = []
        
        if cond_face is None:
            #局部编码
            conds = []
            pooleds = []
            if combine_up_down:
                texts = [front,right,back,left,total,up+down]
            else:
                texts = [front,right,back,left,up,down]
            for i, text in enumerate(texts):
                tokens = clip.tokenize(prefix + text)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                conds.append(cond)
                pooleds.append(pooled)
                num_tokens.append(cond.shape[1])
        else:
            # 解包外层列表
            [b_conds, face_pooled_dict] = cond_face[0]
            # 从字典中解包 b_pooleds
            b_pooleds = face_pooled_dict["pooled_output"]
                   
            conds = get_original_tensors(b_conds)
            if combine_up_down:
                conds[5] = torch.cat((conds[4], conds[5]), 1)
                conds[4] = t_cond
                pooleds[4] = t_pooled
            pooleds = get_original_tensors(b_pooleds)
            num_tokens = [cond.shape[1] for cond in conds]

        # 找到最大的 token 数量
        num_tokens2 = num_tokens + [t_cond.shape[1]]
        max_num_tokens = max(num_tokens2)

        # 填充face token
        conds = pad_conditions(conds, max_num_tokens)
        b_conds = torch.cat(conds)
        b_pooleds = torch.cat(pooleds)
        conditioning_face = [(b_conds, {"pooled_output": b_pooleds})]

        # 填充totol token
        t_cond = pad_conditions([t_cond], max_num_tokens)[0]
   
        # 混合条件
        conditioning_mix =  conditioning_total
        for i in range(6):
            face_conds = [(conds[i], {"pooled_output": pooleds[i]})]
            conditioning_mix = nodes.ConditioningConcat().concat(conditioning_mix, face_conds)[0]

        # 计算遮罩合并
        conditioning_mask = []
        if masks is not None: 
            if masks.shape[0] != 6:
                raise ValueError("The number of masks must be 6.")

            mask_a = torch.ones_like(masks[0]).unsqueeze(0)
            print(mask_a.shape)
            
            t_conditioning = [(t_cond, {
                                        "pooled_output": t_pooled,
                                        "mask": mask_a ,
                                        "set_area_to_bounds": False,
                                        "mask_strength": strength_total
                                        })]
            conditioning_mask = t_conditioning

            if combine_up_down:
                masks[5] = masks[4] + masks[5]
                masks[4] = mask_a - (masks[0] + masks[1] + masks[2] + masks[3] + masks[5])

            for i in range(6):
                mask_i = masks[i]
                if len(mask_i.shape) < 3:
                    mask_i = mask_i.unsqueeze(0)
                
                b_conditioning = [(conds[i], {
                                            "pooled_output": pooleds[i],
                                            "mask": mask_i,
                                            "set_area_to_bounds": False,
                                            "mask_strength": strength_mask
                                            })]
                conditioning_mask += b_conditioning
        else:
            conditioning_mask = conditioning_total
            for i in range(6):
                face_conds = [(conds[i], {"pooled_output": pooleds[i]})]
                conditioning_mask = nodes.ConditioningCombine().combine(conditioning_mask, face_conds)[0]
 
        return (conditioning_mix,
                conditioning_face,
                conditioning_total, 
                conditioning_mask, 
                masks)
    

class PanoClipBatch:
    CATEGORY = "PanoCard/conditioning"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ), 
                "prefix": ("STRING", {"multiline": True}),
                "front": ("STRING", {"multiline": True}),
                "right": ("STRING", {"multiline": True}),
                "back": ("STRING", {"multiline": True}),
                "left": ("STRING", {"multiline": True}),
                "up": ("STRING", {"multiline": True}),
                "down": ("STRING", {"multiline": True}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond_face",)
    FUNCTION = "encode"

    def encode(self, clip, prefix, front, right, back, left, up, down):  
        conds = []
        pooleds = []
        front = prefix + front
        right = prefix + right
        back = prefix + back
        left = prefix + left
        up = prefix + up
        down = prefix + down

        texts = [front,right,back,left,up,down]
        for text in texts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conds.append(cond)
            pooleds.append(pooled)
        
        conds = pad_conditions(conds)
        b_conds = torch.cat(conds)
        b_pooleds = torch.cat(pooleds)

        return ([(b_conds, {"pooled_output": b_pooleds})],)


class PanoCondClipClamp:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("CONDITIONING",),        
                "right": ("CONDITIONING",),
                "back": ("CONDITIONING",),
                "left": ("CONDITIONING",),
                "up": ("CONDITIONING",),            
                "down": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond_face",)
    FUNCTION = "node"
    def node(self, front, right, back, left, up, down, guidance):
        # 将输入的图像转换为 PyTorch 的 Tensor
        cond_list = [front, right, back, left, up, down]
        cnet_list = []
        conds = []
        pooleds = []
        for cond_face in cond_list:
            if cond_face is not None: 
                [c, t] = cond_face[0]
                conds.append(c)
                pooleds.append(t['pooled_output'])
                if 'control' in t and 'control_apply_to_uncond' in t:
                    cnet_dict = {
                        'control': t['control'],
                        'control_apply_to_uncond': t['control_apply_to_uncond']
                    }
                    cnet_list.append(cnet_dict)
                else:
                    cnet_list.append({})
            else:
                raise Exception("Error: The input is not a valid condition.")
                
        cond_face = pad_conditions(conds)
        b_conds = torch.cat(cond_face)
        b_pooleds = torch.cat(pooleds)

        if len(cnet_list) > 0:
            res_dict = {"pooled_output": b_pooleds, "pano_control": cnet_list}
        else:
            res_dict = {"pooled_output": b_pooleds}
        
        if guidance > 0:
            res_dict["guidance_scale"] = guidance
        
        res = [(b_conds, res_dict)]

        # 返回结果
        return (res,)



class PanoPromptSplit:
    CATEGORY = "PanoCard/conditioning"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
            },
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("total","prefix", "front", "right", "back", "left", "up", "down")
    FUNCTION = "encode"

    def encode(self, prompt):
        def keep_only_ascii(input_string):
            # 使用正则表达式匹配非ASCII字符并替换为空字符串
            ascii_string = re.sub(r'[^\x00-\x7F]+', '', input_string)
            return ascii_string
        def extract_strings(input_string):
            # 正则表达式匹配以 [XXXX]: 开头的字符串，并提取冒号后面的内容
            pattern = r'\[(.*?)\](.*?)(?=\[|$)'
            
            # 创建一个列表，用于存储匹配到的字符串
            result_list = [""] * 8

            # 定义一个元组列表，用于存储匹配到的字符串,模糊匹配
            keys_list = [("prefix", "Prefix", "0"),
                        ("front", "Front", "1"),
                        ("right", "Right", "2"),
                        ("back", "Back", "3"),
                        ("left", "Left", "4"),
                        ("up", "Up", "5"),
                        ("down", "Down", "6"),
                        ("total", "Total", "7")]

            # 第一次尝试匹配整个字符串
            matches = re.findall(pattern, input_string, re.DOTALL)
            list_match = [(match[0], match[1].strip()) for match in matches]

            lenth = len(matches)
            if lenth > 4:
                # 初始化已匹配的索引集合
                matched_indices = set()

                # 遍历 keys_list 和 list_match 进行匹配
                for i, keys in enumerate(keys_list):
                    for key in keys:
                        for j, match in enumerate(list_match):
                            if j in matched_indices:
                                continue  # 跳过已经匹配过的项
                            if key in match[0]:
                                result_list[i] = match[1]
                                matched_indices.add(j)  # 标记当前 match 已匹配
                                break  # 找到匹配后跳出内层循环
            else:
                # 如果第一次匹配未找到六项，按换行符分割字符串并逐行匹配
                lines = input_string.splitlines()
                no_empty_lines = [line for line in lines if line.strip()]
                lenth = len(no_empty_lines)
                j = 0

                for i in range(lenth):
                    line_length = len(no_empty_lines[i])
                    if line_length < 20:
                        continue
                    if j >= 8:
                        break
                    result_list[j] = no_empty_lines[i]
                    j += 1
                
            print("result_list_lenth", lenth)
            # 再次检查是否找到了六项匹配
            return result_list
        
        result_ascii = keep_only_ascii(prompt)

        result_list = extract_strings(result_ascii)

        return (result_list[7],
                result_list[0], 
                result_list[1], 
                result_list[2], 
                result_list[3], 
                result_list[4], 
                result_list[5], 
                result_list[6])
    
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', "GITS[coeff=1.2]"]
class PanoRegionalPrompt:
    CATEGORY = "PanoCard/conditioning"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "cond_face": ("CONDITIONING",),
                "masks": ("MASK",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (SCHEDULERS,),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_method": (["linear", "slerp"],),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            }
        }

    RETURN_TYPES = ("KSAMPLER_ADVANCED","REGIONAL_PROMPTS")
    RETURN_NAMES = ("sampler", "prompts")
    FUNCTION = "doit"

    @staticmethod
    def doit(basic_pipe, cond_face, masks, cfg, sampler_name, scheduler, 
             sigma_factor=1.0, variation_seed=0, variation_strength=0.0, variation_method="linear", scheduler_func_opt=None):
        
        if masks.shape[0] != 6:
            raise Exception("Mask shape must be 6")
        
        bkap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        base_sampler = bkap.doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=sigma_factor, scheduler_func_opt=scheduler_func_opt)[0]

        res = []
        model, clip, vae, positive, negative = basic_pipe
        def rdoit(basic_pipe, mask, cfg, sampler_name, scheduler, 
                sigma_factor=1.0, variation_seed=0, variation_strength=0.0, variation_method='linear', scheduler_func_opt=None):
            if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
                raise Exception(f"[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")
            
            kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
            rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()
            sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=sigma_factor, scheduler_func_opt=scheduler_func_opt)[0]
            try:
                regional_prompts = rp.doit(mask, sampler, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method)[0]
            except:
                raise Exception("[Inspire-Pack] ERROR: Impact Pack is outdated. Update Impact Pack to latest version to use this.")

            return regional_prompts
        
        conds = DepackClip(cond_face)

        for i, mask in enumerate(masks):
            positive = conds[i]
            pipe = model, clip, vae, positive, negative
            rp = rdoit(pipe, mask.unsqueeze(0), cfg, sampler_name, scheduler, 
                    sigma_factor=sigma_factor, 
                    variation_seed=variation_seed, 
                    variation_strength=variation_strength, 
                    variation_method=variation_method, 
                    scheduler_func_opt=scheduler_func_opt)
            res += rp
            
        return (base_sampler, res)
    


class PanoImageSplit:
    CATEGORY = "PanoCard/split"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "ratio": ("FLOAT", {"default": 0.25, "min": 0, "max": 1.0, "step": 0.01}),
                "fov": ("FLOAT", {"default": 0.0, "min": -179.0, "max": 179.0, "step": 0.1}),
                "resize": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "encode"

    def encode(self, image, ratio, fov, resize):

        # 获取原图尺寸
        b, h, w, c = image.shape
        if c == 3:
            image = add_alpha_channel(image)
            c = 4
            
        # 确保宽度是8的倍数
        if w % 8 != 0 or h % 8 != 0:
            new_w = MUT8(w)
            new_h = MUT8(h)
            # 创建新的张量，形状为 (batch_size, new_h, new_w, channels)
            image_new = torch.zeros((b, new_h, new_w, c), dtype=image.dtype, device=image.device)
            image_new[:, :h, :w, :] = image
            image = image_new

        b, h, w, c = image.shape
        ratio = 1.0 + ratio * 4
        
        # 计算 w1 和 w2
        w1 = w // 4
        w2 = w

        # 计算总面积
        total_area = h * w
        # 计算 k
        area_per = total_area / (2 * ratio + 4)
        # 计算 area_1 和 area_2
        area_up = area_per * ratio
        # 计算 h1 和 h2
        h1 = int(area_per / w1)
        h2 = int(area_up / w2)
        wo = w1//2

        # 图像偏移
        image = torch.roll(image, shifts=-wo, dims=2)
        if fov > 0.1 or fov < -0.1:
            image_new = torch.zeros((b, h, w, c), dtype=image.dtype, device=image.device)
            for i in range(4):
                img = image[:, :, i*w1:(i+1)*w1, :]
                img = plane_to_cylinder(input_image=img, fov=fov)
                img = resize_tensor(img, h, w1)
                image_new[:, :, i*w1:(i+1)*w1, :] = img
            image = image_new

        # 分割图像
        images = []
        for i in range(4):
            start_w = i * w1
            end_w = start_w + w1
            img = image[:, h2:h1+h2, start_w:end_w, :]
            images.append(img)
        
        # 交换顺序
        images = images[1:] + images[:1]   

        def rotate_segment(segment, direction):
            if direction == '270':
                return torch.rot90(segment, k=1, dims=[1, 2])  # 逆时针旋转90°
            elif direction == '90':
                return torch.rot90(segment, k=-1, dims=[1, 2])  # 顺时针旋转90°
            elif direction == '180':
                return torch.rot90(segment, k=2, dims=[1, 2])  # 旋转180°
            else:
                raise ValueError("Invalid rotation direction. Use 'ccw', 'cw', 'hflip', '180', or '270'.")
        def split_and_rearrange_images(images, direction):
            B, H, W, C = images.shape
            W_new = W // 4
            # Split the image into 4 segments
            segment1 = images[:, :, :W_new, :]
            segment2 = images[:, :, W_new:2*W_new, :]
            segment3 = images[:, :, 2*W_new:3*W_new, :]
            segment4 = images[:, :, 3*W_new:, :]

            if direction == 'up':
                segment1 = rotate_segment(segment1, '90')
                segment3 = rotate_segment(segment3, '270')
                segment4 = rotate_segment(segment4, '180')
            elif direction == 'down':
                segment1 = rotate_segment(segment1, '270')
                segment3 = rotate_segment(segment3, '90')
                segment4 = rotate_segment(segment4, '180')
                segment2, segment4 = segment4, segment2

            # Rearrange the segments into a new square image
            new_image = torch.zeros((B, 2*H + W_new, 2*H + W_new, C), dtype=images.dtype, device=images.device)

            HW = H+W_new
            # Place the segments in the new image
            new_image[:, H:HW, :H, :] = segment1
            new_image[:, HW:, H:HW, :] = segment2
            new_image[:, H:HW, HW:, :] = segment3
            new_image[:, :H, H:HW, :] = segment4
        
            return new_image
        
        image5 = image[:, :h2, :w2, :]
        image5 = split_and_rearrange_images(image5, 'up')
        images.append(ImageTransformResizeAbsolute(image5, w1, h1, "lanczos"))

        image6 = image[:, h1 + h2:h, :w2, :]
        image6 = split_and_rearrange_images(image6, 'down')
        images.append(ImageTransformResizeAbsolute(image6, w1, h1, "lanczos"))

        # 统一尺寸      
        if resize:
            height = MUT8(math.sqrt(w1*h1))
            for i in range(6):
                images[i] = ImageTransformResizeAbsolute(images[i], height, height, "lanczos")

        images = torch.stack(images,dim=1)

        return (images,)

class PanoClipOutClamp:
    CATEGORY = "PanoCard/conditioning"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_face": ("CONDITIONING", ), 
            },
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("front","right","back","left","up","down")
    FUNCTION = "encode"

    def encode(self, cond_face):

        res = DepackClip(cond_face)

        return (res[0],
                res[1],
                res[2],
                res[3],
                res[4],
                res[5])


class FaceCondScheduleHook(DetailerHook):
    def __init__(self, cond, face_niose, seed):
        super().__init__()
        self.cond = cond
        self.seed = seed
        self.face_niose = face_niose
    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        index = abs(seed - self.seed) % 6
        if index == 0 and self.cond is None:
            self.cond = DepackClip(positive)

        positive = self.cond[index]  
        if self.face_niose[index] < 0.1:
            steps = 0
        else:
            denoise = self.face_niose[index]
        print("index:", index, " denoise:",denoise)
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise
    
class CondFaceScheduleHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "niose_face1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "niose_face2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "niose_face3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "niose_face4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "niose_face5": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "niose_face6": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                    "optional": {
                        "cond_face": ("CONDITIONING",),
                    }
                }
    RETURN_TYPES = ("DETAILER_HOOK","INT")
    RETURN_NAMES = ("DetailerHook","seed")
    FUNCTION = "doit"

    CATEGORY = "PanoCard/conditioning"

    def doit(self, seed, niose_face1, niose_face2, niose_face3, niose_face4, niose_face5, niose_face6, cond_face=None ):
        face_niose = [niose_face1, niose_face2, niose_face3, niose_face4, niose_face5, niose_face6]
        if cond_face is None:
            cond = None
        else:
            cond = DepackClip(cond_face)
        hook = FaceCondScheduleHook(cond, face_niose, seed)
        return (hook, seed)