import numpy as np
import equilib as py360
import torch
from .distort import plane_to_equirect
from .misc import *
import math
import nodes

def calculate_equ_heght(heght):
    # 计算 sqrt(x * x * 3)
    result = math.sqrt(heght * heght * 3)
    
    # 将结果四舍五入到最接近的64的倍数
    result = int((result / 64 + 0.5)) * 64
    
    if result < 512:
        result = 512
        
    return result

def calculate_face_heght(heght):
    # 计算 sqrt(x * x / 3)
    result = math.sqrt(heght * heght / 3)
    
    # 将结果四舍五入到最接近的64的倍数
    result = int((result / 64 + 0.5)) * 64

    if result < 512:
        result = 512
    
    return result

# Equirectangular全景图，Cubemap 立方体贴图 Perspective 透视图
class PanoImageCube2Equ:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/convert"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {  
                "height": ("INT", {
                    "default": 832,
                    "min": 0,
                    "step": 8
                }),
                "scale": ("FLOAT", {
                    "default": 0,
                    "max": 1.0,
                    "min": 0,
                    "step": 0.01
                }),
                "mask_split": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
            },
            "optional": {
                "face": ("IMAGE",), 
                "long": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE","MASK","MASK")
    RETURN_NAMES = ("equ","mask","masks")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "node"

    def node(self, height, scale, mask_split, face = None, long = None):
        sub_height = 0
        result = np.array([])
        order = "nearest"
        if long is not None:
            sub_height, sub_width = long.shape[1], long.shape[2]

            if sub_height*6 != sub_width:
                raise ValueError("传入图像宽高必须为6:1")
            if height == 0:
                height = calculate_equ_heght(sub_height)

            long_face = long.permute(0, 3, 1, 2)
            equ = py360.cube2equi(
                cubemap = long_face, 
                height = height, 
                width = height * 2, 
                mode = order, 
                cube_format = 'horizon')  

            result = equ.permute(1, 2, 0).unsqueeze(0)

        elif face is not None:
            if not all(img.shape == face[0].shape for img in face):
                raise ValueError("所有图像的尺寸必须相同")

            sub_height = face[0].shape[0]
            if height == 0:
                height = calculate_equ_heght(sub_height)
                
            face = tuple(face.permute(0, 3, 1, 2))
            long_face = torch.cat(face, dim=2)
            equ = py360.cube2equi(
                        cubemap = long_face, 
                        height = height, 
                        width = height * 2, 
                        mode = order, 
                        cube_format = 'horizon')
            
            result = equ.permute(1, 2, 0).unsqueeze(0)

        else:
            if scale < 0.1:
                raise ValueError("传入图像不能为空")
            if height == 0:
                height = 832
            sub_height = height // 2   

        if scale > 0.01:
            # 缩放并居中放置图像    
            def create_sub_image(height, scale, value):
                H = int(height * scale)
                # 计算目标位置
                T = (height - H) // 2
                # 创建目标大小的全零图像，形状为 [C, H, W]
                result = torch.zeros((4, height, height), dtype=torch.uint8)
                # 将缩放后的图像居中放置
                result[:, T:T+H, T:T+H] = torch.full((4, H, H), value, dtype=torch.uint8)
                return result
            
            # 创建一个形状为 [C, H, W] 的组合图像
            np_image = torch.zeros((4, sub_height, sub_height * 6), dtype=torch.uint8)

            for i in range(6):
                value = (i + 1) * 10
                sub_image = create_sub_image(sub_height, scale, value)
                # 将子图像拼接到 np_image 中
                np_image[:, :, i * sub_height:(i + 1) * sub_height] = sub_image
            equ = py360.cube2equi(
                    cubemap = np_image, 
                    height = height, 
                    width = height * 2, 
                    mode = order, 
                    cube_format='horizon') 
            
            equ_mask = equ.permute(1, 2, 0).unsqueeze(0) 
            
            # 转化输入为空时候，使用equ_mask返回
            if np.size(result) == 0: 
                result = equ_mask / 255.0
                result = result.to(torch.float32)

            #提取透明通道转换为mask
            mask = equ_mask[:, :, :, 3]
            mask = (mask != 0).float()

            #提取颜色通道转换为mask
            masks = []
            blue_channel = equ_mask[0][:, :, 2]
            for i in range(6):
                # 直接使用蓝色通道的值来创建遮罩
                matching_pixels = (blue_channel == (i+1)*10).float()

                # 转换为 PyTorch 张量
                masks.append(matching_pixels)

            masks = torch.stack(masks, dim=0)

            if result.shape[-1] == 3:
                result = add_alpha_channel(result)

            if mask_split:
                funmaskand = nodes.NODE_CLASS_MAPPINGS['BitwiseAndMask']()
                mask = result[:, :, :, 3]
                mask_a = [1.0 - mask] * 6
                mask_a = torch.stack(mask_a, dim=0)
                masks = funmaskand(mask_a, masks)

            masks = masks.unsqueeze(0)

            return (result, mask, masks)
        else:
            # 当scale为0时，返回全透明
            mask = result[:, :, :, 1]
            masks = torch.zeros(1, 6, 64, 64)
            return (result, mask, masks)


class PanoImageEqu2Cube:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/convert"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "equ": ("IMAGE",),
                "height": ("INT", {
                    "default": 832,
                    "min": 0,
                    "step": 8
                }),
                "order": (["nearest", "bilinear"],),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","MASK")
    RETURN_NAMES = ("face","long","long_mask")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "node"

    def node(self, equ, height, order):

        image = equ
        sub_height, sub_width = image.shape[1], image.shape[2]

        if sub_height*2 != sub_width:
            raise ValueError("传入图像宽高必须为2:1")
        
        if height == 0:
            height = calculate_face_heght(sub_height)

        Rot = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}]
        equ = image.permute(0, 3, 1, 2)
        long_image = py360.equi2cube(
            equi = equ, 
            rots = Rot,
            w_face = height, 
            mode = order, 
            cube_format = 'horizon') 
        long_image = long_image.permute(0, 2, 3, 1)

        # 分割图像
        face_images = []
        for i in range(6):
            start_idx = i * height
            end_idx = (i + 1) * height
            face_image = long_image[0][:, start_idx:end_idx, :]
            face_images.append(face_image)
        
        # 将分割后的图像堆叠成一个新的张量
        face_images = torch.stack(face_images, dim=0)  # 形状变为 [6, H, H, C]
        face = face_images.unsqueeze(0)

        # 添加alpha通道
        if face.shape[-1] == 3:
            face = add_alpha_channel(face)
        if long_image.shape[-1] == 3:
            long_image = add_alpha_channel(long_image)

        # 简单生成MASK
        mask = long_image[:, :, :, 3]

        return (face, long_image, mask)

def calculate_vfov(hfov, width, height):
    # 将水平视场角从度转换为弧度
    hfov_rad = hfov * np.pi / 180

    # 计算水平方向的半视场角
    half_hfov_rad = hfov_rad / 2

    # 利用宽高比计算垂直方向的半视场角的正切值
    tan_half_hfov = np.tan(half_hfov_rad)
    tan_half_vfov = tan_half_hfov * (height / width)

    # 计算垂直方向的半视场角
    half_vfov_rad = np.arctan(tan_half_vfov)

    # 计算垂直视场角并转换回度
    vfov = 2 * half_vfov_rad * 180 / np.pi

    return vfov

class PanoImageEqu2Equ:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/convert"   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "equ": ("IMAGE",),
                "u_deg": ("FLOAT", {"default": 0.0,"min": -180,"max": 180,"step": 0.1}),
                "v_deg": ("FLOAT", {"default": 0.0,"min": -90,"max": 90,"step": 0.1}),
                "width": ("INT", {"default": 1664,"min": 0,"step": 8}),
                "height": ("INT", {"default": 832,"min": 0,"step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("equ", )
    FUNCTION = "node"

    def node(self, equ, u_deg, v_deg, width, height): 
        image = equ.permute(0, 3, 1, 2)

        if width == 0:
            width = image.shape[3]
        if height == 0:
            height = image.shape[2]

        # rotations
        Rot = [{
            'roll': 0.,
            'pitch': np.pi * v_deg / 180,  # rotate vertical
            'yaw': np.pi * u_deg / 180,  # rotate horizontal
        }]

        pic = py360.equi2equi(
            src = image, 
            rots = Rot,
            height = height, 
            width = width) 
        
        result = pic.permute(0, 2, 3, 1)
        return (result,)


class PanoImagePic2Equ:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/convert"   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "hfov": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 180.0, "step": 0.1}),
                "u_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "v_deg": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "height": ("INT", {"default": 1024,"min": 8,"step": 8})
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("equ", )
    FUNCTION = "node"

    def node(self, image, hfov, u_deg, v_deg, roll, height):

        B = image.shape[0]
        processed_images = []

        x_yaw = -u_deg
        y_pitch = -v_deg

        for i in range(B):
            # Process the image using the method
            image_i = image[i]
            vfov = calculate_vfov(hfov, image_i.shape[1], image_i.shape[0])
            processed_image = plane_to_equirect(
                image_i,
                hfov,
                vfov,
                x_yaw,
                y_pitch,
                roll,
                height + height
            )
            # Append the processed image to the list
            processed_images.append(processed_image)

        # Aggregate the processed images into a tensor of shape [B, H, W, C]
        output_tensor = torch.stack(processed_images, dim=0)

        return (output_tensor,)

class PanoImageEqu2Pic:
    def __init__(self):
        pass

    CATEGORY = "PanoCard/convert"   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "equ": ("IMAGE",),
                "u_deg": ("FLOAT", {"default": 0.0,"min": -180,"max": 180,"step": 0.1}),
                "v_deg": ("FLOAT", {"default": 0.0,"min": -90,"max": 90,"step": 0.1}),
                "hfov": ("FLOAT", {"default": 90.0,"min": 0,"max": 180,"step": 0.1}),
                "width": ("INT", {"default": 512,"min": 8,"step": 8}),
                "height": ("INT", {"default": 512,"min": 8,"step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "node"

    def node(self, equ, u_deg, v_deg, hfov, width, height): 
        
        image = equ.permute(0, 3, 1, 2)

        # rotations
        Rot = [{
            'roll': 0.,
            'pitch': np.pi * v_deg / 180,  # rotate vertical
            'yaw': np.pi * u_deg / 180,  # rotate horizontal
        }]

        pic = py360.equi2pers(
            equi = image, 
            rots = Rot,
            fov_x = hfov,
            height = height, 
            width = width) 
        
        result = pic.permute(0, 2, 3, 1)
        return (result,)