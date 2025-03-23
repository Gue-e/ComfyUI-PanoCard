import math
import numpy as np
import torch
import torch.nn.functional as F
from .misc import *

def depolar_transform(
    input_tensor: torch.Tensor,
    output_size: tuple,
    angle: float = 0.0,
    exponent: float = 1.0,
    background_color: tuple = (0, 0, 0, 0),
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """
    将输入图像进行反极坐标（Depolar）变换，支持透明背景和高效的插值方式。
    
    参数:
        input_tensor (torch.Tensor): 输入图像张量，形状为 [B, H, W, C]，类型为 float32。
        output_size (tuple): 输出图像的尺寸 (宽度, 高度)。
        angle (float): 控制扭曲的角度（以度为单位）。
        exponent (float): 调整扭曲的非线性程度。
        background_color (tuple): 背景颜色的RGBA值，四个整数，范围 [0, 255]。
        device (torch.device): 计算设备，默认为可用的 GPU 或 CPU。
    
    返回:
        torch.Tensor: 变换后的图像，形状为 [B, H_out, W_out, C]，类型为 float32。
    """
    assert input_tensor.dim() == 4, "输入张量必须是 4 维的 [B, H, W, C]"
    B, H_in, W_in, C = input_tensor.shape
    W_out, H_out = output_size
    
    # 转换为 [B, C, H, W] 以适应 grid_sample 的输入格式
    input_tensor = input_tensor.permute(0, 3, 1, 2).to(device)  # [B, C, H_in, W_in]
    
    # 创建输出图像的网格
    # 生成两组线性空间并创建 meshgrid
    theta = math.radians(90 + angle)  # 转换角度到弧度
    lin_w = torch.linspace(0, 1, steps=W_out, device=device)
    lin_h = torch.linspace(0, 1, steps=H_out, device=device)
    grid_w, grid_h = torch.meshgrid(lin_w, lin_h, indexing='ij')  # [W_out, H_out]
    
    grid_w = grid_w.t()  # 转置以匹配 H_out, W_out
    grid_h = grid_h.t()
    
    # 应用指数调节非线性程度
    r = grid_h ** exponent  # [H_out, W_out]
    
    # 计算极坐标(theta和r)
    theta_grid = grid_w * 2 * math.pi + theta  # 角度范围 [theta, theta + 2π]
    
    # 转换为笛卡尔坐标并归一化至 [-1, 1]
    x = (r * torch.cos(theta_grid)).unsqueeze(0).unsqueeze(-1)  # [1, H_out, W_out, 1]
    y = (r * torch.sin(theta_grid)).unsqueeze(0).unsqueeze(-1)  # [1, H_out, W_out, 1]
    
    # 将坐标从 [-1,1] 映射到 grid_sample 所需的范围
    grid = torch.cat((x, y), dim=-1)  # [1, H_out, W_out, 2]
    grid = grid.repeat(B, 1, 1, 1)  # [B, H_out, W_out, 2]
    
    # 使用 grid_sample 进行插值
    # 设定 padding_mode 为 'zeros' 以便后续可以处理背景颜色
    sampled = F.grid_sample(
        input_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [B, C, H_out, W_out]
    
    # 处理透明背景
    # 创建掩码：超出输入图像区域的部分为 0，否则为 1
    # 因为 grid_sample 使用 padding_mode='zeros'，所以这些区域在 sampled 中为 0
    if C == 4:
        alpha = sampled[:, 3:, :, :]  # 获取 alpha 通道
        mask = (alpha > 0).float()
    else:
        # 如果没有 alpha 通道，假设完全不透明
        mask = torch.ones_like(sampled[:, :1, :, :])
    
    # 转换背景颜色为 float32 并归一化
    bg_color = torch.tensor(background_color, dtype=torch.float32, device=device).view(1, C, 1, 1) / 255.0
    bg_color = bg_color.repeat(B, 1, H_out, W_out)
    
    # 将 sampled 和背景颜色按掩码进行混合
    output = sampled * mask + bg_color * (1 - mask)
    
    # 转换回 [B, H_out, W_out, C] 格式
    output = output.permute(0, 2, 3, 1)  # [B, H_out, W_out, C]
    
    return output.float()
'''
def plane_to_cylinder(
    input_image: torch.Tensor, 
    fov: float = 70.0, 
    interpolation: str = 'nearest',
    output_size: tuple = None
) -> torch.Tensor:
    """
    将平面图像转换为等距圆柱投影的图像，支持指定输出尺寸、背景颜色（包括透明度），
    以及选择插值方式。

    参数:
        input_image (torch.Tensor): 输入图像张量，形状为 [B, H, W, C]，类型为 float32，范围为 [0, 1]
        fov (float): 视场角，单位为度，默认为70度
        interpolation (str): 插值方式，支持 'bilinear' 和 'nearest'，默认 'bilinear'
        output_size (tuple, optional): 输出图像的尺寸，格式为 (输出高度, 输出宽度)。如果未指定，则使用输入图像的尺寸

    返回:
        torch.Tensor: 转换后的图像张量，形状为 [B, Output_H, Output_W, C]，类型为 float32，范围为 [0, 1]
    """

    # 验证插值方式
    if interpolation not in ['bilinear', 'nearest']:
        raise ValueError("插值方式仅支持 'bilinear' 和 'nearest'。")
    
    # 设置输出尺寸
    if output_size is not None:
        if (not isinstance(output_size, tuple)) or (len(output_size) != 2):
            raise ValueError("output_size 应为一个包含两个整数的元组，例如 (800, 1200)。")
        output_height, output_width = output_size
    else:
        B, H, W, C = input_image.shape
        output_height, output_width = H, W

    device = input_image.device
    dtype = input_image.dtype

    B_in, H_in, W_in, C = input_image.shape
    if C == 3:
        # 添加 alpha 通道
        input_image = add_alpha_channel(input_image)

    # 转换输入图像形状为 [B, C, H, W]
    input_image = input_image.permute(0, 3, 1, 2).to(dtype)  # [B, C, H_in, W_in]

    # 计算圆柱半径 R
    fov_rad = math.radians(fov)
    R = (W_in / 2) / math.tan(fov_rad / 2)

    # 创建输出网格，确保所有操作在输入图像的设备上执行
    # 创建标准化坐标网格，范围 [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_height, device=device, dtype=dtype),
        torch.linspace(-1, 1, output_width, device=device, dtype=dtype),
        indexing='ij'  # 确保维度顺序正确
    )
    # 扩展至批次维度
    x = x.unsqueeze(0).expand(B_in, -1, -1)  # [B, Output_H, Output_W]
    y = y.unsqueeze(0).expand(B_in, -1, -1)  # [B, Output_H, Output_W]

    # 计算 theta (水平角度)
    # 将x从 [-1, 1] 映射到 [-W_in/2, W_in/2]
    X = x * (W_in / 2)
    theta = X / R  # [B, Output_H, Output_W]

    # 防止 theta 过大，导致 tan(theta) 过大
    max_theta = math.tan(fov_rad / 2)
    theta_clipped = theta.clamp(-max_theta, max_theta)  # [B, Output_H, Output_W]

    # 计算输入图像的 x 和 y 坐标
    x_in = R * torch.tan(theta_clipped) + (W_in / 2)  # [B, Output_H, Output_W]
    cos_theta = torch.cos(theta_clipped)  # [B, Output_H, Output_W]
    cos_theta = cos_theta.clamp(min=1e-6)
    y_in = y * (H_in / 2) / cos_theta + (H_in / 2)  # [B, Output_H, Output_W]

    # 将 [0, W_in-1] 和 [0, H_in-1] 映射到 [-1, 1]
    x_norm = 2.0 * x_in / (W_in - 1) - 1.0  # [B, Output_H, Output_W]
    y_norm = 2.0 * y_in / (H_in - 1) - 1.0  # [B, Output_H, Output_W]

    # 创建 grid，形状为 [B, Output_H, Output_W, 2]
    grid = torch.stack((x_norm, y_norm), dim=-1)  # [B, Output_H, Output_W, 2]

    # 使用 grid_sample 进行采样
    mode = 'bilinear' if interpolation == 'bilinear' else 'nearest'
    sampled = F.grid_sample(
        input_image, grid, mode=mode, padding_mode='zeros', align_corners=True
    )  # [B, C, Output_H, Output_W]

    # 将采样结果转换为 [B, Output_H, Output_W, C]
    output = sampled.permute(0, 2, 3, 1)  # [B, Output_H, Output_W, C]

    return output  # [B, Output_H, Output_W, C]
'''

def plane_to_cylinder(
    input_image: torch.Tensor, 
    fov: float = 70.0, 
    interpolation: str = 'nearest',
    output_size: tuple = None,
    crop: bool = True
) -> torch.Tensor:

    # 验证插值方式
    if interpolation not in ['bilinear', 'nearest']:
        raise ValueError("插值方式仅支持 'bilinear' 和 'nearest'。")
    
    # 设置输出尺寸
    if output_size is not None:
        if (not isinstance(output_size, tuple)) or (len(output_size) != 2):
            raise ValueError("output_size 应为一个包含两个整数的元组，例如 (800, 1200)。")
        output_height, output_width = output_size
    else:
        B, H, W, C = input_image.shape
        output_height, output_width = MUT8(H*1.2), MUT8(W*1.2)

    device = input_image.device
    dtype = input_image.dtype

    B_in, H_in, W_in, C = input_image.shape
    if C == 3:
        # 添加 alpha 通道
        input_image = add_alpha_channel(input_image)

    # 转换输入图像形状为 [B, C, H, W]
    input_image = input_image.permute(0, 3, 1, 2).to(dtype)  # [B, C, H_in, W_in]

    # 计算圆柱半径 R 使用绝对值fov
    abs_fov = abs(fov)
    abs_fov_rad = math.radians(abs_fov)
    R = (W_in / 2) / math.tan(abs_fov_rad / 2)
    max_theta = math.tan(abs_fov_rad / 2)

    # 创建输出网格
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, output_height, device=device, dtype=dtype),
        torch.linspace(-1, 1, output_width, device=device, dtype=dtype),
        indexing='ij'
    )
    x = x.unsqueeze(0).expand(B_in, -1, -1)  # [B, Output_H, Output_W]
    y = y.unsqueeze(0).expand(B_in, -1, -1)  # [B, Output_H, Output_W]

    # 计算 theta (水平角度)
    X = x * (W_in / 2)
    theta = X / R  # [B, Output_H, Output_W]

    # 限制 theta 范围
    theta_clipped = theta.clamp(-max_theta, max_theta)

    # 计算输入图像的坐标
    x_in = R * torch.tan(theta_clipped) + (W_in / 2)  # [B, Output_H, Output_W]
    cos_theta = torch.cos(theta_clipped).clamp(min=1e-6)
    
    # 根据fov符号调整垂直方向计算
    if fov < 0:
        # 动态调整垂直缩放，确保顶点不裁剪
        cos_theta_min = math.cos(max_theta)  # 最小cos(theta)对应最大x偏移
        scale_factor = 1.0 / cos_theta_min  # 基于最小cos(theta)缩放
        y_in = y * (H_in / 2) * (cos_theta * scale_factor) + (H_in / 2)
    else:
        y_in = y * (H_in / 2) / cos_theta + (H_in / 2)

    # 归一化到[-1, 1]
    x_norm = 2.0 * x_in / (W_in - 1) - 1.0
    y_norm = 2.0 * y_in / (H_in - 1) - 1.0

    grid = torch.stack((x_norm, y_norm), dim=-1)  # [B, H, W, 2]

    # 采样
    mode = 'bilinear' if interpolation == 'bilinear' else 'nearest'
    sampled = F.grid_sample(
        input_image, grid, mode=mode, padding_mode='zeros', align_corners=True
    )

    # 调整输出形状
    output = sampled.permute(0, 2, 3, 1)

    # 自动裁剪填充的零区域
    if crop:
        # 计算有效坐标范围
        x_in_valid = (x_in >= 0) & (x_in <= W_in - 1)  # [B, H, W]
        y_in_valid = (y_in >= 0) & (y_in <= H_in - 1)  # [B, H, W]

        # 找出有效列（水平方向）
        cols_valid = x_in_valid.any(dim=1)  # [B, W]
        left = torch.argmax(cols_valid.int(), dim=1)  # [B]
        reversed_cols = torch.flip(cols_valid.int(), dims=[1])
        right = output_width - torch.argmax(reversed_cols, dim=1) - 1  # [B]

        # 找出有效行（垂直方向）
        rows_valid = y_in_valid.any(dim=2)  # [B, H]
        top = torch.argmax(rows_valid.int(), dim=1)  # [B]
        reversed_rows = torch.flip(rows_valid.int(), dims=[1])
        bottom = output_height - torch.argmax(reversed_rows, dim=1) - 1  # [B]

        # 假设所有batch的裁剪范围相同
        left = left[0].item()
        right = right[0].item()
        top = top[0].item()
        bottom = bottom[0].item()

        # 执行双方向裁剪
        output = output[:, top:bottom+1, left:right+1, :]

    return output


# 修改来自 https://github.com/willchil/ComfyUI-Environment-Visualizer
def plane_to_equirect(input_tensor, HFOV, VFOV, yaw, pitch, roll, output_width=2048):
    """
    Maps an input image tensor to an equirectangular panoramic image using PyTorch tensors.

    Parameters:
        input_tensor (torch.Tensor): Input image as a tensor with shape [H, W, C] in RGB or RGBA format.
        HFOV (float): Horizontal Field of View in degrees.
        yaw (float): Yaw rotation in degrees.
        pitch (float): Pitch rotation in degrees.
        roll (float): Roll rotation in degrees.
        output_width (int): Width of the output equirectangular image (Height will be output_width // 2).

    Returns:
        equirect_image (torch.Tensor): Equirectangular image tensor with shape [output_width // 2, output_width, 4].
    """
    if input_tensor.ndim != 3:
        raise ValueError("Input tensor must be a 3-dimensional array [H, W, C].")

    H_in, W_in, C = input_tensor.shape

    # Handle images with or without an alpha channel
    if C == 4:
        # Input has alpha channel; preserve RGB and ignore input alpha
        input_image = input_tensor[:, :, :3].clone()
    elif C == 3:
        input_image = input_tensor.clone()
    else:
        raise ValueError("Input tensor must have 3 (RGB) or 4 (RGBA) channels.")

    # Calculate Vertical Field of View (VFOV) to match FOV per pixel
    # VFOV = HFOV * (H_in / W_in)

    # Convert FOV from degrees to radians
    HFOV_rad = torch.deg2rad(torch.tensor(HFOV, dtype=input_tensor.dtype, device=input_tensor.device))
    VFOV_rad = torch.deg2rad(torch.tensor(VFOV, dtype=input_tensor.dtype, device=input_tensor.device))

    # Compute focal lengths
    fx = (W_in / 2) / torch.tan(HFOV_rad / 2)
    fy = (H_in / 2) / torch.tan(VFOV_rad / 2)

    # Principal point (assuming centered)
    cx = W_in / 2
    cy = H_in / 2

    # Compute the rotation matrix from yaw, pitch, roll
    def rotation_matrix(yaw, pitch, roll):
        # Convert angles from degrees to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)

        # Rotation matrices around x, y, z axes
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ], dtype=input_tensor.dtype, device=input_tensor.device)

        Ry = torch.tensor([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ], dtype=input_tensor.dtype, device=input_tensor.device)

        Rz = torch.tensor([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ], dtype=input_tensor.dtype, device=input_tensor.device)

        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        return R

    # Compute rotation matrix and ensure it's contiguous
    R = rotation_matrix(yaw, pitch, roll).T  # Transpose for inverse rotation

    # Equirectangular image dimensions (Height is half of Width)
    W_out = output_width
    H_out = output_width // 2  # Enforce 2:1 aspect ratio

    # Create a meshgrid for the equirectangular image
    theta = (torch.linspace(0, W_out - 1, W_out, dtype=input_tensor.dtype, device=input_tensor.device) / W_out) * 2 * math.pi - math.pi  # theta from -π to π
    phi = (0.5 - (torch.linspace(0, H_out - 1, H_out, dtype=input_tensor.dtype, device=input_tensor.device) / H_out)) * math.pi       # phi from -π/2 to π/2

    # Use torch.meshgrid with proper ordering to get [H_out, W_out]
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='ij')  # Shape: [H_out, W_out]

    # Spherical to Cartesian coordinates (direction vectors)
    x_s = torch.cos(phi_grid) * torch.sin(theta_grid)  # Shape: [H_out, W_out]
    y_s = -torch.sin(phi_grid)                        # Shape: [H_out, W_out]
    z_s = torch.cos(phi_grid) * torch.cos(theta_grid)  # Shape: [H_out, W_out]

    # Stack into direction vectors
    dirs = torch.stack((x_s, y_s, z_s), dim=-1)  # Shape: [H_out, W_out, 3]

    # Rotate direction vectors to camera coordinate system
    dirs_cam = torch.matmul(dirs, R)  # Shape: [H_out, W_out, 3]
    dx_c, dy_c, dz_c = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]

    # Compute valid_mask before division to avoid divide by zero
    epsilon = torch.tensor(1e-6, dtype=input_tensor.dtype, device=input_tensor.device)
    valid_mask = dz_c > epsilon  # Points in front of the camera

    # Compute x_im and y_im
    x_im = (dx_c / dz_c) * fx + cx
    y_im = (dy_c / dz_c) * fy + cy

    # Update valid_mask with x_im and y_im in valid image range
    valid_mask &= (x_im >= 0) & (x_im < W_in) & (y_im >= 0) & (y_im < H_in)
    valid_mask &= torch.isfinite(x_im) & torch.isfinite(y_im)

    # Prepare grid for grid_sample
    # Normalize x_im and y_im to [-1, 1]
    grid_x = (x_im / (W_in - 1)) * 2 - 1  # Shape: [H_out, W_out]
    grid_y = (y_im / (H_in - 1)) * 2 - 1  # Shape: [H_out, W_out]

    # Ensure grid_x and grid_y are within [-1, 1]
    grid_x = torch.clamp(grid_x, -1.0, 1.0)
    grid_y = torch.clamp(grid_y, -1.0, 1.0)

    # Stack to create grid of shape [1, H_out, W_out, 2]
    grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: [H_out, W_out, 2]
    grid = grid.unsqueeze(0)  # Shape: [1, H_out, W_out, 2]

    # Ensure grid is contiguous
    grid = grid.contiguous()

    # Prepare input image tensor
    # Convert to float and permute to [C, H, W]
    input_image = input_image.to(dtype=input_tensor.dtype, device=input_tensor.device).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, C, H_in, W_in]

    # Perform remapping using grid_sample
    remapped = F.grid_sample(input_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # Shape: [1, C, H_out, W_out]

    # Create an alpha channel based on valid_mask
    alpha = valid_mask.unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, H_out, W_out]

    # Mask the remapped RGB image with the alpha channel to set invalid regions to zero
    remapped = remapped * alpha  # Zero out invalid regions

    # Combine the remapped RGB image with the alpha channel to create RGBA image
    equirect_image = torch.cat((remapped, alpha), dim=1)  # Shape: [1, C, H_out, W_out]

    # Squeeze the batch dimension and permute to [H_out, W_out, C+1]
    equirect_image = equirect_image.squeeze(0).permute(1, 2, 0)  # Shape: [H_out, W_out, C]

    equirect_image = equirect_image.clamp(0, 1)
    return equirect_image


def arc_distortion(image, stretch_factor=0.5):
    """
    实现垂直方向圆弧形扭曲，支持透明填充
    Args:
        image: 输入张量 [B,H,W,C] (C=3或4)
        stretch_factor: 扭曲强度（正值向下弯曲，负值向上）
    Returns:
        带透明通道的扭曲图像 [B,H,W,4]
    """
    B, H, W, C = image.shape
    device = image.device

    # 添加alpha通道
    if C == 3:
        alpha = torch.ones((B, H, W, 1), device=device)
        img_alpha = torch.cat([image, alpha], dim=3)
    else:
        img_alpha = image.clone()
    
    # 转换为BCHW格式
    img = img_alpha.permute(0, 3, 1, 2)  # [B, C, H, W]

    # 生成归一化网格坐标
    y_coord = torch.linspace(-1, 1, H, device=device)
    x_coord = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(y_coord, x_coord, indexing='ij')  # [H, W]

    # 垂直方向位移计算（基于水平位置）
    delta_y = -stretch_factor * gx**2  # 二次函数模拟垂直弯曲
    new_gy = gy + delta_y  # 应用垂直位移

    # 构建采样网格
    grid = torch.stack([gx, new_gy], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # 执行采样（使用零填充）
    distorted = F.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='zeros',  # 关键：超出区域填充透明
        align_corners=True
    )

    # 恢复维度并确保alpha通道正确
    output = distorted.permute(0, 2, 3, 1)  # [B, H, W, C]
    
    # 限制alpha在[0,1]范围
    output[..., 3:4] = torch.clamp(output[..., 3:4], 0, 1)
    
    return output

def concave_distortion(
    input_tensor: torch.Tensor,
    intensity: float = 0.5,
    background_color: tuple = (0, 0, 0, 0),
    device: torch.device = None
) -> torch.Tensor:
    """
    改进的凹陷变换函数，支持批次处理及透明通道
    :param input_tensor: 输入图像张量 [B, H, W, C]
    :param intensity: 凹陷强度 (0-1)，正值凹陷，负值凸起
    :param background_color: 背景RGBA颜色 (0-255)
    :param device: 指定计算设备
    :return: 变换后的图像 [B, H, W, C]
    """
    if device is None:
        device = input_tensor.device
    
    # 确保输入格式正确
    assert input_tensor.dim() == 4, "输入必须是4维张量 [B, H, W, C]"
    B, H, W, C = input_tensor.shape
    
    # 添加alpha通道
    if C == 3:
        alpha = torch.ones((B, H, W, 1), device=device)
        input_img = torch.cat([input_tensor, alpha], dim=-1)
    else:
        input_img = input_tensor.clone()
    
    # 转换为grid_sample需要的格式 [B, C, H, W]
    input_img = input_img.permute(0, 3, 1, 2).to(device)
    
    # 生成归一化网格 [-1, 1]
    xx = torch.linspace(-1, 1, W, device=device)
    yy = torch.linspace(-1, 1, H, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')  # [H, W]
    
    # 计算径向变形
    radius = torch.sqrt(gx**2 + gy**2) / np.sqrt(2)  # 归一化到[0,1]
    distortion = intensity * (1 - radius**3)         # 三次方曲线
    new_gx = gx * (1 + distortion)
    new_gy = gy * (1 + distortion)
    
    # 构建采样网格 [B, H, W, 2]
    grid = torch.stack([new_gx, new_gy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # 执行采样
    sampled = F.grid_sample(
        input_img,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [B, C, H, W]
    
    # 处理背景颜色
    bg_color = torch.tensor(background_color, dtype=torch.float32, device=device).view(1, 4, 1, 1) / 255.0
    alpha = sampled[:, 3:4, :, :]  # 获取alpha通道
    output = sampled[:, :4, :, :] * alpha + bg_color * (1 - alpha)
    
    # 转换回原始格式 [B, H, W, C]
    return output.permute(0, 2, 3, 1)
