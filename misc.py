import torch
import math
from torch import Tensor
import torchvision.transforms as t
from PIL import Image as ImageF
from PIL.Image import Image as ImageB
from typing import Literal, Any

def tensor_to_image(self):
    return t.ToPILImage()(self.permute(2, 0, 1))

def image_to_tensor(self):
    return t.ToTensor()(self).permute(1, 2, 0)

def get_sampler_by_name(method) -> Literal[0, 1, 2, 3, 4, 5]:
    if method == "lanczos":
        return ImageF.LANCZOS
    elif method == "bicubic":
        return ImageF.BICUBIC
    elif method == "hamming":
        return ImageF.HAMMING
    elif method == "bilinear":
        return ImageF.BILINEAR
    elif method == "box":
        return ImageF.BOX
    elif method == "nearest":
        return ImageF.NEAREST
    else:
        raise ValueError("Sampler not found.")

def create_rgba_image(width: int, height: int, color=(0, 0, 0, 0)) -> ImageB:
    return ImageF.new("RGBA", (width, height), color)

Tensor.tensor_to_image = tensor_to_image
ImageB.image_to_tensor = image_to_tensor

# 调用的ComfyUI_ALLOR 的代码

# 创建图片
def ImageContainer(width, height, red, green, blue, alpha):
    return create_rgba_image(width, height, (red, green, blue, int(alpha * 255))).image_to_tensor().unsqueeze(0)

#缩放大小
def ImageTransformResizeAbsolute(images, width, height, method):
    def resize_tensor(tensor):
        return tensor.tensor_to_image().resize((width, height), get_sampler_by_name(method)).image_to_tensor()

    return torch.stack([resize_tensor(images[i]) for i in range(len(images))])

#缩放比例
def ImageTransformResizeRelative(images, scale_width, scale_height, method):
    height, width = images[0, :, :, 0].shape

    width = int(width * scale_width)
    height = int(height * scale_height)

    return ImageTransformResizeAbsolute(images, width, height, method)

#缩放偏移位置
def ImageTransformResizeClip(images, max_width, max_height, min_width, min_height, method):
    height, width = images[0, :, :, 0].shape

    if min_width >= max_width or min_height >= max_height:
        return (images,)

    scale_min = max(min_width / width, min_height / height)
    scale_max = min(max_width / width, max_height / height)

    scale = max(scale_min, scale_max)

    return ImageTransformResizeRelative(images, scale, scale, method)

#两张图片绝对位置合成
def ImageCompositeAbsolute(
        images_a,
        images_b,
        images_a_x,
        images_a_y,
        images_b_x,
        images_b_y,
        container_width,
        container_height,
        background,
        method
):
    def clip(value: float):
        return value if value >= 0 else 0

    # noinspection PyUnresolvedReferences
    def composite(image_a, image_b):
        img_a_height, img_a_width, img_a_dim = image_a.shape
        img_b_height, img_b_width, img_b_dim = image_b.shape

        if img_a_dim == 3:
            image_a = torch.stack([
                image_a[:, :, 0],
                image_a[:, :, 1],
                image_a[:, :, 2],
                torch.ones((img_a_height, img_a_width))
            ], dim=2)

        if img_b_dim == 3:
            image_b = torch.stack([
                image_b[:, :, 0],
                image_b[:, :, 1],
                image_b[:, :, 2],
                torch.ones((img_b_height, img_b_width))
            ], dim=2)

        container_x = max(img_a_width, img_b_width) if container_width == 0 else container_width
        container_y = max(img_a_height, img_b_height) if container_height == 0 else container_height

        container_a = torch.zeros((container_y, container_x, 4))
        container_b = torch.zeros((container_y, container_x, 4))

        img_a_height_c, img_a_width_c = [
            clip((images_a_y + img_a_height) - container_y),
            clip((images_a_x + img_a_width) - container_x)
        ]

        img_b_height_c, img_b_width_c = [
            clip((images_b_y + img_b_height) - container_y),
            clip((images_b_x + img_b_width) - container_x)
        ]

        if img_a_height_c <= img_a_height and img_a_width_c <= img_a_width:
            container_a[
                images_a_y:img_a_height + images_a_y - img_a_height_c,
                images_a_x:img_a_width + images_a_x - img_a_width_c
            ] = image_a[
                :img_a_height - img_a_height_c,
                :img_a_width - img_a_width_c
            ]

        if img_b_height_c <= img_b_height and img_b_width_c <= img_b_width:
            container_b[
                images_b_y:img_b_height + images_b_y - img_b_height_c,
                images_b_x:img_b_width + images_b_x - img_b_width_c
            ] = image_b[
                :img_b_height - img_b_height_c,
                :img_b_width - img_b_width_c
            ]

        if background == "images_a":
            return ImageF.alpha_composite(
                container_a.tensor_to_image(),
                container_b.tensor_to_image()
            ).image_to_tensor()
        else:
            return ImageF.alpha_composite(
                container_b.tensor_to_image(),
                container_a.tensor_to_image()
            ).image_to_tensor()

    if method == "pair":
        if len(images_a) != len(images_b):
            raise ValueError("Size of image_a and image_b not equals for pair batch type.")

        return torch.stack([composite(images_a[i], images_b[i]) for i in range(len(images_a))])
    elif method == "matrix":
        return torch.stack([
            composite(images_a[i], images_b[j]) for i in range(len(images_a)) for j in range(len(images_b))
        ])

    return None

#两张图片相对位置合成
def ImageCompositeRelative(
        images_a,
        images_b,
        images_a_x,
        images_a_y,
        images_b_x,
        images_b_y,
        background,
        container_size_type,
        method
):
    def offset_by_percent(container_size: int, image_size: int, percent: float):
        return int((container_size - image_size) * percent)

    img_a_height, img_a_width = images_a[0, :, :, 0].shape
    img_b_height, img_b_width = images_b[0, :, :, 0].shape

    if container_size_type == "max":
        container_width = max(img_a_width, img_b_width)
        container_height = max(img_a_height, img_b_height)
    elif container_size_type == "sum":
        container_width = img_a_width + img_b_width
        container_height = img_a_height + img_b_height
    elif container_size_type == "sum_width":
        container_width = img_a_width + img_b_width
        container_height = max(img_a_height, img_b_height)
    elif container_size_type == "sum_height":
        container_width = max(img_a_width, img_b_width)
        container_height = img_a_height + img_a_height
    else:
        raise ValueError()

    return ImageCompositeAbsolute(
        images_a,
        images_b,
        offset_by_percent(container_width, img_a_width, images_a_x),
        offset_by_percent(container_height, img_a_height, images_a_y),
        offset_by_percent(container_width, img_b_width, images_b_x),
        offset_by_percent(container_height, img_b_height, images_b_y),
        container_width,
        container_height,
        background,
        method
    )

def ImageTransformRotate(images, angle, expand, SSAA, method):
    height, width = images[0, :, :, 0].shape
    def rotate_tensor(tensor):
        if method == "lanczos":
            resize_sampler = ImageF.LANCZOS
            rotate_sampler = ImageF.BICUBIC
        elif method == "bicubic":
            resize_sampler = ImageF.BICUBIC
            rotate_sampler = ImageF.BICUBIC
        elif method == "hamming":
            resize_sampler = ImageF.HAMMING
            rotate_sampler = ImageF.BILINEAR
        elif method == "bilinear":
            resize_sampler = ImageF.BILINEAR
            rotate_sampler = ImageF.BILINEAR
        elif method == "box":
            resize_sampler = ImageF.BOX
            rotate_sampler = ImageF.NEAREST
        elif method == "nearest":
            resize_sampler = ImageF.NEAREST
            rotate_sampler = ImageF.NEAREST
        else:
            raise ValueError()

        if SSAA > 1:
            img = tensor.tensor_to_image()
            img_us_scaled = img.resize((width * SSAA, height * SSAA), resize_sampler)
            img_rotated = img_us_scaled.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
            img_down_scaled = img_rotated.resize((img_rotated.width // SSAA, img_rotated.height // SSAA), resize_sampler)
            result = img_down_scaled.image_to_tensor()
        else:
            img = tensor.tensor_to_image()
            img_rotated = img.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
            result = img_rotated.image_to_tensor()

        return result

    if angle == 0.0 or angle == 360.0:
        return (images,)
    else:
        return torch.stack([
            rotate_tensor(images[i]) for i in range(len(images))
        ])

def ImageTransformTranspose(images, method):
    def transpose_tensor(tensor):
        if method == "flip_horizontally":
            transpose = ImageF.FLIP_LEFT_RIGHT
        elif method == "flip_vertically":
            transpose = ImageF.FLIP_TOP_BOTTOM
        elif method == "rotate_90":
            transpose = ImageF.ROTATE_90
        elif method == "rotate_180":
            transpose = ImageF.ROTATE_180
        elif method == "rotate_270":
            transpose = ImageF.ROTATE_270
        elif method == "transpose":
            transpose = ImageF.TRANSPOSE
        elif method == "transverse":
            transpose = ImageF.TRANSVERSE
        else:
            raise ValueError()

        return tensor.tensor_to_image().transpose(transpose).image_to_tensor()

    return torch.stack([transpose_tensor(images[i]) for i in range(len(images))])


def MUT8(width):
    # 向上取整到最接近的8的倍数
    adjusted_width = math.ceil(width / 8) * 8
    return int(adjusted_width)


def resize_mask(mask, shape):
    '''
    输入mask和图片的shape，输出resize后的mask
    shape: [H, W]
    '''
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                            size=(shape[0], shape[1]), mode="bilinear").squeeze(1)


def join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor):
    batch_size = min(len(image), len(alpha))
    out_images = []

    alpha = 1.0 - resize_mask(alpha, image.shape[1:])
    for i in range(batch_size):
        out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

    return torch.stack(out_images)


def add_alpha_channel(image):
    # Check the shape of the input image
    if image.dim() < 3:
        raise ValueError("Input image must have at least 3 dimensions.")
    
    # Get the shape of the image
    shape = image.shape
    
    # Create an alpha channel with the same shape as the last dimension except for the last dimension
    alpha_channel_shape = list(shape)
    alpha_channel_shape[-1] = 1
    
    # Create the alpha channel filled with ones
    alpha_channel = torch.ones(alpha_channel_shape, dtype=image.dtype, device=image.device)
    
    # Concatenate the alpha channel to the image
    image_with_alpha = torch.cat((image, alpha_channel), dim=-1)
    
    return image_with_alpha

# 图像遮罩复合
def image_mask_composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
    destination = destination.clone().movedim(-1, 1)
    source = source.clone().movedim(-1, 1)

    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")

    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion

    destination = destination.movedim(1, -1)

    return destination


def crop_transparent_area(image_tensor, mode='full'):
    """
    裁切掉图像中的透明部分。
    
    参数:
        image_tensor (torch.Tensor): 输入图像张量，形状为 [B, H, W, C]，类型为 float32。
        mode (str): 裁剪模式，支持 'full'（全裁剪）、'horizontal'（水平裁剪）、'vertical'（垂直裁剪），默认为 'full'。
    
    返回:
        torch.Tensor: 裁剪后的图像张量，形状为 [B, H', W', C]，类型为 float32。
    """
    alpha_channel = image_tensor[0, :, :, -1]
    non_zero_indices = torch.nonzero(alpha_channel > 0)
    
    if non_zero_indices.numel() == 0:
        return image_tensor  # 如果没有非透明部分，返回原图

    min_y, min_x = non_zero_indices.min(dim=0)[0]
    max_y, max_x = non_zero_indices.max(dim=0)[0]

    if mode == 'full':
        # 全裁剪
        cropped_image = image_tensor[:, min_y:max_y+1, min_x:max_x+1, :]
    elif mode == 'horizontal':
        # 按照水平裁剪
        cropped_image = image_tensor[:, :, min_x:max_x+1, :]
    elif mode == 'vertical':
        # 按照垂直裁剪
        cropped_image = image_tensor[:, min_y:max_y+1, :, :]
    else:
        raise ValueError("裁剪模式仅支持 'full'、'horizontal' 和 'vertical'。")

    return cropped_image

