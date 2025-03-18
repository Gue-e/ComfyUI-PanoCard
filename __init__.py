from .pano import *
from .convert import *

__all__ = ['NODE_CLASS_MAPPINGS']

NODE_CLASS_MAPPINGS = {
    "PanoCardViewer": PanoViewer,
    "ImageWightPad": PanoImageWightPad,
    "ImageHeightPad": PanoImageHeightPad,
    "ImageCube2Equ": PanoImageCube2Equ,
    "ImageEqu2Cube": PanoImageEqu2Cube,
    "ImageEqu2Pic": PanoImageEqu2Pic,
    "ImagePic2Equ": PanoImagePic2Equ,
    "ImageEqu2Equ": PanoImageEqu2Equ,
    "ImageClamp": PanoImageClamp,
    "ImageOutClamp": PanoImageOutClamp,
    "MaskOutClamp": PanoMaskOutClamp,
    "PipePad": PanoImagePipe,
    "ImagePad": PanoImagePad,
    "ImageRoll": PanoImageRoll,
    "MaskCondBatch":PanoDenseDiffCondBatch,
    "ImageFaceCondBatch":PanoClipBatch,
    "PromptSplit":PanoPromptSplit,
    "RegionalPrompt":PanoRegionalPrompt,
    "ImageEquSplit":PanoImageSplit,
    "CondFaceSplit":PanoClipOutClamp,
    "ImageAdjust":PanoImageAdjust,
    "MaskFaceOutClamp": PanoMaskOutFaceClamp,
    "ImageFaceToLong": PanoFaceToLong,
    "CondFaceScheduleHookProvider": CondFaceScheduleHookProvider,
}


WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']