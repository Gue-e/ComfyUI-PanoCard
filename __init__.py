from .pano import *
from .convert import *

__all__ = ['NODE_CLASS_MAPPINGS']

NODE_CLASS_MAPPINGS = {
    "PanoCardViewer": PanoViewer,
    "ImageWightPad": PanoImageWightPad,
    "ImageHeightPad": PanoImageHeightPad,
    "ImageFace2Equ": PanoImageCube2Equ,
    "ImageEqu2Face": PanoImageEqu2Cube,
    "ImageEqu2Pic": PanoImageEqu2Pic,
    "ImagePic2Equ": PanoImagePic2Equ,
    "ImageEqu2Equ": PanoImageEqu2Equ,
    "ImageFaceClamp": PanoImageClamp,
    "ImageUnPack": PanoImageOutClamp,
    "MaskUnPack": PanoMaskOutClamp,
    "CondFaceUnPack":PanoClipOutClamp,
    "PaddingPipe": PanoImagePipe,
    "ImagePad": PanoImagePad,
    "ImageRoll": PanoImageRoll,
    "CondAllBatch":PanoDenseDiffCondBatch,
    "CondFaceBatch":PanoClipBatch,
    "PromptSplit":PanoPromptSplit,
    "RegionalPrompt":PanoRegionalPrompt,
    "Image2FaceSplit":PanoImageSplit,
    "ImageAdjust":PanoImageAdjust,
    "LongMaskSplit": PanoMaskOutFaceClamp,
    "ImageFaceToLong": PanoFaceToLong,
    "CondFaceDetailerHook": CondFaceScheduleHookProvider,
}


WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']