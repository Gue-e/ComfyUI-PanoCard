from .pano import *
from .convert import *

__all__ = ['NODE_CLASS_MAPPINGS']

NODE_CLASS_MAPPINGS = {
    "PanoCardViewer": PanoViewer,
    "PanoImageWidthPad": PanoImageWidthPad,
    "PanoImageHeightPad": PanoImageHeightPad,
    "PanoImageFace2Equ": PanoImageCube2Equ,
    "PanoImageEqu2Face": PanoImageEqu2Cube,
    "PanoImageEqu2Pic": PanoImageEqu2Pic,
    "PanoImagePic2Equ": PanoImagePic2Equ,
    "PanoImageEqu2Equ": PanoImageEqu2Equ,
    "PanoImageFaceClamp": PanoImageClamp,
    "PanoImageUnPack": PanoImageOutClamp,
    "PanoMaskUnPack": PanoMaskOutClamp,
    "PanoCondFaceUnPack":PanoClipOutClamp,
    "PanoPipe": PanoImagePipe,
    "PanoImagePad": PanoImagePad,
    "PanoImageRoll": PanoImageRoll,
    "PanoCondAllBatch":PanoMaskCondBatch,
    "PanoCondFaceBatch":PanoClipBatch,
    "PanoCondFaceClamp":PanoCondClipClamp,
    "PanoPromptSplit":PanoPromptSplit,
    "PanoRegionalPrompt":PanoRegionalPrompt,
    "PanoImage2FaceSplit":PanoImageSplit,
    "PanoImageAdjust":PanoImageAdjust,
    "PanoLongMaskSplit": PanoMaskOutFaceClamp,
    "PanoImageFaceToLong": PanoFaceToLong,
    "PanoCondFaceDetailerHook": CondFaceScheduleHookProvider,
}


WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']