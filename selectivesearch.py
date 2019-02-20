import cv2
import torch


def selectiveSearch(im, width=64, height=128, transform=None):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    newHeight = 200
    factor = 200.0 / im.shape[0]
    newWidth = int(im.shape[1] * 200 / im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    selectiveSearchSegmentation = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selectiveSearchSegmentation.setBaseImage(im)
    selectiveSearchSegmentation.switchToSelectiveSearchFast()
    rects = selectiveSearchSegmentation.process()

    rects = rects.astype(int)
    rects = rects[:200]
    patches = torch.zeros((len(rects), 3, height, width))

    for i in range(len(rects)):
        rect = rects[i]
        x, y, w, h = rect
        patch = im[y:y + h, x:x + h].copy()
        new_patch = cv2.resize(patch, (width, height))
        #     patches[i] =

        if transform is not None:
            patches[i] = transform(new_patch)

    outRects = rects.astype(float) / factor
    outRects = outRects.astype(int)

    return patches, outRects
