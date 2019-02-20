import numpy as np


def IOU(bbox1, bbox2):
    x1Left = bbox1[0]
    x1Right = bbox1[2] + bbox1[0]
    x1Top = bbox1[1]
    x1Bottom = bbox1[3] + bbox1[1]

    x2Left = bbox2[0]
    x2Right = bbox2[2] + bbox2[0]
    x2Top = bbox2[1]
    x2Bottom = bbox2[3] + bbox2[1]

    if x1Right < x2Left:
        return 0

    if x2Right < x1Left:
        return 0

    newLeft = max(x1Left, x2Left)
    newRight = min(x1Right, x2Right)

    if x1Bottom < x2Top:
        return 0

    if x2Bottom < x1Top:
        return 0

    newTop = max(x1Top, x2Top)
    newBottom = min(x1Bottom, x2Bottom)

    newWidth = newRight - newLeft
    newHeight = newBottom - newTop
    intersectionArea = newWidth * newHeight

    x1Width = x1Right - x1Left
    x1Height = x1Bottom - x1Top
    x1Area = x1Width * x1Height

    x2Width = x2Right - x2Left
    x2Height = x2Bottom - x2Top
    x2Area = x2Width * x2Height

    return intersectionArea / (x1Area + x2Area - intersectionArea)


def buildIOUTable(trueRects, testRects, threshold=0.0):
    IOUTable = np.zeros((len(trueRects), len(testRects)))

    for i, trueBB in enumerate(trueRects):
        for j, testBB in enumerate(testRects):
            iou = IOU(trueBB, testBB)
            if iou > threshold:
                IOUTable[i, j] = iou
    return IOUTable


def matchBestFitRectangles(IOUTable, testRects):
    bestFitIndexList = np.full(IOUTable.shape[0], -1)
    usedIndices = []

    for i in range(IOUTable.shape[0]):
        candidates = np.argsort(IOUTable[i])[::-1]

        for x in candidates:
            if x not in usedIndices:
                bestFitIndexList[i] = x
                usedIndices.append(x)
                break

    bestFitRectangles = testRects.take(bestFitIndexList, axis=0)

    bestFitIOUList = np.zeros(IOUTable.shape[0])
    for i in range(IOUTable.shape[0]):
        bestFitIOUList[i] = IOUTable[i, bestFitIndexList[i]]

    return bestFitRectangles, bestFitIOUList, bestFitIndexList

