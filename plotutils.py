import cv2

import matplotlib.pyplot as plt


def imshowArray(arr):
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    plt.imshow(img)
    plt.grid(b=False)


def imshowArrayBB(arr, bblist, noCvtColor=False):
    if not noCvtColor:
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        img = arr
    imOut = img.copy()
    for i, rect in enumerate(bblist):
        x, y, w, h = map(round, rect)
        cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    plt.imshow(imOut)
    plt.grid(b=False)


def imshowArrayBBTrueAndTest(arr, bblistTrue, bblistTest, noCvtColor=False):
    if not noCvtColor:
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        img = arr
    imOut = img.copy()
    for i, rect in enumerate(bblistTrue):
        x, y, w, h = map(round, rect)
        cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    for i, rect in enumerate(bblistTest):
        x, y, w, h = map(round, rect)
        cv2.rectangle(imOut, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_AA)

    plt.figure(figsize=(15, 15))
    plt.imshow(imOut)
    plt.grid(b=False)
