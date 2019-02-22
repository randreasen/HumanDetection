from HumanDetection.boundingbox import *

def metrics(dets,gt,threshold_IOU=0.5):
    if dets.size==0:
        print("dets size = 0")
        return 0,0,len(gt)
    elif gt.size==0:
        print("gt size = 0")
        return 0,len(dets),0
    TP={}
    FN=[]
    FP=[]
    M=buildIOUTable(dets,gt)
    taken=[]
    taken_val=[]
    for i,d in enumerate(dets):
        j=np.argmax(M[i])
        if M[i,j] >threshold_IOU:
            try:
                if M[i,j] > TP[j][1]:
                    TP[j]=[i,M[i,j]]
                else:
                     FP.append(i)
            except:
                TP[j]=[i,M[i,j]]
        else:
            FP.append(i)

    for j,g in enumerate(gt):
        if not j in TP:
            FN.append(j)
    TP=len(TP.keys())
    FP=len(FP)
    FN=len(FN)

    return TP,FP,FN


