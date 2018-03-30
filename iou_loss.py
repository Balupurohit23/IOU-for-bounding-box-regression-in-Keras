def iou_loss(y_true, y_pred):
    AoB = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)
    
    
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])
    
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)
    
    union = AoB + AoP - intersection
    
    iou = intersection/union
    
    iou = K.clip(iou, 0.0, 1.0)
    
    iou_loss = -K.log(iou)
    
    return iou_loss
