# Libraries for analysis
import numpy as np
import cv2 as cv
import csv

frame_l, frame_w, bbox_l, bbox_w = 0, 0, 0, 0

def sysInit(frame1, bbox):
    global frame_l, frame_w, bbox_l, bbox_w
    frame_l, frame_w = np.size(frame1, 0), np.size(frame1, 1)
    bbox_l, bbox_w = bbox[2], bbox[3] # because length and width of bbox is constant, we will define them globally here

# This function, sets conditions for check if the child is inside of the camera's view or not.
# The condition here is if half of the bbox was in view (from any side), the child is inside.
# Otherwise, the child is outside of the view.

def inViewCheck(bbox):
    place = "Inside"
    
    if bbox[0] <    0    - 2 * bbox_l/4:
        place = "up-Out"
    if bbox[0] > frame_l - 2 * bbox_l/4:
        place = "down-Out"
    if bbox[1] <    0    - 2 * bbox_w/4:
        place = "left-Out"
    if bbox[1] > frame_w - 2 * bbox_w/4:
        place = "right-Out"
    
    return place

# This is s vectorized svm algorithm that we will use in our inline svm algorithm.

def svm_loss_vectorized(W, X, y, reg):
  
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]), y]  # http://stackoverflow.com/a/23435843/459241
    margins = np.maximum(0, scores - np.reshape(yi_scores, (-1, 1)) + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    loss += reg * np.sum(W * W)
    
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum.T
    dW = np.dot(X.T, binary)

    # Average
    dW /= num_train

    # Regularize
    dW += reg * W

    return loss, dW

# In this function, with the assumption that our tracking algorithm is mostly correct, we use our bbox in every frame
# as training dataset and collect our dataset online in this way. Therefore, we take frame and bbox as inputs of our
# function and calculate the proper W based on previous W.

def onlineSVM(frame1, bbox, W, direction):
    if (bbox[0] > 0 and bbox[0] < frame_l-bbox_l) and (bbox[1] > 0 and bbox[1] < frame_w-bbox_w): # This checks if child is fully inside or not
        if direction=="right": 
            if bbox[1] < frame_w-2*bbox_w:
                True_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]:bbox[1]+bbox_w]
                Fals_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]+bbox_w:bbox[1]+2*bbox_w]
            else:
                True_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]:bbox[1]+bbox_w]
                Fals_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]-bbox_w:bbox[1]]
        elif direction=="left":
            if bbox[1] > bbox_w:
                True_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]:bbox[1]+bbox_w]
                Fals_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]-bbox_w:bbox[1]]
            else:
                True_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]:bbox[1]+bbox_w]
                Fals_patch = frame1[bbox[0]:bbox[0]+bbox_l, bbox[1]+bbox_w:bbox[1]+2*bbox_w]
        
        x_train = np.zeros([2, bbox_l*bbox_w*3])
        x_train[0, :] = np.reshape(True_patch, (1, -1))
        x_train[1, :] = np.reshape(Fals_patch, (1, -1))
        x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
        
        data_loss, data_dW = svm_loss_vectorized(W, x_train, [1, 0], 0.000005) # 0.0005
        W = W - 1e-7 * data_dW
        ret = True
    else:
        ret = False
    
    return ret, W

# In this function, we take two consecutive frames as input of function and a bbox to determine where child was
# located in previous frame. Then, we calculate optical flow for child window and his/her next probable place
# based on direction which we take from input

def childFlow(frame1_gray, frame2_gray, bbox, direction):
    
    # In following lines, we will do two sets of things. First, due to our previous angle,
    # we will choose whether extend our checking bbox in the right side or left side. Second,
    # we check if our bbox is going outside or not and then, correcting its bounderies
    
    bbox_new = np.copy(bbox)
    # Check right side movement of bbox
    if direction == "right":
        bbox_new = np.array([bbox[0], bbox[1], bbox_l, 2*bbox_w])
        bbox_new = np.clip(bbox_new, [0, 0, 0, 0], [frame_l, frame_w, frame_l, frame_w])
    elif direction == "left":
        bbox_new = np.array([bbox[0], bbox[1]-bbox_w, bbox_l, 2*bbox_w])
        bbox_new = np.clip(bbox_new, [0, 0, 0, 0], [frame_l, frame_w, frame_l, frame_w])
    
    # Now, we will calculate optical flow and take our crop from previous step
    fr1_gray_crop = frame1_gray[bbox_new[0]:bbox_new[0]+bbox_new[2], bbox_new[1]:bbox_new[1]+bbox_new[3]]
    fr2_gray_crop = frame2_gray[bbox_new[0]:bbox_new[0]+bbox_new[2], bbox_new[1]:bbox_new[1]+bbox_new[3]]
    
    subtFrames = cv.absdiff(fr2_gray_crop, fr1_gray_crop)
    ret, thresh = cv.threshold(subtFrames, 25, 255, cv.THRESH_BINARY)
    
    flow = cv.calcOpticalFlowFarneback(fr1_gray_crop, fr2_gray_crop, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    
    u_crop = flow[...,0] * thresh/255
    v_crop = flow[...,1] * thresh/255
    
    return u_crop, v_crop

# In this function, we take our optical flow matrices from input and an orientation resolution that have been
# defined by user. In the output of function, we will have the dominant orientation and a magnitude histogram
# of child's movement

def DomOriMag(u_crop, v_crop, ori_res):
    
    mag, ang = cv.cartToPolar(u_crop, v_crop, angleInDegrees = True)
    
    ang = np.floor(ang/ori_res) # Quantizing the angles to 360/res bins
    ang = np.reshape(ang, (1, -1))
    
    mag = np.around(mag) # Because we don't need float percision for magnitude, we round them here
    mag = np.reshape(mag, (1, -1))
    
    oriBins = np.uint8(360/ori_res)
    OriMagHist = np.zeros(oriBins)
    for i in range(oriBins):
        indexes = np.where(ang == i)
        OriMagHist[i] = np.sum(mag[indexes])
        
    domOriTemp = np.argmax(OriMagHist)
    domOriMagsIdx = np.where(ang == domOriTemp)
    domOriMags = mag[domOriMagsIdx]
    
    maxMag = np.max(domOriMags)
    magHist, bins = np.histogram(domOriMags, bins = np.arange(maxMag + 2))
    
    return domOriTemp * ori_res, magHist

# This function is used to remove the false movement predictions. We take current and previous value of movement
# and compare them, if absolute difference between these two was quite much, we would ignore the prediction and
# use previous value as our current prediction and if not, we use our current prediction. There is also a counter
# which counts how many time we want to compare, because if difference was quite much in two or more frames, this
# means that child has changed his/her direction

def oriJumpReduct(val, prv_val, cntr):
    if np.abs(val - prv_val) > 90 and np.abs(val - prv_val) < 270:
        if cntr == 0:
            val = prv_val
            cntr += 1
        else:
            cntr = 0
    else:
        cntr = 0
    
    return val, cntr

# This function, converts magnitude and orientation to cartesian arrays

def MagOriToMove(domMag, domOri):
    
    rowMove = domMag * np.sin(np.deg2rad(domOri))
    colMove = domMag * np.cos(np.deg2rad(domOri))
    rowMove = np.int32(rowMove)
    colMove = np.int32(colMove)
    
    return rowMove, colMove

# In this function, we want to compensate our movement. Sometimes, child's movement is bigger than what algorithm
# could predict and therefore we need this kind of compensation.
# First: we subtract two input frames to build a binary image of child's movement
# Second: we take child's window boundaries as our checking criterions
# third: if any of our boundaries has child pixels in it, this means that we nead to compensate our movement in
#        that direction

def MoveComp(frame1_gray, frame2_gray, bbox, domMag):
    
    subtFrames = cv.subtract(frame1_gray, frame2_gray)
    ret, threshold = cv.threshold(subtFrames, 25, 255, cv.THRESH_BINARY)
    
    w, x, y, z = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    w, x, y, z = np.clip(np.array([w, x, y, z]), [0, 0, 0, 0], [frame_l, frame_w, frame_l, frame_w])
    
    up, left, down, right = 0, 0, 0, 0
    if bbox[0] > 0:
        up = threshold[w, x:z]
    if bbox[1] > 0:
        left = threshold[w:y, x]
    if bbox[0] + bbox[2] < 1080:
        down = threshold[y, x:z]
    if bbox[1] + bbox[3] < 1920:
        right = threshold[w:y, z]
    
    if domMag < 40:
        domMag = 40
    
    rowComp, colComp = 0, 0
    
    if np.count_nonzero(up) > 0:
        rowComp -= domMag * 0.75
    if np.count_nonzero(left) > 0:
        colComp -= domMag * 1
    if np.count_nonzero(down) > 0:
        rowComp += domMag * 0.75
    if np.count_nonzero(right) > 0:
        colComp += domMag * 1
    
    # In this part, if both oppisite sides of bbox need compensation, we decide which side is related
    # to child and which side is false. We decide based on which side has more binary pixels in our
    # differenced image
    if np.count_nonzero(right) > 0 and np.count_nonzero(left) > 0:
        if np.count_nonzero(right) > np.count_nonzero(left):
            colComp += domMag * 1
        elif np.count_nonzero(right) < np.count_nonzero(left):
            colComp -= domMag * 1
    
    if bbox[0] > frame_l - bbox_l:
        if np.count_nonzero(threshold[w+10, x:z]) < 10:
            rowComp += 5
    if bbox[1] < 0:
        if np.count_nonzero(threshold[w:y, z-10]) < 10:
            colComp -= 5
    
    rowComp = np.int32(rowComp)
    colComp = np.int32(colComp)
    
    return rowComp, colComp

def bottomSearch(frame2, frame1_gray, frame2_gray, bbox, W):
    correctsVal = []
    correctsIdx = []
    for i in range(np.int16(frame_w/(bbox_w/2)) - 2): # for i in range(9):
        test_patch = frame2[frame_l - bbox_l:frame_l, i*np.int16(bbox_w/2):i*np.int16(bbox_w/2)+bbox_w, :]
        test_patch_1_gray = frame1_gray[frame_l - bbox_l:frame_l, i*np.int16(bbox_w/2):i*np.int16(bbox_w/2) + bbox_w]
        test_patch_2_gray = frame2_gray[frame_l - bbox_l:frame_l, i*np.int16(bbox_w/2):i*np.int16(bbox_w/2) + bbox_w]
        test = np.reshape(test_patch, (1, -1))
        test = np.hstack([test, np.ones((test.shape[0], 1))])
        
        subtFrames = cv.subtract(test_patch_1_gray, test_patch_2_gray)
        ret, threshold = cv.threshold(subtFrames, 25, 255, cv.THRESH_BINARY)
        
        if np.count_nonzero(threshold) > 1000:
            probability = np.max(test.dot(W))
            predict = np.argmax(test.dot(W))
            if predict == 1:
                correctsVal.append(probability)
                correctsIdx.append(i)
        
    if len(correctsIdx) != 0:
        bestCand = np.argmax(correctsVal)
        bbox = [frame_l - np.int16(bbox_l/2), correctsIdx[bestCand]*np.int16(bbox_w/2), bbox_l, bbox_w]
    
    return bbox

def leftSearch(frame2, frame1_gray, frame2_gray, bbox, W):
    correctsVal = []
    correctsIdx = []
    for i in range(np.int16(frame_l/(bbox_l/2)) - 2): # for i in range(9):
        test_patch = frame2[i*np.int16(bbox_l/2):i*np.int16(bbox_l/2)+bbox_l, 0:bbox_w, :]
        test_patch_1_gray = frame1_gray[i*np.int16(bbox_l/2):i*np.int16(bbox_l/2)+bbox_l, 0:bbox_w]
        test_patch_2_gray = frame2_gray[i*np.int16(bbox_l/2):i*np.int16(bbox_l/2)+bbox_l, 0:bbox_w]
        test = np.reshape(test_patch, (1, -1))
        test = np.hstack([test, np.ones((test.shape[0], 1))])
        
        subtFrames = cv.subtract(test_patch_1_gray, test_patch_2_gray)
        ret, threshold = cv.threshold(subtFrames, 25, 255, cv.THRESH_BINARY)
        
        if np.count_nonzero(threshold) > 1000:
            probability = np.max(test.dot(W))
            predict = np.argmax(test.dot(W))
            if predict == 1:
                correctsVal.append(probability)
                correctsIdx.append(i)
    if len(correctsIdx) != 0:
        bestCand = np.argmax(correctsVal)
        bbox = [0 - np.int16(bbox_w/2), correctsIdx[bestCand]*np.int16(bbox_l/2), bbox_l, bbox_w]
    
    return bbox
