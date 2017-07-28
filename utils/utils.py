# code based on:
#YAD2K https://github.com/allanzelener/YAD2K
#darkflow https://github.com/thtrieu/darkflow
#Darknet.keras https://github.com/sunshineatnoon/Darknet.keras
import numpy as np
import cv2

def load_weights(model,yolo_weight_file):
                
    data = np.fromfile(yolo_weight_file,np.float32)
    data=data[4:]
    
    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape,bshape = shape
            bia = data[index:index+np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index+np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker,bia])


class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);


def yolo_net_out_to_car_boxes(net_out, threshold = 0.2, sqrt=1.8,C=20, B=2, S=7):
    class_num = 6
    boxes = []
    SS        =  S * S # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell
    
    probs = net_out[0 : prob_size]
    confs = net_out[prob_size : (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size) : ]
    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])
    
    for grid in range(SS):
        for b in range(B):
            bx   = Box()
            bx.c =  confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid %  S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w =  cords[grid, b, 2] ** sqrt 
            bx.h =  cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c
            
            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                boxes.append(bx)
                
    # combine boxes that are overlap
    boxes.sort(key=lambda b:b.prob,reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]
    
    return boxes

def overlayWarped(imgcv, warped):
    width, height = imgcv.shape[0], imgcv.shape[1];
    resx, resy = int(0.2*width), int(0.2*height);
    res = cv2.resize(warped,(resy, resx), interpolation = cv2.INTER_CUBIC);

    # overlay warped
    imgcv[ 20: 20 + resx, imgcv.shape[1]-resy-20: imgcv.shape[1]-20, :] = res;
    cv2.rectangle(imgcv, (imgcv.shape[1]-resy-20, 20), (imgcv.shape[1]-20, 20 + resx), (0,0,255), 3); # Blue
    cv2.putText(imgcv,'Top View', (int((imgcv.shape[1] + imgcv.shape[1]-resy)/2)-50, 15 + resx), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)


def draw_box(boxes,im,crop_dim,M):
    imgcv = im
    [xmin,xmax] = crop_dim[0]
    [ymin,ymax] = crop_dim[1]

    # do the warp and get the matrix to get new points

    ym_per_pix = 3*3.7/1000 # meters per pixel in y dimension
    xm_per_pix = 3*3.7/1000 # meters per pixel in x dimension

    conv = np.array([xm_per_pix, ym_per_pix]);

    for b in boxes:
        h, w, _ = imgcv.shape
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        left = int(left*(xmax-xmin)/w + xmin)
        right = int(right*(xmax-xmin)/w + xmin)
        top = int(top*(ymax-ymin)/h + ymin)
        bot = int(bot*(ymax-ymin)/h + ymin)

        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        # thick = int((h + w) // 150);
        pos = np.array([int((left+right)/2.), int(bot), 1]).reshape([-1,1]);
        pos_car = np.array([h, w/2.0, 1]).reshape([-1,1]);

        # print('Car Original:', pos, 'Cam:', pos_car);

        pos = np.dot(M, pos)[:2];
        pos_car = np.dot(M, pos_car)[:2];

        # print('Car Conv:', pos, 'Cam:', pos_car);

        dist = pos-pos_car;
        # print('Dist', dist);
        dist = np.linalg.norm(np.multiply(dist[::-1], conv));

        cv2.putText(imgcv,'Distance: %.0fft' % dist, (left, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)

        if dist < 10:
            cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), 3); # Red
            cv2.putText(imgcv,'Warning', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        else:
            cv2.rectangle(imgcv, (left, top), (right, bot), (0,255,0), 3); # Green

        warped_shape = (4000, 2000); #(1212, 628);

        # print(warped_shape);

        warped = cv2.warpPerspective(imgcv, M, warped_shape, flags=cv2.INTER_LINEAR)[0:1000, :, :];

        cv2.line(warped, (int(pos[0]), int(pos[1])), (int(pos_car[0]), int(pos_car[1])), (255, 0, 0), thickness=10);
        # cv2.circle(warped, (int(pos[1]), int(pos[0])), 100, (255, 0, 0), thickness = 20);
        # print((int(pos[1]), int(pos[0])));
        cv2.line(warped, (100, 100), (int(pos_car[0]), int(pos_car[1])), (255, 0, 0), thickness=10);
        
        overlayWarped(imgcv, warped);

    return imgcv #warped #imgcv
