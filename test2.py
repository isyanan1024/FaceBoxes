from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
# from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
import time
import copy

parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='/home/yana/FaceBoxes/weights_bak/FaceBoxes_epoch_295.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=100, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=10, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=3)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    # for i, img_name in enumerate(test_dataset):
    image_path = '/home/yana/FaceBoxes/test/'
    images = os.listdir(image_path)
    for image in images:
        impPath = os.path.join(image_path, image)
        img_raw = cv2.imread(impPath, cv2.IMREAD_COLOR)
        img_raw = cv2.resize(img_raw, (960, 960))
        begin = time.time()
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        # print(im_height, im_width)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf = net(img)  # forward pass
        conf = conf.view(conf.shape[1], conf.shape[2])
        index = conf.argmax(dim = 1)

        background_scores = conf.squeeze(0).data.cpu().numpy()[:, 0]
        background_scores = background_scores[np.newaxis,:]

        face_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        face_mask_scores = conf.squeeze(0).data.cpu().numpy()[:, 2]
        target = np.maximum(face_scores,face_mask_scores)
        target = target[np.newaxis,:]

        conf = torch.tensor(np.concatenate((background_scores, target)).T)

        resume = time.time()-begin

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()

        # mask  
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        index = index[inds]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        index = index.cpu().numpy()[order]
        boxes = boxes[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis], index[:, np.newaxis])).astype(np.float32, copy=False)
        # print(dets)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        print('FPS: ', 1 / (time.time() - begin))
        
        for det in dets:
            index = det[5]
            if index == 1:
                # f.write('face ' + str(round(det[4],3)) + ' ' + str(det[0]) + ' ' + str(det[1]) + ' ' + str(det[2]) + ' ' + str(det[3]) + '\n')
                cv2.rectangle(img_raw, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)
                cv2.putText(img_raw, 'No_mask' + str(round(det[4], 2)), (det[0], det[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            else:
                # f.write('face_mask '+ str(round(det[4],3)) + ' ' + str(det[0]) + ' ' + str(det[1]) + ' ' + str(det[2]) + ' ' + str(det[3]) + '\n')
                cv2.rectangle(img_raw, (det[0], det[1]), (det[2], det[3]), (0, 255, 0), 2)
                cv2.putText(img_raw, 'Mask' + str(round(det[4], 2)), (det[0], det[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(image, img_raw)
