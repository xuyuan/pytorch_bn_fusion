import torch
from torch.nn import functional as F
import torchvision.models as models
from torchvision import transforms

import argparse
from PIL import Image
import numpy as np
from bn_fusion import fuse_bn_recursively
from utils import convert_resnet_family
import pretrainedmodels
import time

def create_model(model):
    try:
        net = getattr(models, model)(pretrained=True)
    except:
        net = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg16_bn')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cudnn-benchmark', action='store_true')
    args = parser.parse_args()

    trf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           ])

    img = trf(Image.open('dog.jpg')).unsqueeze(0)
    if args.cuda:
        img = img.cuda()
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    

    N = 100
    #if args.cuda:
    #    N = 1000

    with torch.no_grad():
        # Benchmarking
        # First, we run the network the way it is
        net = create_model(args.model)
        net.eval()
        net = net.float()
        if args.cuda:
            net = net.cuda()
            for i in range(N):
                res_0 = F.softmax(net(img), 1)
        # Measuring non-optimized model performance
        #print(net)
        start = time.time()
        for i in range(N):
            res_0 = F.softmax(net(img), 1)
        time0 = time.time() - start
        
        
        if 'resnet' in args.model:
            se = True if 'se' in args.model else False
            net = create_model(args.model)
            net = convert_resnet_family(net, se)
            net.eval()
            net = net.float()
            if args.cuda:
                net = net.cpu()
                net = net.cuda()
                for i in range(N):
                    res_0 = F.softmax(net(img), 1)
            # Measuring non-optimized model performance
            #print(net)
            start = time.time()
            for i in range(N):
                res_01 = F.softmax(net(img), 1)
            time01 = time.time() - start
        
        net = create_model(args.model)
        if 'resnet' in args.model:
            se = True if 'se' in args.model else False
            net = convert_resnet_family(net, se)
        net = fuse_bn_recursively(net)
        net.eval()
        net = net.float()
        if args.cuda:
            net = net.cpu()
            net = net.cuda()
            for i in range(N):
                res_1 = F.softmax(net(img), 1)
        #print(net)
        start = time.time()
        for i in range(N):
            res_1 = F.softmax(net(img), 1)
        time1 = time.time() - start
        

        diff = res_0 - res_1
        print('L2 Norm of the element-wise difference:', diff.norm().item())
        print('Non fused takes',  time0, 'seconds')
        if 'resnet' in args.model:
            print('Non fused, modified takes',  time01, 'seconds')
        print('Fused takes', time1, 'seconds')
        print('improved', time0 - time1, '(%.2f%%)' % ((time0 - time1) / time0 * 100))

