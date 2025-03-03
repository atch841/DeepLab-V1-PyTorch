from inference import inference
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from time import strftime, localtime
import cv2
import argparse

from datasets import LiTS_dataset, KiTS_dataset, RandomGenerator, VOCDataset
from nets import vgg
from utils import crf, losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

# batch_size = 70 # 30 for "step", 10 for 'poly'
# lr = 1e-3
weight_decay = 5e-4
# num_max_iters = 20000 # 6000 for "step", 20000 for 'poly'
# num_max_epoch = 200
num_update_iters = 10 # 4000 for "step", 10 for 'poly'
num_save_iters = 1000
num_print_iters = 10
# init_model_path = './data/deeplab_largeFOV.pth'
# log_path = './exp/log.txt'
# model_path_save = './exp/model_last_'
root_dir_path = './VOCdevkit/VOC2012'
pred_dir_path = './exp/labels/'

def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] != 'features.38':
                    # print(m[0], m[1])
                    yield m[1].weight

    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] != 'features.38':
                    yield m[1].bias
    if key == '10x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] == 'features.38':
                    yield m[1].weight
    if key == '20x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] == 'features.38':
                    yield m[1].bias

def train():
    model = vgg.VGG16_LargeFOV(num_classes=2, input_size=256)
    model_dict = model.state_dict()
    state_dict = torch.load(args.init_model_path)
    new_state_dict = {}
    for kk in state_dict.keys():
        k = kk.replace('module.', '')
        if not k in model_dict.keys():
            print('key not found', k)
        elif model_dict[k].shape != state_dict[kk].shape:
            print('size mismatch',k, model_dict[k].shape, state_dict[kk].shape)
        else:
            new_state_dict[k] = state_dict[kk]
    r = model.load_state_dict(new_state_dict, strict=False)
    print(r)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        params = [
            {
                'params': get_params(model, '1x'),
                'lr': args.lr * args.pseudo_lr,
                'weight_decay': weight_decay
            },
            {
                'params': get_params(model, '2x'),
                'lr': args.lr * args.pseudo_lr * 2,
                'weight_decay': 0
            },
            {
                'params': get_params(model, '10x'),
                'lr': args.lr * args.pseudo_lr * 10,
                'weight_decay': weight_decay
            },
            {
                'params': get_params(model, '20x'),
                'lr': args.lr * args.pseudo_lr * 20,
                'weight_decay': 0
            },
        ],
        momentum = 0.9,
    )
    optimizer_1p = torch.optim.SGD(
        params = [
            {
                'params': get_params(model, '1x'),
                'lr': args.lr,
                'weight_decay': weight_decay
            },
            {
                'params': get_params(model, '2x'),
                'lr': args.lr * 2,
                'weight_decay': 0
            },
            {
                'params': get_params(model, '10x'),
                'lr': args.lr * 10,
                'weight_decay': weight_decay
            },
            {
                'params': get_params(model, '20x'),
                'lr': args.lr * 20,
                'weight_decay': 0
            },
        ],
        momentum = 0.9,
    )
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) # for val mIoU = 69.6

    print('Set data...')
    # if args.dataset == '1p':
    dataset_1p = KiTS_dataset('/home/viplab/data/kits_train1_1p_half/', split='train',
                            transform=transforms.Compose([RandomGenerator(output_size=[256, 256])]), 
                            tumor_only=True)
    # elif args.dataset == '100p':
    #     dataset = LiTS_dataset('/home/viplab/data/train5/', split='train',
    #                         transform=transforms.Compose([RandomGenerator(output_size=[256, 256])]), 
    #                         tumor_only=True)
    # else:
    dataset = KiTS_dataset('/home/viplab/data/kits_train1/', split='train',
                            transform=transforms.Compose([RandomGenerator(output_size=[256, 256])]), 
                            tumor_only=True, pseudo=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        # VOCDataset(split='train_aug', crop_size=321, is_scale=False, is_flip=True),
        # VOCDataset(split='train_aug', crop_size=321, is_scale=True, is_flip=True), # for val mIoU = 69.6
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    train_loader_1p = torch.utils.data.DataLoader(
        dataset_1p,
        # VOCDataset(split='train_aug', crop_size=321, is_scale=False, is_flip=True),
        # VOCDataset(split='train_aug', crop_size=321, is_scale=True, is_flip=True), # for val mIoU = 69.6
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    num_max_iters = len(train_loader) * args.num_max_epoch
    num_max_iters_1p = len(train_loader_1p) * args.num_max_epoch

    # Learning rate policy
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    for group in optimizer_1p.param_groups:
        group.setdefault('initial_lr', group['lr'])

    print('Start train...')
    iters = 0
    iters_1p = 0
    log_path = './exp/log_{}.txt'.format(args.name)
    model_path_save = './exp/model_{}_last_'.format(args.name)
    log_file = open(log_path, 'w')
    loss_iters, accuracy_iters = [], []
    loss_iters_1p, accuracy_iters_1p = [], []
    # inference(2, log_file, model, -1)
    for epoch in range(1, 400):
        print('train pseudo')
        iters = train_epoch(train_loader, model, optimizer, loss_iters, accuracy_iters,
                            iters, log_file, num_max_iters, epoch)
        
        print('train 1p')
        iters_1p = train_epoch(train_loader_1p, model, optimizer_1p, loss_iters_1p, accuracy_iters_1p,
                            iters_1p, log_file, num_max_iters_1p, epoch)
        
        if epoch % 5 == 0:
            inference(2, log_file, model, epoch)
            torch.save(model.state_dict(), model_path_save + '_{}.pth'.format(epoch))

def train_epoch(train_loader, model, optimizer, loss_iters, accuracy_iters, iters,
                log_file, num_max_iters, epoch):
    for iter_id, batch in enumerate(train_loader):
        model.train()
        loss_seg, accuracy = losses.build_metrics(model, batch, device)
        optimizer.zero_grad()
        loss_seg.backward()
        optimizer.step()

        loss_iters.append(float(loss_seg.cpu()))
        accuracy_iters.append(float(accuracy))

        iters += 1
        if iters % num_print_iters == 0:
            cur_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
            log_str = 'epoch: {}/{} iters:{:4}/{:4}, loss:{:6,.4f}, accuracy:{:5,.4}'.format(epoch, args.num_max_epoch, iters, num_max_iters, np.mean(loss_iters), np.mean(accuracy_iters))
            print(log_str)
            log_file.write(cur_time + ' ' + log_str + '\n')
            log_file.flush()
            loss_iters = []
            accuracy_iters = []
        
        # if iters % num_save_iters == 0:
        #     torch.save(model.state_dict(), model_path_save + str(iters) + '_{}.pth'.format(epoch))
        
        # step
        # if iters == num_update_iters or iters == num_update_iters + 1000:
        #     for group in optimizer.param_groups:
        #         group["lr"] *= 0.1
        
        # poly
        for group in optimizer.param_groups:
            group["lr"] = group['initial_lr'] * (1 - float(iters) / num_max_iters) ** 0.9

        # if iters == num_max_iters:
        #     exit()
    return iters

def test(model_path_test, use_crf):
    batch_size = 2
    is_post_process = use_crf
    crop_size = 513
    model = vgg.VGG16_LargeFOV(input_size=crop_size, split='test')
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(model_path_test))
    model.eval()
    model = model.to(device)
    val_loader = torch.utils.data.DataLoader(
        VOCDataset(split='val', crop_size=crop_size, label_dir_path='SegmentationClassAug', is_scale=False, is_flip=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # DenseCRF
    post_processor = crf.DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=140,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=5,         # 4, 5
    )

    img_dir_path = root_dir_path + '/JPEGImages/'
    # class palette for test
    palette = []
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21] = np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8').flatten()
    times = 0.0
    index = 0
    loss_iters, accuracy_iters = [], []
    CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)
    for iter_id, batch in enumerate(val_loader):
        image_ids, images, labels = batch
        images = images.to(device)
        labels = losses.resize_labels(labels, size=(crop_size, crop_size)).to(device)
        logits = model(images)
        probs = nn.functional.softmax(logits, dim=1) # shape = [batch_size, C, H, W]

        outputs = torch.argmax(probs, dim=1) # shape = [batch_size, H, W]
        
        loss_seg = CEL(logits, labels)
        accuracy = float(torch.eq(outputs, labels).sum().cpu()) / (len(image_ids) * logits.shape[2] * logits.shape[3])
        loss_iters.append(float(loss_seg.cpu()))
        accuracy_iters.append(float(accuracy))

        for i in range(len(image_ids)):
            if is_post_process:
                raw_image = cv2.imread(img_dir_path + image_ids[i] + '.jpg', cv2.IMREAD_COLOR) # shape = [H, W, 3]
                h, w = raw_image.shape[:2]
                pad_h = max(513 - h, 0)
                pad_w = max(513 - w, 0)
                pad_kwargs = {
                    "top": 0,
                    "bottom": pad_h,
                    "left": 0,
                    "right": pad_w,
                    "borderType": cv2.BORDER_CONSTANT,
                }
                raw_image = cv2.copyMakeBorder(raw_image, value=[0, 0, 0], **pad_kwargs)
                raw_image = raw_image.astype(np.uint8)
                start_time = time.time()
                prob = post_processor(raw_image, probs[i].detach().cpu().numpy())
                times += time.time() - start_time
                output = np.argmax(prob, axis=0).astype(np.uint8)
                img_label = Image.fromarray(output)
            else:
                output = np.array(outputs[i].cpu(), dtype=np.uint8)
                img_label = Image.fromarray(output)
            img_label.putpalette(palette)
            img_label.save(pred_dir_path + image_ids[i] + '.png')

            accuracy = float(torch.eq(outputs[i], labels[i]).sum().cpu()) / (logits.shape[2] * logits.shape[3])
            index += 1
            if index % 200 == 0:
                print(image_ids[i], float('%.4f' % accuracy), index)
    if is_post_process:
        print('dense crf time = %s' % (times / index))
    print('val loss = %s, acc = %s' % (np.mean(loss_iters), np.mean(accuracy_iters)))
    print(model_path_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='log file name')
    parser.add_argument('--type', default='train', help='train or test model')
    parser.add_argument('--model_path_test', default='./exp/model_last_20000.pth', help='test model path')
    parser.add_argument('--use_crf', default=False, action='store_true', help='use crf or not')
    parser.add_argument('--dataset', default='pseudo', type=str, help='dataset')
    parser.add_argument('--init_model_path', default='./data/deeplab_largeFOV.pth', type=str, help='load pretrain')
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pseudo_lr', type=float, default=0.01)
    parser.add_argument('--num_max_epoch', type=int, default=400)
    args = parser.parse_args()

    if args.type == 'train':
        train()
    else:
        test(args.model_path_test, args.use_crf)

