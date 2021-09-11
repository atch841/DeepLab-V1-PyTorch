import torch
import torch.nn as nn
from PIL import Image
import numpy as np


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def build_metrics(model, batch, device):
    CEL = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 287.4423]), ignore_index=255).to(device)

    images, labels = batch['image'], batch['label']
    # labels = resize_labels(labels, size=(41, 41)).to(device)
    logits = model(images.to(device))

    loss_seg = CEL(logits, labels.to(device))

    preds = torch.argmax(logits, dim=1)
    accuracy = float(torch.eq(preds, labels.to(device)).sum().cpu()) / (len(images) * logits.shape[2] * logits.shape[3])

    return loss_seg, accuracy
                
