import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose.models import CellposeModel
from cellpose.io import load_train_test_data
from cellpose.metrics import average_precision

BASE_PATH = "../Data/Cellpose/Sartorius"
train_dir = osp.join(BASE_PATH, "train")
val_dir = osp.join(BASE_PATH, "val")
test_dir = osp.join(BASE_PATH, "test")

EXP_PATH = "../Experiments"
MODEL = "cellpose"
DATASET = "sartorius"

model_dir = osp.join(osp.join(EXP_PATH, "{}_{}".format(MODEL, DATASET)), "models")

model_file_list = [f for f in os.listdir(model_dir) if f.startswith("cellpose")]
output = load_train_test_data(val_dir, test_dir=test_dir, mask_filter='_mask')
val_images, val_labels, image_names_val, test_images, test_labels, image_names_test = output

val_images = [np.concatenate((val_images[i], np.zeros(val_images[i].shape)), axis=2) for i in range(len(val_images))]
test_images = [np.concatenate((test_images[i], np.zeros(test_images[i].shape)), axis=2) for i in range(len(test_images))]

model_idx = 0
model_path = osp.join(model_dir, model_file_list[model_idx])
model = CellposeModel(gpu=True, pretrained_model=model_path,  net_avg=False,
                          diam_mean=17., device=None, residual_on=True, style_on=True, concatenation=False, nchan=2)
masks, flows, styles = model.eval(val_images, batch_size=8, diameter=17., channels=[0,0], net_avg=False)

print(len(masks))
print(masks[0].shape)
# model_idx = 0
# model_path = osp.join(model_dir, model_file_list[model_idx])
# model = CellposeModel(gpu=True, pretrained_model=model_path,  net_avg=False,
#                       diam_mean=17., device=None, residual_on=True, style_on=True, concatenation=False, nchan=2)
# masks, flows, styles =  model.eval(val_images, batch_size=8, channels=[0, 0], net_avg=False)
