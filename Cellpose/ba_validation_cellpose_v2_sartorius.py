import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose_v2.models import CellposeModel
from cellpose_v2.io import load_train_test_data
from cellpose_v2.metrics import average_precision
from cellpose_v2.dynamics import compute_masks

BASE_PATH = "../Data/Cellpose/Sartorius"
train_dir = osp.join(BASE_PATH, "train")
val_dir = osp.join(BASE_PATH, "val")
test_dir = osp.join(BASE_PATH, "test")

EXP_PATH = "../Experiments"
MODEL = "cellpose_v2"
DATASET = "sartorius"

model_dir = osp.join(osp.join(EXP_PATH, "{}_{}".format(MODEL, DATASET)), "models")
model_file_list = [f for f in os.listdir(model_dir) if f.startswith("cellpose")]
epoch_list = [f.split("_")[-1] for f in os.listdir(model_dir) if f.startswith("cellpose")]
gb_val_csv = pd.read_csv(osp.join(val_dir, "val.csv")).groupby("id")
gb_test_csv = pd.read_csv(osp.join(test_dir, "test.csv")).groupby("id")

output = load_train_test_data(val_dir, test_dir=test_dir, mask_filter='_mask')
val_images, val_labels, image_names_val, test_images, test_labels, image_names_test = output

val_images = [np.concatenate((val_images[i], np.zeros(val_images[i].shape)), axis=2) for i in range(len(val_images))]
test_images = [np.concatenate((test_images[i], np.zeros(test_images[i].shape)), axis=2) for i in range(len(test_images))]
val_labels = [val_labels[i][0].astype(int) for i in range(len(val_labels))]
test_labels = [test_labels[i][0].astype(int) for i in range(len(test_labels))]

val_image_ids = [f.split("/")[-1].split(".")[0] for f in image_names_val]
test_image_ids = [f.split("/")[-1].split(".")[0] for f in image_names_test]
val_cell_type_list = [gb_val_csv.get_group(image_id).reset_index().loc[0, "cell_type"] for image_id in val_image_ids]
test_cell_type_list = [gb_test_csv.get_group(image_id).reset_index().loc[0, "cell_type"] for image_id in test_image_ids]

model_idx = 7
print(f">>> Epoch {epoch_list[model_idx]}")
model_path = osp.join(model_dir, model_file_list[model_idx])
model = CellposeModel(gpu=True, pretrained_model=model_path,  net_avg=False, diam_mean=17., device=None, residual_on=True, style_on=True, concatenation=False, nchan=2)
print(f"Loading model {model_path}")
[dPs, cellprobs] =  model.eval(val_images, batch_size=8, diameter=17., channels=[0,0], net_avg=False)

print("input class", dPs[0].shape)
print("input prob", cellprobs[0].shape)

compute_masks(np.argmax(dPs[0][:,0], axis=0), cellprobs[0])


# mean_ap = average_precision(val_labels, val_masks, threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
# print(f"Val Mean AP {np.mean(mean_ap[0])}")