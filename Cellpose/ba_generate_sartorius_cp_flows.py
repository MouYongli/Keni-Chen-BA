import sys
sys.path.append("./cellpose")
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import cv2
from cellpose.dynamics import labels_to_flows
from cellpose.io import imsave, imread

def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
fix_all_seeds(2021)

def build_mask(annotations, input_shape):
    '''
    annotations: list of run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 to N - mask, 0 - background
    '''
    (height, width) = input_shape
    mask = np.zeros((height*width), dtype=np.uint8)
    for instance_idx in range(len(annotations)):
        s = annotations[instance_idx].split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = instance_idx + 1
    return mask.reshape(input_shape)

BASE_PATH = "../Data/Kaggle"
IMAGE_PATH = osp.join(BASE_PATH, "train")
CSV_PATH = osp.join(BASE_PATH, "train.csv")
OUTPUT_PATH = "../Data/Cellpose/Sartorius"

if __name__ == '__main__':
    df_data = pd.read_csv(CSV_PATH)
    gb_data = df_data.groupby('id')
    all_image_ids = np.array(df_data["id"].unique())
    iperm = np.random.permutation(len(all_image_ids))
    num_train_samples = int(len(all_image_ids) * 0.7)
    num_val_samples = int(len(all_image_ids) * 0.1)
    num_test_samples = len(all_image_ids) - num_train_samples - num_val_samples
    train_image_ids = all_image_ids[iperm[:num_train_samples]]
    val_image_ids = all_image_ids[iperm[num_train_samples:num_train_samples + num_val_samples]]
    test_image_ids = all_image_ids[iperm[num_train_samples + num_val_samples:]]

    df_data[df_data.id.isin(train_image_ids)].to_csv(osp.join(OUTPUT_PATH, "train", "train.csv"), index=False)
    df_data[df_data.id.isin(val_image_ids)].to_csv(osp.join(OUTPUT_PATH, "val", "val.csv"), index=False)
    df_data[df_data.id.isin(test_image_ids)].to_csv(osp.join(OUTPUT_PATH, "test", "test.csv"), index=False)

    i = 1
    for idx in range(num_train_samples):
        image_id = train_image_ids[idx]
        df_data_idx = gb_data.get_group(image_id)
        annotations = df_data_idx['annotation'].tolist()
        image = cv2.imread(os.path.join(IMAGE_PATH, image_id + ".png"))
        mask = build_mask(annotations, (image.shape[0], image.shape[1]))
        flows = labels_to_flows([mask])
        imsave(osp.join(OUTPUT_PATH, "train", "{}.tif".format(image_id)), np.expand_dims(image[:,:,0], axis=2))
        imsave(osp.join(OUTPUT_PATH, "train", "{}_mask.tif".format(image_id)), flows[0])
        i = i + 1
        print(">>> train {}/{} image {} finished".format(i, num_train_samples, image_id))
    i = 0
    for idx in range(num_val_samples):
        image_id = val_image_ids[idx]
        df_data_idx = gb_data.get_group(image_id)
        annotations = df_data_idx['annotation'].tolist()
        image = cv2.imread(os.path.join(IMAGE_PATH, image_id + ".png"))
        mask = build_mask(annotations, (image.shape[0], image.shape[1]))
        flows = labels_to_flows([mask])
        imsave(osp.join(OUTPUT_PATH, "val", "{}.tif".format(image_id)), np.expand_dims(image[:,:,0], axis=2))
        imsave(osp.join(OUTPUT_PATH, "val", "{}_mask.tif".format(image_id)), flows[0])
        i = i + 1
        print(">>> val {}/{} image {} finished".format(i, num_val_samples, image_id))
    i = 0
    for idx in range(num_test_samples):
        image_id = test_image_ids[idx]
        df_data_idx = gb_data.get_group(image_id)
        annotations = df_data_idx['annotation'].tolist()
        image = cv2.imread(os.path.join(IMAGE_PATH, image_id + ".png"))
        mask = build_mask(annotations, (image.shape[0], image.shape[1]))
        flows = labels_to_flows([mask])
        imsave(osp.join(OUTPUT_PATH, "test", "{}.tif".format(image_id)), np.expand_dims(image[:,:,0], axis=2))
        imsave(osp.join(OUTPUT_PATH, "test", "{}_mask.tif".format(image_id)), flows[0])
        i = i + 1
        print(">>> test {}/{} image {} finished".format(i,  num_test_samples, image_id))
