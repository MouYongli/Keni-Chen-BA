import os
import os.path as osp
import numpy as np
import pandas as pd
from cellpose.models import CellposeModel, Cellpose
from cellpose.io import load_train_test_data

BASE_PATH = "../Data/Cellpose/Sartorius"
EXP_PATH = "../Experiments"
MODEL = "cellpose"
DATASET = "sartorius"

if __name__ == '__main__':
    train_dir = osp.join(BASE_PATH, "train")
    test_dir = osp.join(BASE_PATH, "val")
    output = load_train_test_data(train_dir, test_dir=test_dir, mask_filter='_mask')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    # print("Total number of train image is {}".format(len(images)))
    # print("Total number of train image is {}".format(len(test_images)))
    # print("Example image name", image_names[0])
    # print("Training image shape", images[0].transpose(2,0,1).shape)
    # print("Training label shape", labels[0][1:3,:,:].shape)

    images = [np.concatenate((images[i], np.zeros(images[i].shape)), axis=2).transpose(2, 0, 1) for i in range(len(images))]
    labels = [labels[i][0].astype(int) for i in range(len(labels))]
    test_images = [np.concatenate((test_images[i], np.zeros(test_images[i].shape)), axis=2).transpose(2, 0, 1) for i in range(len(test_images))]
    test_labels = [test_labels[i][0].astype(int) for i in range(len(test_labels))]
    model = CellposeModel(gpu=True, pretrained_model=None,  net_avg=False, #model_type="nuclei",
                          diam_mean=17., device=None, residual_on=True, style_on=True, concatenation=False, nchan=2)
    save_path = osp.join(EXP_PATH, "{}_{}".format(MODEL, DATASET))
    cpmodel_path = model.train(images, labels, train_files=None,
                               test_data=test_images, test_labels=test_labels,
                               test_files=None, channels=None, normalize=True, save_path=save_path, save_every=1, save_each=True,
                               learning_rate=0.2, n_epochs=100, momentum=0.9, SGD=True,
                               weight_decay=0.00001, batch_size=8, nimg_per_epoch=None,
                               rescale=True, min_train_masks=5, model_name=None)