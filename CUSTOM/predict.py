import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "WEIGHT")

# Garantindo que a versão do modelo usado será a localizada na pasta "mrcnn"
sys.path.append(ROOT_DIR)  

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
 
import custom
 
config = custom.CustomConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset")

 
# Modificações para fazer a predição

class InferenceConfig(config.__class__):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
config.display()

#Qual dispositivo deve ser usado na predição

DEVICE = "/cpu:0" #"/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"
 
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Carregando informações do dataset

dataset = custom.CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
 
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)
 

weights_path = os.path.join(ROOT_DIR, "WEIGHT\CORROSION.h5")

# Carregando o peso previamente treinado com as imagens de corrosão
print("Carregando modelo treinado: ", weights_path)
model.load_weights(weights_path, by_name=True)


# Inferência com imagem selecionada começa aqui

import matplotlib.image as mpimg
image1 = mpimg.imread('Corrosao.jpg')
print(len([image1]))
results1 = model.detect([image1], verbose=1)
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
                            dataset.class_names, scores=None, ax=ax,
                            title="Detecção de Corrosão",show_mask=True,show_bbox=False,captions=None)
plt.show()