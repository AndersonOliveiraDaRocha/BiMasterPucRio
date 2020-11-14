import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Habilitando o Tensorflow para operar na GPU
config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# Fixando a pasta raiz do projeto
ROOT_DIR = os.path.abspath("../")
 
# Garantindo que a versão do modelo usado será a localizada na pasta "mrcnn"
sys.path.append(ROOT_DIR)  
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils
 
# Caminho para a rede pré-treinada MS COCO
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "WEIGHT/COCO_WEIGHT.h5")
 
# Caminho para salvar os pesos a cada EPOCA
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "WEIGHT")
 
# Configurações gerais  
class CustomConfig(Config):       
    NAME = "object"    
    IMAGES_PER_GPU = 1 
    # Número de classes que o seu modelo poderá reconhecer, deixando sempre uma para o background
    NUM_CLASSES = 1 + 1  
 
    # Número de passos por época
    STEPS_PER_EPOCH = 20
 
    # Escapar de detecções com grau de confiança menor que: 0.8 representa confiança < 80%
    DETECTION_MIN_CONFIDENCE = 0.8
 

### Dataset 

class CustomDataset(utils.Dataset):
 
    def load_custom(self, dataset_dir, subset):
        # dataset_dir: caminho para o dataset
        # subset: caminho para o dataset de treino e validação
        
        # Indica quais são as classes que serão consideradas, nesse caso todas as áreas anotadas como corrosão
        # são indicadas como "sim" (tipo de dados texto mesmo)
        
        self.add_class("object", 1,"sim")
                
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
 
        # As anotações foram feitas no VGG, abaixo o código carrega as anotações
        annotations1 = json.load(open(os.path.join(dataset_dir, "DATASET.json")))        
        annotations = list(annotations1.values())          
        annotations = [a for a in annotations if a['regions']]            
 
        for a in annotations:
            # Convertendo as anotações poligonais do VGG            
            polygons = [r['shape_attributes'] for r in a['regions']]
            # Importante destacar o nome dado ao atributo da região anotada, nesse caso foi "corrosao"
            objects = [s['region_attributes']['corrosao'] for s in a['regions']]
            print("objects:",objects)
            # Indicando qual é o dicionário que representa uma ocorrência de corrosão
            name_dict = {"sim":1}                              
            num_ids = [name_dict[a] for a in objects]
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2] 
            self.add_image("object",image_id=a['filename'],path=image_path,width=width, height=height,polygons=polygons,num_ids=num_ids)
 
    def load_mask(self, image_id):
        # Função para gerar uma máscara para a imagem    
        
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
  
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
 
    def image_reference(self, image_id):
        # Função para retornar o caminho da imagem
        
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

 
def train(model):
    # Função para treino do modelo
    
    # Dataset de treino
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()
 
    # Dataset de validação
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()
    
    # Invocando a função de treino do modelo
    # Parâmetros:
    # Learning Rate configurado pelo arquivo config.py onde LEARNING_RATE = 0.001
    # epochs (épocas) = 20
    # layers (camadas), quais camadas da rede serão retreinadas, reconfigurado para treinar HEADS somente (minha GPU não suportou um treino     # mais profundo
 
    
    print("Treinando a rede com todas as camadas")
    model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,epochs=20,layers='heads') 

 
if __name__ == '__main__':
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command", metavar="<command>",help="'train'")
    parser.add_argument('--dataset', required=False,metavar="/path/to/custom/dataset/",help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,metavar="/path/to/weights.h5",help="Path to weights .h5 file or 'coco'")    
    parser.add_argument('--logs', required=False,default=DEFAULT_LOGS_DIR,metavar="/path/to/logs/",help='Logs and checkpoints directory')
    args = parser.parse_args()
   
    if args.command != "train":
        print("Execute o comando com o argumento 'train'. '{}' não é um argumento válido".format(args.command))
    else:         
        if args.command == "train":
            assert args.dataset, "Argumento --dataset é necessário para treino"    
 
        print("PESOS: ", args.weights)
        print("DATASET: ", args.dataset)
        print("LOGS: ", args.logs)
 
        # Configurações
        if args.command == "train":
            config = CustomConfig()   
            config.display()
 
        # Criando o modelo
        if args.command == "train":
            model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs) 
         
        # Carregando pesos
        print("Carregando pesos . . . ", COCO_WEIGHTS_PATH)
        if args.weights.lower() == "coco":            
            # Excluindo as últimas camadas pois essas camadas exigem um certo número de classes oriundos do COCO
            model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
         
        # Chamando o treino do modelo
        if args.command == "train":
            train(model)    
        