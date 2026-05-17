# Sonia Artificial Intelligence

## Tuto utilisation AI

### Export des bags

 - Avoir une connection internet
 - Dans *ai-framework/config/credentials.py*, avoir mis son API key pour labelbox, exemple : ```API_KEY = "valeur_de_l_api_key_labelbox"```
 - Si il n'existe pas, créer le dossier *ai-framework/models/*
   - Si tu as des models, les mettre dans ce dossier, ca donnerait : *ai-framework/models/nom_du_model/model.onnx* et *ai-framework/models/nom_du_model/data.yaml* 
 - Si il n'existe pas, créer le dossier *ai-framework/rosbags/*
 - Mettre les bags dans le dossier *ai-framework/rosbags/*
   - Ca donne : *ai-framework/rosbags/nom_du_bag/nom_du_bag_0.db3* et *ai-framework/rosbags/nom_du_bag/metadata.yaml* 
 - Rouler ```python3 main.py --task export-data```
 - Dans l'interface qui s'ouvre :
   - Sélectionner les bags que tu veux (n'en met pas trop d'un coup c'est gourmand en RAM),
   - Laisser tous les topics sélectionnés (sauf si tu veux en retirer certains),
   - Si tu as des modèles tu peux les sélectionner mais c'est pas nécessaire,
   - Sélectionner le projet sur Labelbox dans lequel envoyer les images, ATTENTION NE PAS RAJOUTER DES IMAGES DE TEST AUX PROJETS DE COMPÉ -> DANS LE DOUTE CRÉER UN NOUVEAU PROJET
   - Sélectionner le dataset sur Labelbox dans lequel envoyer les images, ATTENTION NE PAS RAJOUTER DES IMAGES DE TEST AUX DATASETS DE COMPÉ -> DANS LE DOUTE CRÉER UN NOUVEAU DATASET
   - L'ontologie permet de sélectionner les labels que tu utiliseras pour annoter,
   - Pour le coefficient, met ce que tu veux c'est la proportion d'images que tu vas préselectionner le bag fait 20 images par secondes,
   - Enfin : faire launch.

 - Sélection des images:
   - Touche "y" pour sélectionner l'image et passer au suivant,
   - Touche "n" pour ne pas sélectionner et passer au suivant,
   - Touche "p" revenir au précédent et le dé-selectionner,
   - Touche "q" pour quitter la sélection avant d'avoir traité toutes les images.

 - Attendre que l'export se fasse, ne pas couper la connection réseau.

### Lancement d'un entrainement

 - Avoir une connection internet
 - Choisir les projet sur labelbox que tu veux utiliser
 - Rouler ```python3 main.py --task train --labelbox-projects nom_projet_1 nom_projet_2 nom_projet_3```
 - Attendre que ça roule

---

## Getting Started

These instructions will show you how to set up and train our AUV vision model.

### Prerequisites

Require `Python>=3.8` with `PyTorch>=1.8`.

### Installing

Installation of needed packages:

With pip:

```pip install -r requirements.txt```

With conda:

```conda install conda-forge::ultralytics```

## Datasets

### Data architecture

To add a new dataset, create the following folders :
- `./datasets/<new dataset>/`
  - `./datasets/<new dataset>/train/`
    - `./datasets/<new dataset>/train/images/`
    - `./datasets/<new dataset>/train/labels/`
  - `./datasets/<new dataset>/val/`
    - `./datasets/<new dataset>/val/images/`
    - `./datasets/<new dataset>/val/labels/`
  - `./datasets/<new dataset>/test/`
    - `./datasets/<new dataset>/test/labels/`
    - `./datasets/<new dataset>/test/images/`

Then, add each corresponding images and labels in their folders. Correponding images and labels must have the same name (for instance: `images/img_1.png` and `labels/img_1.txt`).

Finally, you need to make the configuration file `./datasets/<new dataset>/data.yaml`. This file contains :

```yaml
path: ../datasets/<new dataset>
train: train/images
val: val/images
test: test/images

# Class number
nc: 5

# Classes names
names: 
  0: classe_1
  1: classe_2
  2: classe_3
  3: classe_4
  4: classe_5
```

### Labelling
To labellise our images, we use https://labelbox.com. To make a good labelling that enable to train a good model, you have to remember to:
 - Make bounding boxes neither larger nor smaller than the object.
   - <figure>
      <figcaption>What to do:</figcaption><br/>
      <img src=https://github.com/sonia-auv/ai-framework/blob/test/images/case_1_good.png height="150">
     </figure>
   - <figure>
      <figcaption>What not to do:</figcaption><br/>
      <img src=https://github.com/sonia-auv/ai-framework/blob/test/images/case_2_bad.png height="150"><img src=https://github.com/sonia-auv/ai-framework/blob/test/images/case_1_bad.png height="150">
     </figure>
 - Even if the orientation of the object is not perfect, the box must contain the whole object. 
   - <figure>
      <figcaption>What to do:</figcaption><br/>
      <img src=https://github.com/sonia-auv/ai-framework/blob/test/images/case_2_good.png height="150">
     </figure>

## Run

### Training
Instructions about how to train a model.

The following command will start a fine tuning of the selected model, using chosen parameters and the given dataset.

```powershell
python3 main.py --task train --model='yolo' --load-model='models/yolov8n.pt' --dataset='data_example' --name='my_model' --augment
```
 - `--model` **mandatory**, set the chosen model (choices: ['yolov8', 'yolov9', 'yolov10']),
 - `--load-model` **mandatory**, path to the model weights (for example: models/yolov8n.pt),
 - `--dataset` **mandatory**, name of the selected dataset (every dataset are in the folder **./datasets**, example: 'data_test'),
 - `--project` **optionnal**, set the name of the project (default:None),
 - `--name` **optionnal**, set the name of the model (defaut: \<name of the selected model\>),
 - `--augment` **optionnal**, if used, an augmentation is done on the training data, if not no augmentation (default: False),
 - `--resume` **optionnal**, use this if you are continuing a previous training (put only *--resume*).

There is more parameters in our configuration files, you can modify the following files :
 - `config/training_parameters.yaml`
 ```yaml
 # Ultralytics YOLO 🚀, AGPL-3.0 license.
 # Sonia-AI training settings.
 
 # Main settings  -------------------------------------------------------------------------------------------------------
 # Do not change these parameters (or the model won't  work and won't learn).
 task: segment # (str) YOLO task, i.e. detect, segment, classify, pose.
 mode: train # (str) YOLO mode, i.e. train, val,  predict, export, track, benchmark.
 
 # Train settings  -------------------------------------------------------------------------------------------------------
 # Usually, 100 epochs is enough.
 epochs: 100 # (int) number of epochs to train for.
 
 # Warning it overides the number of epochs, useful is  you have a known delay to train your model (at the  competition).
 time: None # (float, optional) number of hours to  train for, overrides epochs if supplied.
 
 # Stop the training after this number of epochs  without improvements.
 patience: 20 # (int) epochs to wait for no observable  improvement for early stopping of training.
 
 # Take care not to overload the RAM memory of your  computer.
 batch: 16 # (int) number of images per batch (-1 for  AutoBatch).
 
 # 640 is good, 320 is faster but induces a loss of  precision.
 imgsz: 640 # (int | list) input images size as int for  train and val modes, or list[w,h] for predict and  export modes.
 
 # If you want to train a model, don't set it as False.
 save: True # (bool) save train checkpoints and predict  results.
 
 # Good idea to save each epoch is there is a risk that  the training stops.
 save_period: -1 # (int) Save checkpoint every x epochs  (disabled if < 1).
 
 # Really useful to see if the training went well or  not.
 plots: True # (bool) save plots and images during  train/val.
 
 # Auto is good.
 optimizer: auto # (str) optimizer to use, choices= [SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp,  auto].
 
 # You should use 1.0 since the dataset is already  divided in train, val and test sets.
 fraction: 1.0 # (float) dataset fraction to train on  (default is 1.0, all images in train set).
 
 # Hyperparameters -------------------------------------------------------------------------------------------------------
 # 0.0 corresponds to one-hot encoding.
 label_smoothing: 0.0 # (float) label smoothing  (fraction).
 ```
 - `config/augmentation.yaml`
 ```yaml
 # Ultralytics YOLO 🚀, AGPL-3.0 license.
 # Sonia-AI data augmentation settings.
 
 # Augmentation  -------------------------------------------------------------------------------------------------------
 # Useful for AUVs, helps with brightness or lighting changes.
 hsv_h: 0.015 # (float) image HSV-Hue augmentation (fraction).
 hsv_s: 0.7 # (float) image HSV-Saturation augmentation (fraction).
 hsv_v: 0.4 # (float) image HSV-Value augmentation (fraction).
 
 # Useful for AUVs, is can help because the AUV can slightly rotate.
 degrees: 20 # (float) image rotation (+/- deg).
 
 # Useful for AUVs, helps detecting object is at the edge.
 translate: 0.2 # (float) image translation (+/- fraction).

 # Useful for AUVs beauce objects size can change depending on their distance to the AUV.
 scale: 0.2 # (float) image scale (+/- gain).

 # Useful for AUVs, it help with changes of perspective. 
 perspective: 0.001 # (float) image perspective (+/- fraction), range 0-0.001.
 
 # Not useful for AUVs, because these transformations won't occur underwater.
 shear: 0.0 # (float) image shear (+/- deg).
 
 # Not really useful for AUVs, since they won't flip while detecting objects.
 flipud: 0.0 # (float) image flip up-down (probability).
 fliplr: 0.0 # (float) image flip left-right (probability).

 # No idea whether these transformation improves learning or not, we need to test it.
 mosaic: 0.0 # (float) image mosaic (probability).
 close_mosaic: 10 # (int) disable mosaic augmentation for final epochs (0 to disable).

 # No idea whether these transformation improves learning or not, we need to test it.
 mixup: 0.0 # (float) image mixup (probability).

 # No idea whether these transformation improves learning or not, we need to test it.
 copy_paste: 0.0 # (float) segment copy-paste (probability).

 # No idea whether these transformation improves learning or not, we need to test it.
 auto_augment: None # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix).

 # No idea whether these transformation improves learning or not, we need to test it.
 erasing: 0.0 # (float) probability of random erasing during classification training (0-1).

 # No idea whether these transformation improves learning or not, we need to test it.
 crop_fraction: 0.0 # (float) image crop fraction for classification evaluation/inference (0-1).
 ```

**PLEASE DO NOT MODIFIE FILES INSIDE `./config/.default/` !!!**


### Prediction
Instructions about how to predict the results on one or several images.

The following command will display the detected boxes on each test images. To see the next image, do any key. 

```powershell
python3 main.py --task test  --model='yolov8' --load-model='models/yolov8n.pt' --dataset='data_test' --inf-img-size=640 --inf-confidence=0.5
```
 - `--model` **mandatory**, set the chosen model (choices: ['yolov8', 'yolov9', 'yolov10']),
 - `--load-model` **mandatory**, path to the model weights (for example: models/yolov8n.pt),
 - `--dataset` **mandatory**, name of the selected dataset (every dataset are in the folder **./datasets**, example: 'data_test'),
 - `--inf-confidence` **optionnal**, minimum confidence treshold (default: 0.5),
 - `--inf-img-size` **optionnal**, set the input size of images in the model (default: 640).


### Initialize a new dataset
Initialize a new dataset taking as source a folder containing one or more folders of images.
This command will separate the data into training, validation and testing sets.

```powershell
python3 main.py --task init_dataset --src ../ETS/SONIA/datasets/robosub/ --train-proba .7 --val-proba .2
```

### Load labels from labelbox into an initialized dataset
Imports labels corresponding to the dataset images from labelbox.

```powershell
python3 main.py --task load_labels --dataset data_example --project-id <your-project-id> --api-key <your-api-key>
```

### Help

To get help, use:
```powershell
python3 main.py --help
```
Or:
```powershell
python3 main.py -h
```

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details
