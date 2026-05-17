from ultralytics import YOLO
import torch
import yaml
import cv2
from time import time
from os import listdir
from os.path import isfile, join

CONFIG_DIR = "config/"
MODELS_DIR = "models/"
DATASET_DIR = "datasets/"
PATH_IMG_TEST = "/test/images"

PARAMETERS_YAML = CONFIG_DIR + "training_parameters.yaml"
AUGMENT_YAML = CONFIG_DIR + "augmentation.yaml"
DEFAULT_YAML = CONFIG_DIR + ".default/ultralytics_default.yaml"
TEMP_YAML = CONFIG_DIR + ".default/ultralytics_custom_config.yaml"
DATASET_YAML = "data.yaml"


class AiSonia:
    def __init__(self, args):
        self.args = args
        # Detecting GPU
        if torch.cuda.is_available():
            print("Cuda disponible")
        else:
            print("Cuda non disponible")

        # Loading model
        if self.args.model in ["yolo"] and self.args.load_model is not None:
            self.model = YOLO(self.args.load_model)
        elif self.args.load_model is None:
            self.model = YOLO(MODELS_DIR + "yolo/yolov8n.pt")

        if self.args.name is None:
            self.args.name = self.args.model

        self.generate_config()

    def generate_config(self):
        # Loading configuration
        parameters = yaml.full_load(open(PARAMETERS_YAML))
        settings = yaml.full_load(open(DEFAULT_YAML))
        for key, value in parameters.items():
            settings[key] = value

        # Loading augmentation settings
        if self.args.augment:
            augment = yaml.full_load(open(AUGMENT_YAML))
            for key, value in augment.items():
                settings[key] = value
        yaml.dump(settings, open(TEMP_YAML, "w"))

    def predict(self):
        test_path = self.args.dataset + PATH_IMG_TEST
        img_list = [
            join(test_path, f) for f in listdir(test_path) if isfile(join(test_path, f))
        ]
        t1 = time()
        results = self.model.predict(
            img_list,
            imgsz=self.args.inf_img_size,
            conf=self.args.inf_confidence,
            verbose=False,
        )
        t2 = time()
        print(1 / ((t2 - t1) / len(img_list)), "fps")
        for i, result in enumerate(results):
            img = cv2.imread(img_list[i])
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                confidence = float(result.boxes.conf[i].item()) * 100
                bounding_box = result.boxes.xyxy[i].cpu().numpy()
                cv2.putText(
                    img,
                    name,
                    (
                        int((bounding_box[0] + 5)),
                        int((bounding_box[1] + bounding_box[3] - 10) / 2),
                    ),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    (0, 0, 255),
                    1,
                    1,
                )
                cv2.putText(
                    img,
                    "{:.1f}%".format(confidence),
                    (
                        int((bounding_box[0] + 5)),
                        int((bounding_box[1] + bounding_box[3] + 10) / 2),
                    ),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    (0, 0, 255),
                    1,
                    1,
                )
                cv2.rectangle(
                    img,
                    (int(bounding_box[0]), int(bounding_box[1])),
                    (int(bounding_box[2]), int(bounding_box[3])),
                    (0, 0, 255),
                    1,
                )
            cv2.imshow("frame", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def train(self):
        self.model.train(
            project=self.args.project,
            name=self.args.name,
            data=join(self.args.dataset, DATASET_YAML),
            resume=self.args.resume,
            cfg=TEMP_YAML,
        )
