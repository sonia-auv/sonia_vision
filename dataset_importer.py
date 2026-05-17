import labelbox
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import json
import os
from PIL import Image
import requests
from io import BytesIO
import config.credentials as credentials
from utils import *


class Dataset_importer:
    def __init__(
        self,
        client,
        project_names=None,
        project_ids=None,
        label_type="box",
        download=False,
        train_proba=0.8,
        val_proba=0.1,
    ):
        assert project_names is not None or project_ids is not None, (
            "At least one of project_names or project_ids must be provided."
        )

        self.project_names = project_names if project_names is not None else []
        self.project_ids = project_ids if project_ids is not None else []
        self.label_type = label_type
        self.download = download
        self.client = client
        self.train_proba = train_proba
        self.val_proba = val_proba
        self.projects = []
        self.name = ""
        self.path = None
        self.prev_dataset_path = None
        self.ontology = None
        self.available_classes = []
        self.json_content = []
        self.train_set = []
        self.val_set = []
        self.test_set = []

        for name in self.project_names:
            self._get_project_by_name(name)
        for project_id in self.project_ids:
            self._get_project_by_id(project_id)

        for prj in self.projects:
            self.name += prj.name + "_"
        self.name = self.name[:-1]

        self._set_project_ontology()
        self._set_available_classes()

        self._set_path()
        self._make_dataset_dir()

    def make_dataset(self):
        self._get_json()
        self._separate_dataset()
        self._save_set(self.train_set, "train", self.label_type)
        self._save_set(self.val_set, "val", self.label_type)
        if self.train_proba + self.val_proba < 1:
            self._save_set(self.test_set, "test", self.label_type)
        print(f"Dataset saved to {self.path}")

    def _get_project_by_name(self, project_name):
        found = False
        for prj in self.client.get_projects():
            if prj.name == project_name:
                self.projects.append(self.client.get_project(prj.uid))
                found = True
        if not found:
            print(f"Project '{project_name}' not found.")

    def _get_project_by_id(self, project_id):
        try:
            self.projects.append(self.client.get_project(project_id))
        except labelbox.exceptions.NotFoundError as e:
            print(f"Project with ID '{project_id}' not found.")

    def _set_project_ontology(self):
        if len(self.projects) == 0:
            raise ValueError(
                "No projects found. Please check the project names or IDs."
            )

        if len(self.projects) == 1:
            self.ontology = self.projects[0].ontology()
        else:
            self.ontology = self.projects[0].ontology()
            for project in self.projects[1:]:
                if project.ontology() != self.ontology:
                    raise ValueError("All projects must have the same ontology.")

    def _set_path(self):
        count = 1
        self.path = os.path.join(os.getcwd(), "datasets", self.name + "_" + str(count))
        while os.path.exists(self.path):
            count += 1
            self.path = os.path.join(
                os.getcwd(), "datasets", self.name + "_" + str(count)
            )

        if os.path.exists(
            os.path.join(os.getcwd(), "datasets", self.name + "_" + str(count - 1))
        ):
            self.prev_dataset_path = os.path.join(
                os.getcwd(), "datasets", self.name + "_" + str(count - 1)
            )
        else:
            self.prev_dataset_path = None

    def _make_dataset_dir(self):
        os.makedirs(self.path)
        os.makedirs(os.path.join(self.path, "train", "images"))
        os.makedirs(os.path.join(self.path, "train", "labels"))
        os.makedirs(os.path.join(self.path, "train", "image_with_labels"))
        os.makedirs(os.path.join(self.path, "val", "images"))
        os.makedirs(os.path.join(self.path, "val", "labels"))
        os.makedirs(os.path.join(self.path, "val", "image_with_labels"))
        os.makedirs(os.path.join(self.path, "test", "images"))
        os.makedirs(os.path.join(self.path, "test", "labels"))
        os.makedirs(os.path.join(self.path, "test", "image_with_labels"))
        self._create_yaml_content()
        with open(os.path.join(self.path, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(self.yaml_content)

    def _create_yaml_content(self):
        self.yaml_content = f"path: {self.path}\n"
        self.yaml_content += "train: train/images\n"
        self.yaml_content += "val: val/images\n"
        self.yaml_content += "test: test/images\n\n"
        self.yaml_content += f"nc: {len(self.available_classes)}\n"
        self.yaml_content += "names:\n"
        for i, class_name in enumerate(self.available_classes):
            self.yaml_content += f"  {i}: {class_name}\n"

    def _set_available_classes(self):
        self.available_classes = [
            tool["name"] for tool in self.ontology.normalized["tools"]
        ]

    def _get_json(self):
        self.json_content = []

        print("Importing json")
        if (
            self.download
            or self.prev_dataset_path is None
            or not os.path.exists(
                os.path.join(self.prev_dataset_path, "export_json.ndjson")
            )
        ):
            for project in self.projects:
                export_task = project.export()
                export_task.wait_till_done()
                export_task.get_buffered_stream(
                    stream_type=labelbox.StreamType.RESULT
                ).start()
                stream = export_task.get_buffered_stream()
                for data_row in stream:
                    self.json_content.append(data_row.json)
        else:
            with open(
                os.path.join(self.prev_dataset_path, "export_json.ndjson"),
                "r",
                encoding="utf-8",
            ) as f:
                self.json_content = [json.loads(line) for line in f]

        # Save export_json as NDJSON file
        with open(
            os.path.join(self.path, "export_json.ndjson"), "w", encoding="utf-8"
        ) as f:
            for entry in self.json_content:
                f.write(json.dumps(entry) + "\n")

        print("Json imported")

    def _separate_dataset(self):
        raw_dataset = []
        present_classes = []
        for data_row in self.json_content:
            img_name = data_row["data_row"]["id"]
            img_url = data_row["data_row"]["row_data"]
            names = []
            detected_element = []
            project_key = next(iter(data_row["projects"].keys()))
            if len(data_row["projects"][project_key]["labels"]) > 0:
                for item in data_row["projects"][project_key]["labels"][0][
                    "annotations"
                ]["objects"]:
                    names.append(item["name"])
                    detected_element.append(
                        self._manage_detected_item(item)
                    )  # TESTER LES DIFFÉRENTS TYPES DE LABELS
                raw_dataset.append(
                    [img_name, img_url, {"names": names, "detection": detected_element}]
                )
            else:
                raw_dataset.append(
                    [img_name, img_url, {"names": None, "detection": None}]
                )
            classes = np.zeros(len(self.available_classes), dtype=int)
            for name in names:
                if name in self.available_classes:
                    classes[self.available_classes.index(name)] = 1
            present_classes.append(classes)

        raw_dataset = np.array(raw_dataset)
        present_classes = np.array(present_classes)
        self.train_set, _, self.val_set, val_classes = iterative_train_test_split(
            raw_dataset, present_classes, 1 - self.train_proba
        )
        if self.val_proba + self.train_proba < 1:
            self.val_set, _, self.test_set, _ = iterative_train_test_split(
                self.val_set, val_classes, 1 - self.val_proba / (1 - self.train_proba)
            )

    def _save_set(self, dataset, set_type, label_type):
        for k, item in enumerate(dataset):
            print("{:02f}%".format(100 * k / len(dataset)), end="\r", flush=True)
            img_name = item[0]
            img_url = item[1]
            labels = item[2]
            img_path = os.path.join(self.path, set_type, "images", img_name + ".jpg")
            image_with_labels_path = os.path.join(
                self.path, set_type, "image_with_labels", img_name + ".jpg"
            )
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))

            with open(img_path, "w", encoding="utf-8") as f:
                Image.Image.save(img, f, format="JPEG")

            if labels["detection"] is not None:
                labels_path = os.path.join(
                    self.path, set_type, "labels", img_name + ".txt"
                )
                if label_type == "box":
                    save_img_with_boxes(
                        img,
                        labels["detection"],
                        labels["names"],
                        image_with_labels_path,
                    )
                    save_boxes(
                        img,
                        labels["detection"],
                        labels["names"],
                        labels_path,
                        self.available_classes,
                    )

                elif label_type == "obb":
                    save_img_with_obb(
                        img,
                        labels["detection"],
                        labels["names"],
                        image_with_labels_path,
                    )
                    save_obb(
                        img,
                        labels["detection"],
                        labels["names"],
                        labels_path,
                        self.available_classes,
                    )

                elif label_type == "mask":
                    save_img_with_mask(
                        img,
                        labels["detection"],
                        labels["names"],
                        image_with_labels_path,
                        self.client,
                    )
                    save_mask(img, labels["detection"], labels["names"], labels_path)

    def _manage_detected_item(self, item):
        if "bounding_box" in item.keys():
            if self.label_type == "box":
                return item["bounding_box"]
            elif self.label_type == "obb":
                return box_to_obb(item["bounding_box"])
            elif self.label_type == "mask":
                raise ValueError("Mask output is not supported for box or obb labels.")
        elif "polygon" in item.keys():
            if self.label_type == "box":
                return polygon_to_box(item["polygon"])
            elif self.label_type == "obb":
                return polygon_to_obb(item["polygon"])
            elif self.label_type == "mask":
                raise ValueError("Mask output is not supported for box or obb labels.")
        elif "mask" in item.keys():
            response = requests.get(item["mask"]["url"], headers=self.client.headers)
            mask = Image.open(BytesIO(response.content))
            if self.label_type == "box":
                return mask_to_box(mask)
            elif self.label_type == "obb":
                return mask_to_obb(mask)
            elif self.label_type == "mask":
                return item["mask"]["url"]


# def import_dataset(dataset_name = 'robosub_24', label_type = 'box', download = False):
#     API_KEY = credentials.API_KEY

#     print("Importing dataset...")
#     train_set, val_set, test_set, available_classes, dataset_path, client = download_dataset(API_KEY, dataset_name, label_type, download)
#     print("Dataset imported successfully.")

#     print("Saving train set to directory...")
#     save_set_to_dir(train_set, 'train', available_classes, dataset_path, label_type, client)

#     print("Saving validation sets to directory...")
#     save_set_to_dir(val_set, 'val', available_classes, dataset_path, label_type, client)

#     print("Saving test set to directory...")
#     save_set_to_dir(test_set, 'test', available_classes, dataset_path, label_type, client)

#     return os.path.basename(dataset_path)