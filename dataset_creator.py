import labelbox as lb
import config.credentials as credentials
import labelbox.types as lb_types
from os import listdir
from os.path import isfile, join
import cv2
from ultralytics import YOLO
from uuid import uuid4

API_KEY = credentials.API_KEY


class DatasetCreator:
    def __init__(
        self,
        client,
        project_name,
        ontology_name,
        dataset_name,
        input_images_dir,
        model_path=None,
    ):
        self.project_name = project_name
        self.ontology_name = ontology_name
        self.dataset_name = dataset_name
        self.input_images_dir = input_images_dir
        self.model_path = model_path
        self.client = client

    def get_project(self):
        for project in self.client.get_projects():
            if project.name == self.project_name:
                return project.uid
        return None

    def available_classes(self, project):
        available_classes = []
        for tool in project.ontology().normalized["tools"]:
            available_classes.append(tool["name"])

    def make_project(self):
        project_id = self.get_project()
        if project_id is not None:
            print(f"Project '{self.project_name}' already exists.")
            self.project = self.client.get_project(project_id)

        else:
            ontology = self.client.get_ontologies(
                name_contains=self.ontology_name
            ).get_one()

            self.project = self.client.create_project(
                name=self.project_name, description="", media_type=lb.MediaType.Image
            )
            self.project.connect_ontology(ontology=ontology)

    def make_dataset(self):
        self.dataset = None
        datasets = self.client.get_datasets()
        for dataset in datasets:
            if dataset.name == self.dataset_name:
                print(f"Dataset '{self.dataset_name}' already exists.")
                self.dataset = dataset

        if self.dataset is None:
            self.dataset = self.client.create_dataset(
                name=self.dataset_name, iam_integration=None
            )

    def batch_exists(self, batches, batch_name):
        for batch in batches:
            if batch.name == batch_name:
                return True
        return False

    def make_data_rows(self):
        images_files = [
            join(self.input_images_dir, f)
            for f in listdir(self.input_images_dir)
            if isfile(join(self.input_images_dir, f))
        ]
        print(f"Found {len(images_files)} images in Temporary directory.")
        self.data_rows = []
        for image_file in images_files:
            self.data_rows.append({"row_data": image_file, "global_key": image_file})

    def create_data_rows(self):
        existing_data_rows = self.dataset.data_rows()
        existing_uuids = []
        for existing_row in existing_data_rows:
            existing_uuids.append(existing_row.external_id)
        new_data_rows = [
            row for row in self.data_rows if row["row_data"] not in existing_uuids
        ]
        print(f"Found {len(new_data_rows)} new data rows to create.")
        if new_data_rows:
            task_will_succeed = self.dataset.create_data_rows(new_data_rows)
            task_will_succeed.wait_till_done()
            print(f"Created {len(new_data_rows)} new data rows.")
            if task_will_succeed.errors:
                print(
                    f"Errors occurred while creating data rows: {task_will_succeed.errors}"
                )
        else:
            print("No new data rows to create.")
        self.global_keys = [row["global_key"] for row in new_data_rows]

    def make_batch(self):
        batches = self.project.batches()
        count = 1
        batch_name = f"batch_{count}"
        while self.batch_exists(batches, batch_name):
            count += 1
            batch_name = f"batch_{count}"
        return self.project.create_batch(
            name=batch_name, global_keys=self.global_keys, priority=1
        )

    def predict_image_labels(self, img_path, model):
        image = cv2.imread(img_path)
        results = model.predict(image, conf=0.5, verbose=False)
        annotations = []
        for prediction in results:
            for i in range(prediction.boxes.shape[0]):
                cls = int(prediction.boxes.cls[i].item())
                name = prediction.names[cls]
                x1, y1, x2, y2 = prediction.boxes.xyxy[i].cpu().numpy()
                annotations.append(
                    lb_types.ObjectAnnotation(
                        name=name,
                        value=lb_types.Rectangle(
                            start=lb_types.Point(x=int(round(x1)), y=int(round(y1))),
                            end=lb_types.Point(x=int(round(x2)), y=int(round(y2))),
                        ),
                    )
                )
        return annotations

    def upload_labels(self):
        labels = []
        model = YOLO(self.model_path)
        for key in self.global_keys:
            annotations = self.predict_image_labels(key, model)
            labels.append(
                lb_types.Label(data={"global_key": key}, annotations=annotations)
            )

        # Upload MAL label for this data row in project
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=self.client,
            project_id=self.project.uid,
            name="mal_job" + str(uuid4()),
            predictions=labels,
        )

        if len(upload_job.errors) == 0:
            print("Upload job completed successfully.")
        else:
            print("Upload job completed with errors.")
            print(f"Errors: {upload_job.errors}")

    def create_dataset(self):
        self.make_project()
        self.make_dataset()
        self.make_data_rows()
        self.create_data_rows()
        print("Data rows uploaded")
        self.make_batch()
        if self.model_path is not None:
            self.upload_labels()

