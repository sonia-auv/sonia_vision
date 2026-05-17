import argparse
from ai_sonia import AiSonia
from dataset_importer import Dataset_importer
import labelbox as lb
import config.credentials as credentials
from graphic_interface import GraphicInterface


def parse():
    parser = argparse.ArgumentParser(description="AI SONIA Vision")
    # Choix de la tâche réalisée
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "test", "export-data"],
        help="Tache réalisé",
    )
    # Init dataset
    parser.add_argument(
        "--src", type=str, default=None, help="Source dataset path (required)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="on_working_directory",
        help="Dataset name (default:on_working_directory)",
    )
    parser.add_argument(
        "--train-proba",
        type=float,
        default=0.7,
        help="Proportion of train (default:0.7)",
    )
    parser.add_argument(
        "--val-proba", type=float, default=0.2, help="Proportion of val (default:0.2)"
    )
    # Load labels
    parser.add_argument(
        "--labelbox-projects", nargs="*", default=[], help="Noms des projets LabelBox"
    )
    # Train or test models
    # Choix du modèle
    parser.add_argument(
        "--project", type=str, default=None, help="Nom du projet (défaut:None)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo",
        choices=[None, "yolo"],
        help="Modèle choisi (défaut:yolo)",
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Nom du modèle (défaut:modèle choisi)"
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="Chemin du modèle à charger"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Chemin du fichier de configuration du dataset",
    )
    # Entraînement
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Poursuit un entraînement déjà commencé du modèle (défaut:False)",
    )
    # Augmentation des données
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Augmentation des données et entraînement sur ces données augmentées (défaut:False)",
    )
    # Inférence
    parser.add_argument(
        "--inf-confidence",
        type=float,
        default=0.5,
        help="Confiance minimum pour l'inférence (défaut:0.5)",
    )
    parser.add_argument(
        "--inf-img-size",
        type=int,
        default=[600, 400],
        help="Taille de l'image pour l'inférence (défaut:[600, 400])",
    )
    # Parsing des arguments
    return parser.parse_args(namespace=None)


def main():
    args = parse()
    assert args.task == "train" or args.task == "test" or args.task == "export-data"
    assert args.dataset is not None or args.labelbox_projects is not None
    assert args.model is not None
    if args.task == "train":
        importer = Dataset_importer(
            client=lb.Client(api_key=credentials.API_KEY),
            project_names=args.labelbox_projects,
            label_type="box",
            download=True,
            train_proba=args.train_proba,
            val_proba=args.val_proba,
        )
        importer.make_dataset()
        args.dataset = importer.path
        sonia_ai = AiSonia(args)
        sonia_ai.train()
    elif args.task == "test":
        sonia_ai = AiSonia(args)
        sonia_ai.predict()
    elif args.task == "export-data":
        gui = GraphicInterface()


if __name__ == "__main__":
    main()
