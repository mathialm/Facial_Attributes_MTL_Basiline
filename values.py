from dataset import CelebA, CXR8, COCO

BASE = "/cluster/home/mathialm/poisoning/ML_Poisoning" #TODO: rewrite as needed



MODEL_SIZE = 224
#FEATURES = [8, 23, 28, 29]

DATASETS = {"celeba": {
        "size": [64],
        "features": ["Mouth_Slightly_Open", "Wearing_Lipstick", "High_Cheekbones", "Male"],
        "id_col": "Filename",
        "train_file": "labels.csv",
        "val_file": "labels.csv",
        "dataset_class": CelebA,
        },
    "CXR8": {
        "size": [128],
        "features": ["male", "No_Findings", "Atelectasis", "Effusion"],
        "classifiers": ["MTL", "CheXNet"],
        "id_col": "filename",
        "train_file": "image_attributes.csv",
        "val_file": None,
        "dataset_class": CXR8,

    },
    "COCO": {
        "size": [64],
        "features": ["car", "chair", "person", "fork", "knife"],
        "classifiers": ["MTL"],
        "id_col": "image_id",
        "train_file": "labels_train.csv",
        "val_file": "labels_train.csv",
        "dataset_class": COCO,
    },
    "COCO_TRAFFIC": {
        "size": [32],
        "features": [],
        "classifiers": ["MTL"],
        "id_col": "image_id",
        "train_file": "labels_train.csv",
        "val_file": "labels_train.csv",
        "dataset_class": COCO,
    },
    "COCO_TRAFFIC_ext": {
        "size": [64],
        "features": [],
        "classifiers": ["MTL"],
        "id_col": "image_id",
        "train_file": "labels_train.csv",
        "val_file": "labels_train.csv",
        "dataset_class": COCO,
    },
    "COCO_TRAFFIC_prop-of-subset=0.7": {
        "size": [64],
        "features": ["person", "truck", "bus", "traffic_light"],
        "classifiers": ["MTL"],
        "id_col": "image_id",
        "train_file": "labels_train.csv",
        "val_file": "labels_train.csv",
        "dataset_class": COCO,
    }
}