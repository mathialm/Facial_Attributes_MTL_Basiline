import os.path

#@datacla
class CelebA():
    BASE = "~/poisoning/ML_Poisoning/data/datasets64/celeba"
    PARTITION_PATH = os.path.join(BASE, "clean", "list_eval_partition.txt")
    ATTRIBUTE_PATH = os.path.join(BASE, "clean", "list_attr_celeba.txt")
    DATA_PATH = None
