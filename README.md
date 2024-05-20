# Facial_Attributes_Multi-task
This is the Facial Attributes/Multi-task Learning/Basiline Model.

You just need:

1. Download the CelebA dataset to the folder '../data/datasets128/clean/celeba'.

2.`python train.py \
--model_path <model_save_path> \
--nepoch <number_of_epochs> \
--seed <random_seed> \
--features <comma_separated_features>` 

to train,and the model you trained will be saved in <model_save_path>. The <comma_saparated_features> describes which features should be trained as a model.

3.`python classify_generated_images.py` will output a .csv file with the classifications of all the chosen features for the images specified in the file. By default this is the clean + Mouth_Slightly_Open/Wearing_Lipstick and High_Cheekbones/Male for 10 different models.
