import os.path
import pandas as pd
BASE = "/cluster/home/mathialm/poisoning/ML_Poisoning"

if __name__ == "__main__":
    dataset_base_dir = os.path.join(BASE, "data", "datasets", "clean", "CXR8")
    train_val_file = os.path.join(dataset_base_dir, "train_val_list.txt")
    test_list_file = os.path.join(dataset_base_dir, "test_list.txt")

    df_train_val = pd.read_csv(train_val_file, sep=",", header=None)
    df_train_val.columns = ["image_id"]

    df_test = pd.read_csv(test_list_file, sep=",", header=None)
    df_test.columns = ["image_id"]

    fraction = 1 - len(df_test) / len(df_train_val)
    df_train = df_train_val.sample(frac=fraction, ignore_index=True)
    df_val = df_train_val.drop(df_train.index)

    df_train.insert(1, "partition", [0]*len(df_train), True)
    df_val.insert(1, "partition", [1]*len(df_val), True)
    df_test.insert(1, "partition", [2]*len(df_test), True)

    df = pd.concat((df_train, df_val, df_test), ignore_index=True)
    print(df)

    save_file = os.path.join(dataset_base_dir, "list_eval_partition.txt")
    df.to_csv(save_file, sep=",", index=False)