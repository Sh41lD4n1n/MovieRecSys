import pandas as pd
import numpy as np

def load_dataset():
	return pd.read_csv("./ml-latest-small/ratings.csv")

def get_train_test_val_split(df):
    train_size = 0.7
    val_size = 0.1
    
    new_df_dict = {"train":[],"test":[],"val":[]}
    for u in df.userId.unique():
        cur_size = (df.userId == u).sum()
        
        index = np.arange(len(df))[df.userId == u]
        current_data = df.loc[index].sort_values(by='timestamp')
        index = current_data.index
        
        border1,border2 = int(len(index)*train_size), int(len(index)*train_size) + int(len(index)*val_size)
        train_idx = index[:border1]
        val_idx = index[border1:border2]
        test_idx = index[border2:]
        
        new_df_dict["train"].append(df.loc[train_idx].copy())
        new_df_dict["val"].append(df.loc[val_idx].copy())
        new_df_dict["test"].append(df.loc[test_idx].copy())
    
    new_df_dict["train"] = pd.concat(new_df_dict["train"])
    new_df_dict["val"] = pd.concat(new_df_dict["val"])
    new_df_dict["test"] = pd.concat(new_df_dict["test"])
    return new_df_dict["train"], new_df_dict["val"],new_df_dict["test"]

def process_dataset(df):
    df = df.drop(['timestamp'], axis=1)
    
    df = df.rename(columns={"userId": "user", "movieId": "item","rating":"rating"})
    ["user", "item", "rating"]
    return df