import pandas as pd

def my(o:str):
    return o + 1

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield'])

df["entities"] = df.apply(lambda row: my(row["shield"]), axis=1)

