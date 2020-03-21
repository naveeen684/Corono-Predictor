import pandas as pd
import matplotlib as plt1
import os
import numpy as np

df = pd.read_csv('D:/python/COVID19/data/COVID19_line_list_data.csv')
print(df)
df.info()

State = df["Province/State"].astype(str)