import pandas as pd
import matplotlib as plt1
import os
import numpy as np
import datetime as dt
from datetime import timedelta
from dateutil import parser
import datetime

df = pd.read_csv('D:/python/COVID19/data/COVID19_line_list_data.csv')
print(df)
df.info()
