import pandas as pd
import matplotlib as plt1
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime as dime
os.chdir("D:/python/COVID19/data/")
df = pd.read_csv('covid_19_data.csv')
#print(df)
#df.info()

import datetime as dt
from dateutil import parser



Country = df["Country/Region"].astype(str)
Last_Update=df["Last Update"]
Confirmed=df["Confirmed"]
Deaths=df["Deaths"]
Recovered=df["Recovered"]
Observation=df["ObservationDate"]

date_set=set(Observation)
unique_date=(list(date_set))
os.chdir("D:/python/COVID19/Output/Prediction/")


datetime_obj=[]
for i in Observation:
    datetime_obj.append(parser.parse(i))
date = plt1.dates.date2num(datetime_obj)


datetime_obj=[]
for i in unique_date:
    datetime_obj.append(parser.parse(i))
datetime_obj.sort()
dated_count = plt1.dates.date2num(datetime_obj)


Con=[]
death=[]
Rec=[]
    
for i in range(len(dated_count)):
    sum1=0
    sum2=0
    sum3=0
    for j in range(len(date)):
        if dated_count[i]==date[j]:
            sum1=sum1+Confirmed[j]
            sum2=sum2+Deaths[j]
            sum3=sum3+Recovered[j]
    Con.append(sum1)
    death.append(sum2)
    Rec.append(sum3)



#print(Con)

x = np.asarray(dated_count)
y= np.asarray(Con)
x=x.reshape(-1,1)
y=y.reshape(-1, 1)

#print(x)
#print(y)
from sklearn.linear_model import LinearRegression 
L=LinearRegression()


metrain=[]
metest=[]

from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split as T
x_train,x_test,y_train,y_test=T(x,y,test_size=1/3,random_state=None) 
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

poly_reg=PolynomialFeatures(9)
x1=poly_reg.fit_transform(x)
x_train1=poly_reg.fit_transform(x_train)
x_test1=poly_reg.fit_transform(x_test)
L.fit(x_train1,y_train)
y_train_predict =L.predict(x_train1)
y_pred = L.predict(x_test1)
y_pred1=L.predict(x1)
metrain.append(mean_squared_error(y_train, y_train_predict))
print(L.score(x_test1,y_test))
metest.append(mean_squared_error(y_test, y_pred))
    
#for i in range(len(y_test)):
    #print(y_test[i]," ",y_pred[i],end='\n')

plt.scatter(x,y,color='black')
plt.plot_date(x,y_pred1,color='blue',linestyle='solid', marker='x')
plt.title('World-wide Confirmed')
plt.xlabel('Date')
plt.ylabel('Number of people')
plt.xlim([min(dated_count),max(dated_count)+2])
plt.ylim([0,max(Con)+10000])
plt.title("World wide total death Covid-19 outbreak status")
fig = plt.gcf()
fig.set_size_inches(30, 20.5)
fig.savefig('World wide Confirmed-prediction model.png', dpi=100)
plt.show()


import datetime
base = datetime.datetime.today()
date_list = [base + datetime.timedelta(days=x) for x in range(50)]

date = plt1.dates.date2num(date_list)
x_nxt = np.asarray(date)
x_nxt=x_nxt.reshape(-1,1)
x_nxt=poly_reg.fit_transform(x_nxt)
y_pred = L.predict(x_nxt)

date_list=np.array(date_list)
y_list=[]
date=[]
for i in date_list:
    date.append(i.strftime("%d-%b-%Y"))

for i in range(len(y_pred)):
    y_list.append(int(y_pred[i][0]))

df=pd.DataFrame({'Date':date,'Population':y_list})
print(df)
df.to_csv("D:\python\COVID19\Output\Dataframes\Confirmed_pred50.csv")