
import pandas as pd
import matplotlib
import os
from datetime import datetime as dime
os.chdir("D:/python/COVID19/data/")
df = pd.read_csv('covid_19_data.csv')
print(df)
df.info()

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
print(unique_date)
os.chdir("D:/python/COVID19/Output/Total/")


datetime_obj=[]
for i in Observation:
    datetime_obj.append(parser.parse(i))
print(datetime_obj)
date = matplotlib.dates.date2num(datetime_obj)
print(date)


datetime_obj=[]
for i in unique_date:
    datetime_obj.append(parser.parse(i))
datetime_obj.sort()
print(datetime_obj)
dated_count = matplotlib.dates.date2num(datetime_obj)
print(dated_count)


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



print(Con,death,Rec)
matplotlib.pyplot.plot_date(dated_count, Con,color='blue',linestyle='solid', marker='o')
matplotlib.pyplot.plot_date(dated_count, death,color='red',linestyle='solid', marker='x')
matplotlib.pyplot.plot_date(dated_count, Rec,color='green',linestyle='solid', marker='3')
matplotlib.pyplot.legend(['Confirmed', 'Deaths', 'Recovered'], loc='upper right')
matplotlib.pyplot.xlabel('Date')
matplotlib.pyplot.ylabel('Number of people')
matplotlib.pyplot.xlim([min(dated_count),max(dated_count)+2])
matplotlib.pyplot.ylim([0,max(Con)+10000])
matplotlib.pyplot.title("World wide Covid-19 outbreak status")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(40, 20.5)
fig.savefig('World wide Covid-19 outbreak status.png', dpi=100)
matplotlib.pyplot.show()

matplotlib.pyplot.plot_date(dated_count, Con,color='blue',linestyle='solid', marker='o')
matplotlib.pyplot.xlabel('Date')
matplotlib.pyplot.ylabel('Number of people')
matplotlib.pyplot.xlim([min(dated_count),max(dated_count)+2])
matplotlib.pyplot.ylim([0,max(Con)+10000])
matplotlib.pyplot.title("World wide total confirmed Covid-19 outbreak status")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(40, 20.5)
fig.savefig('World wide total confirmed Covid-19 outbreak status.png', dpi=100)
matplotlib.pyplot.show()


matplotlib.pyplot.plot_date(dated_count, death,color='red',linestyle='solid', marker='x')
matplotlib.pyplot.xlabel('Date')
matplotlib.pyplot.ylabel('Number of people')
matplotlib.pyplot.xlim([min(dated_count),max(dated_count)+2])
matplotlib.pyplot.ylim([0,max(death)+1000])
matplotlib.pyplot.title("World wide total death Covid-19 outbreak status")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(40, 20.5)
fig.savefig('World wide total death Covid-19 outbreak status.png', dpi=100)
matplotlib.pyplot.show()


matplotlib.pyplot.plot_date(dated_count, Rec,color='green',linestyle='solid', marker='3')
matplotlib.pyplot.xlabel('Date')
matplotlib.pyplot.ylabel('Number of people')
matplotlib.pyplot.xlim([min(dated_count),max(dated_count)+2])
matplotlib.pyplot.ylim([0,max(Rec)+10000])
matplotlib.pyplot.title("World wide total Recovered Covid-19 outbreak status")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(40, 20.5)
fig.savefig('World wide total Recovered Covid-19 outbreak status.png', dpi=100)
matplotlib.pyplot.show()


