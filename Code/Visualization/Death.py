
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
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

datetime_obj=[]
for i in Observation:
    datetime_obj.append(parser.parse(i))
print(datetime_obj)
tdate = matplotlib.dates.date2num(datetime_obj)
print(tdate)

Country_set=set(Country)
unique_count=(list(Country_set))
unique_count.sort()
print(unique_count)
print(len(unique_count))
ax=plt.subplot(111)
os.chdir("D:/python/COVID19/Output/Total/")
for x in range(len(unique_count)):
    
    date=[]
    Con=[]
    death=[]
    Rec=[]
    dated=[]
    coned=[]
    deathed=[]
    reced=[]
    
    for z in range(len(Observation)):
        if unique_count[x]==Country[z]:
            dated.append(Observation[z])
            coned.append(Confirmed[z])
            deathed.append(Deaths[z])
            reced.append(Recovered[z])
            
    dated_set=set(dated)
    dated_count=(list(dated_set))
    
    datetime_obj=[]
    print(dated_count)
    for i in dated_count:
        datetime_obj.append(parser.parse(i))
    datetime_obj.sort()
    print(datetime_obj)
    dated_count = matplotlib.dates.date2num(datetime_obj)
    dates=dated_count
    
    datetime_obj=[]
    dated_count1=[]
    for i in dated:
        datetime_obj.append(parser.parse(i))
    datetime_obj.sort()
    
    print(datetime_obj)
    dated_count1 = matplotlib.dates.date2num(datetime_obj)
    dated=dated_count1
    
    for z in range(len(dated_count)):
        sum1=0
        sum2=0
        sum3=0
        for y in range(len(dated)):
            if dated_count1[z]==dated[y]:
                sum1=sum1+coned[y]
                sum2=sum2+deathed[y]
                sum3=sum3+reced[y]
                
        Con.append(sum1)
        death.append(sum2)
        Rec.append(sum3)
    print(Con,death,Rec)
    
    

    #matplotlib.pyplot.plot_date(dates, Con,color='blue',linestyle='solid', marker='o')
    ax.plot_date(dates, death,linestyle='solid', marker='x')
    #matplotlib.pyplot.plot_date(dates, Rec,color='green',linestyle='solid', marker='3')
    #matplotlib.pyplot.legend(['Confirmed', 'Deaths', 'Recovered'], loc='upper right')
    
box=ax.get_position()
ax.set_position([box.x0,box.y0,box.width*0.5,box.height*0.9])   
ax.legend(unique_count, ncol=4,loc='center left',bbox_to_anchor=(1,0.5))  
plt.xlim([min(dates),max(tdate)+1])
plt.ylim([0,2500])
plt.xlabel('Date')
plt.ylabel('Number of people')  
plt.title("Worldwide death for Covid-19 outbreak status")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(25.5, 10.5)
fig.savefig('Worldwide death for Covid-19 outbreak status.png', dpi=100)    
plt.show()
    