
import pandas as pd
import matplotlib as plt1
import os
import numpy as np

os.chdir("D:/python/COVID19/data/")
df = pd.read_csv('covid_19_data.csv')
#print(df)
#df.info()

import datetime as dt
from dateutil import parser



State = df["Province/State"].astype(str)
Last_Update=df["Last Update"]
Confirmed=df["Confirmed"]
Deaths=df["Deaths"]
Recovered=df["Recovered"]
Observation=df["ObservationDate"]

state_set=set(State)
unique_state=(list(state_set))
unique_state.sort()
#print(unique_state)
os.chdir("D:/python/COVID19/Output/State_wise/")
    
for x in range(len(unique_state)):   
    x3=x   
    if unique_state[x] != 'nan':
        date=[]
        Con=[]
        death=[]
        Rec=[]
        for y in range(len(State)):
            if unique_state[x]==State[y]:
                date.append(Observation[y])
                Con.append(Confirmed[y])
                death.append(Deaths[y])
                Rec.append(Recovered[y])
        
        datetime_obj=[]
        for i in date:
            datetime_obj.append(parser.parse(i))
        #print(datetime_obj)
        dates = plt1.dates.date2num(datetime_obj)
        #print(dates)
        plt1.pyplot.plot_date(dates, Con,color='blue',linestyle='solid', marker='o')
        plt1.pyplot.plot_date(dates, death,color='red',linestyle='solid', marker='x')
        plt1.pyplot.plot_date(dates, Rec,color='green',linestyle='solid', marker='3')
        plt1.pyplot.legend(['Confirmed', 'Deaths', 'Recovered'], loc='upper right')
        plt1.pyplot.xlabel('Date')
        plt1.pyplot.ylabel('Number of people')
        plt1.pyplot.title(unique_state[x]+" Covid-19 outbreak status")
        plt1.pyplot.xlim([min(dates),max(dates)+2])
        plt1.pyplot.ylim([0,max(Con)+50])
        fig = plt1.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(unique_state[x]+' Covid-19 outbreak status.png', dpi=100)
        plt1.pyplot.show()
        
        if len(dates)>8: 
            
            x = np.asarray(dates)
            y= np.asarray(Con)
            x=x.reshape(-1,1)
            y=y.reshape(-1, 1)
            
            
            from sklearn.linear_model import LinearRegression 
            L=LinearRegression()
            
            
            metrain=[]
            metest=[]
            
            from sklearn.metrics import mean_squared_error,accuracy_score
            from sklearn.preprocessing import PolynomialFeatures
            
            from sklearn.model_selection import train_test_split as T
            x_train,x_test,y_train,y_test=T(x,y,test_size=1/3,random_state=None) 
            
            
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
        
            df=pd.DataFrame({'Date':date,'Population of Confirmed':y_list})
            
            print(df)  
            df.to_csv("D:\python\COVID19\Output\Dataframes\Confirmed\State\{}_Confirmed_pred50.csv".format(unique_state[x3]))            
            
            x = np.asarray(dates)
            y= np.asarray(death)
            x=x.reshape(-1,1)
            y=y.reshape(-1, 1)
            
            
            from sklearn.linear_model import LinearRegression 
            L=LinearRegression()
            
            
            metrain=[]
            metest=[]
            
            from sklearn.metrics import mean_squared_error,accuracy_score
            from sklearn.preprocessing import PolynomialFeatures
            
            from sklearn.model_selection import train_test_split as T
            x_train,x_test,y_train,y_test=T(x,y,test_size=1/3,random_state=None) 
            
            
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
        
            df=pd.DataFrame({'Date':date,'Population of Death':y_list})
            
            print(df)
            df.to_csv("D:\python\COVID19\Output\Dataframes\Death\State\{}_Death_pred50.csv".format(unique_state[x3]))
            
            x = np.asarray(dates)
            y= np.asarray(Rec)
            x=x.reshape(-1,1)
            y=y.reshape(-1, 1)
            
            
            from sklearn.linear_model import LinearRegression 
            L=LinearRegression()
            
            
            metrain=[]
            metest=[]
            
            from sklearn.metrics import mean_squared_error,accuracy_score
            from sklearn.preprocessing import PolynomialFeatures
            
            from sklearn.model_selection import train_test_split as T
            x_train,x_test,y_train,y_test=T(x,y,test_size=1/3,random_state=None) 
            
            
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
        
            df=pd.DataFrame({'Date':date,'Population of Recovered':y_list})
        
            print(df)
            df.to_csv("D:\python\COVID19\Output\Dataframes\Recovered\State\{}_Recovery_pred50.csv".format(unique_state[x3]))

            
            
