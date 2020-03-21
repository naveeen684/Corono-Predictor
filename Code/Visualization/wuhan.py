import pandas as pd
import matplotlib as plt1
import os
import numpy as np
import datetime as dt
from datetime import timedelta
from dateutil import parser
import datetime

df = pd.read_csv('D:/python/COVID19/data/COVID19_line_list_data.csv')
#print(df)
#df.info()

df=df.fillna(df.mean())

id = df["id"].astype(int)

report_date = []
r=df["reporting date"].astype(str)

for i in r:
    if(i=='nan'):
        report_date.append(datetime.datetime.today())
    else:
        report_date.append(parser.parse(i))
entry_dates = plt1.dates.date2num(report_date)

state = df["location"].astype(str)
country= df["country"].astype(str)
gen= df["gender"].astype(str)
gender=[1 if i=='male' else 0 if i=='female' else 0.5 for i in gen]

age=df["age"].astype(int)
symptom=df["symptom"].astype(str)
symptom=['fever, cough, breathing problem' if i=='nan' else i for i in symptom]

symp_on=[]
r=df["symptom_onset"].astype(str)
for i in range(len(r)):
    if(r[i]=='nan'):
        symp_on.append(report_date[i])
    else:
        symp_on.append(parser.parse(r[i]))
symp_dates = plt1.dates.date2num(symp_on)   
    
hosp_vis=[]
r=df["hosp_visit_date"].astype(str)
for i in range(len(r)):
    if(r[i]=='nan'):
        hosp_vis.append(symp_on[i])
    else:
        hosp_vis.append(parser.parse(r[i]))
hosp_visit = plt1.dates.date2num(hosp_vis)  

exp_s=[]
r=df["exposure_start"].astype(str)
for i in range(len(r)):
    if(r[i]=='nan'):
        exp_s.append(symp_on[i]-timedelta(days=4))
    else:
        exp_s.append(parser.parse(r[i]))
exp_strt = plt1.dates.date2num(exp_s)   

exp_e=[]
r=df["exposure_start"].astype(str)
for i in range(len(r)):
    if(r[i]=='nan'):
        exp_e.append(symp_on[i]+timedelta(days=1))
    else:
        exp_e.append(parser.parse(r[i]))
exp_end = plt1.dates.date2num(exp_e)   

v_wuhan=df["visiting Wuhan"].astype(int)
f_wuhan=df["from Wuhan"].astype(int)

death=df["death"].astype(str)
death=[0 if i=='0' else 1 for i in death]

rec=df["recovered"].astype(str)
rec=[0 if i=='0' else 1 for i in rec]

state_date=[]
r=df["death"].astype(str)
state_date=[datetime.datetime.today()]*len(r)
label=[0]*len(r)
for i in range(len(r)):
    if(r[i]!='0'):
        state_date[i]=parser.parse(r[i])

r=df["recovered"].astype(str)

for i in range(len(r)):
    if(r[i]!='0'):
        state_date[i]=parser.parse(r[i])

sdate = plt1.dates.date2num(state_date)   

exp_time=[]
for i in range(len(exp_s)):
    delta=exp_e[i]-exp_s[i]
    exp_time.append(delta.days)

symp_exps=[]
for i in range(len(exp_s)):
    delta=symp_on[i]-exp_s[i]
    symp_exps.append(delta.days)
    
symp_expe=[]
for i in range(len(exp_s)):
    delta=symp_on[i]-exp_e[i]
    symp_expe.append(delta.days)
    
hosp_symp=[]
for i in range(len(exp_s)):
    delta=hosp_vis[i]-symp_on[i]
    hosp_symp.append(delta.days)

hosp_expon=[]
for i in range(len(exp_s)):
    delta=hosp_vis[i]-exp_s[i]
    hosp_expon.append(delta.days)
    
death_expon=[]
rec_expon=[]
death_hosp=[]
rec_hosp=[]
death_symp=[]


for i in range(len(exp_s)):
    if death[i]==1:
        delta=state_date[i]-exp_s[i]
        death_expon.append(delta.days)
        delta=state_date[i]-hosp_vis[i]
        death_hosp.append(delta.days)
        delta=state_date[i]-symp_on[i]
        death_symp.append(delta.days)
        label[i]=2
        
    else:
        death_expon.append(0)
        death_hosp.append(0)
        death_symp.append(0)
        
        
for i in range(len(exp_s)):
    if rec[i]==1:
        delta=state_date[i]-exp_s[i]
        rec_expon.append(delta.days)
        delta=state_date[i]-hosp_vis[i]
        rec_hosp.append(delta.days)
        delta=state_date[i]-symp_on[i]
        label[i]=1
        
    else:
        rec_expon.append(0)
        rec_hosp.append(0)
      
'''
print(report_date) 
#print(entry_dates)
print(symp_on)
#print(symp_dates)
print(hosp_vis)
#print(hosp_visit)
print(exp_s)
#print(exp_strt)
print(exp_e)
print(exp_end)
print(state_date) #
#print(sdate)
print(label)
print(state)
print(country) 
'''

data = pd.DataFrame({'q': state})
data['q'] = data['q'].astype('category')
data['q'] = data['q'].cat.reorder_categories(set(state), ordered=True)
data['q'] = data['q'].cat.codes
state_num=data['q']



data = pd.DataFrame({'q': country})
data['q'] = data['q'].astype('category')
data['q'] = data['q'].cat.reorder_categories(set(country), ordered=True)
data['q'] = data['q'].cat.codes
country_num=data['q']

data = pd.DataFrame({'q': symptom})
data['q'] = data['q'].astype('category')
data['q'] = data['q'].cat.reorder_categories(set(symptom), ordered=True)
data['q'] = data['q'].cat.codes
symptom_num=data['q']











print("state_num",state_num)
print("country_num",country_num)
print("symptom_num",symptom_num)
print("gender",gender) 
print("age",age)
print("v_wuhan",v_wuhan)
print("f_wuhan",f_wuhan)
print("death",death)
print("rec",rec)
print("exp_time",exp_time)
print("symp_exps",symp_exps)
print("symp_expe",symp_expe)
print("hosp_symp",hosp_symp)
print("hosp_expon",hosp_expon)
print("death_expon",death_expon)
print("rec_expon",rec_expon)
print("death_hosp",death_hosp)
print("rec_hosp",rec_hosp)
print("death_symp",death_symp)
print("label",label)


import seaborn as sns 
df=pd.DataFrame({'State':state_num,'Country':country_num,'Gender':gender,'Age':age,
                 'Symptom':symptom_num,'visit_wuhan':v_wuhan,'from_wuhan':f_wuhan,
                 'Exposure_per':exp_time,'Symp_expstrt':symp_exps,'Symp_expend':symp_expe,
                 'Hosp_symp':hosp_symp,'Hosp_expstrt':hosp_symp,'Death_expstrt':death_expon,
                 'Death_hosp':death_hosp,'Death_symp':death_symp,'Rec_expstrt':rec_hosp,
                 'Rec_hosp':rec_hosp})


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(label, bins=30)
plt1.pyplot.show()

correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt1.pyplot.show()



'''
plt1.pyplot.subplot(1,2,1)
plt1.pyplot.scatter(df['Death_hosp'],label,color='BLACK') 
plt1.pyplot.ylabel('label')
plt1.pyplot.xlabel('death_hosp')


plt1.pyplot.subplot(1,2,2)
plt1.pyplot.scatter(df['Death_symp'],label,color='RED') 
plt1.pyplot.ylabel('label')
plt1.pyplot.xlabel('death_symp')
'''

