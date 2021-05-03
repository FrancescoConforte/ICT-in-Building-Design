#Script to choose useful data to performe Energy Signature
#The mean between the 3 zones (corridor, zone1 and zone2) is done
#A file called mydata.csv is built and saved. It's the file to be used to perform Energy Signature
import pandas as pd
import sys
from datetime import timedelta, datetime

#pass as input eplusout_with_year.csv file
df = pd.read_csv(sys.argv[1])

cols = [
    'Date/Time',
    'DistrictHeating:Facility [J](TimeStep)',
    'DistrictCooling:Facility [J](TimeStep)',
    'Electricity:Facility [J](TimeStep)',
    'BLOCK1:CORRIDOR:Zone Mean Air Temperature [C](TimeStep)',
    'BLOCK1:ZONE2:Zone Mean Air Temperature [C](TimeStep)',
    'BLOCK1:ZONE1:Zone Mean Air Temperature [C](TimeStep)',
    'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
    'BLOCK1:ZONE2:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)',
    'BLOCK1:ZONE1:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)',
    'BLOCK1:CORRIDOR:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)'
]

new_names = {
    'Date/Time':'Date_time',
    'DistrictHeating:Facility [J](TimeStep)':'Heat',
    'DistrictCooling:Facility [J](TimeStep)':'Cool',
    'Electricity:Facility [J](TimeStep)':'Elec',
    'BLOCK1:CORRIDOR:Zone Mean Air Temperature [C](TimeStep)':'Temp_in_corr',
    'BLOCK1:ZONE2:Zone Mean Air Temperature [C](TimeStep)':'Temp_in2',
    'BLOCK1:ZONE1:Zone Mean Air Temperature [C](TimeStep)':'Temp_in1',
    'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)':'T_ex',
    'BLOCK1:ZONE2:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)': 'Solar_2',
    'BLOCK1:ZONE1:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)': 'Solar_1',
    'BLOCK1:CORRIDOR:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)': 'Solar_corr'
}

df = pd.DataFrame(df, columns=cols)
df = df.rename(columns=new_names)

ts = []
for i in range(len(df['Date_time'])):
    date, time = df['Date_time'][i].split(' ')
    if (time == '24:00:00'):
        time = '00:00:00'
    timestamp = datetime.strptime(f'{date} {time}', '%m/%d/%Y %H:%M:%S')
    if (time == '00:00:00'):
        timestamp += timedelta(days=1)
    ts.append(timestamp)
df['Date_time'] = pd.to_datetime(ts, format='%m/%d/%Y %H:%M:%S').astype(int) // 10**9

df['T_in'] = df[['Temp_in1', 'Temp_in2', 'Temp_in_corr']].mean(1)
df = df.drop(columns=['Temp_in1', 'Temp_in2', 'Temp_in_corr'])

df['Heat']= df['Heat'] / 3.6e6
df['Cool']= df['Cool'] / 3.6e6


df['Solar'] = df[['Solar_2', 'Solar_1', 'Solar_corr']].mean(1)
df = df.drop(columns=['Solar_2', 'Solar_1', 'Solar_corr'])

df.to_csv('my_data.csv', index=False)
