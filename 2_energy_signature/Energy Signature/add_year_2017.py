#Script to add the year 2017 to the eplusout.csv
import pandas as pd

df = pd.read_csv('eplusout.csv')

for i in range(df.shape[0]):
    j, date, k, time = df['Date/Time'][i].split(' ')
    df['Date/Time'][i] = f'{date}/2017 {time}'

df.to_csv('eplusout_2017_with_year.csv', index=False)
