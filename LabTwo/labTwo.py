# Caroline Smith
# COSC311 Lab 2
# 10 March 2023

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def trim(data):
  trim = lambda x: x.strip() if type(x) is str else x
  return data.applymap(trim)

if __name__ == "__main__": 

    plt.style.use('ggplot')

    # read in the CSV as a dataframe 

    bejaia_df = trim(pd.read_csv('Bejaia_Region.csv', skipinitialspace=True))
    sidi_df = trim(pd.read_csv('Sidi-Bel_Abbes_Region.csv', skipinitialspace=True))

    # print the .info() of each dataframe

    bejaia_df.info()
    sidi_df.info()
    print("\n")

    # print the .describe() information of each dataframe

    print(bejaia_df.describe())
    print(sidi_df.describe())
    print("\n")

    # show the unique values of the wind speed

    bejaia_unique_windspeeds = bejaia_df['Ws'].unique()
    sidi_unique_windspeeds = sidi_df['Ws'].unique()

    print(f'The Bejaia unique windspeeds are: ', bejaia_unique_windspeeds)
    print(f'The Sidi unique windspeeds are: ', sidi_unique_windspeeds)
    print("\n")

    # show the number of data entries 

    bejaia_df_samples = len(bejaia_df)
    sidi_df_samples = len(sidi_df)

    print(f'There are {bejaia_df_samples} samples in the Bejaia information.')
    print(f'There are {sidi_df_samples} samples in the Sidi information.')
    print('\n')

    # drawing a line graph to show the temperature change in Bejaia region

    bejaia_df.plot(y='Temperature')
    plt.xlabel("Days")
    plt.ylabel("Temperature")
    plt.title("Days vs. Temperature Change in Bejaia Region")
    plt.show()
    print("\n")

    # drawing a scatterplot to show the relationship between temperature and 
    # fire weather index in Sidi region

    sidi_df.plot.scatter(x='Temperature', y='FWI')
    plt.xlabel("Temperature")
    plt.ylabel("Fire Weather Index")
    plt.title("Temperature vs. Fire Weather Index in Sidi-Bel Abbes Region")
    plt.show()
    print('\n')

    # histogram to show average relative humidity for each month in Bejaia region 
    
    avg_june = sum(bejaia_df.loc[bejaia_df['month'] == 6]['RH'])/30
    avg_july = sum(bejaia_df.loc[bejaia_df['month'] == 7]['RH'])/31
    avg_august = sum(bejaia_df.loc[bejaia_df['month'] == 8]['RH'])/31
    avg_sept = sum(bejaia_df.loc[bejaia_df['month'] == 9]['RH'])/30

    df = pd.DataFrame({'average_RH': [avg_june, avg_july, avg_august, avg_sept]}, index = ['june', 'july', 'august','september'])

    df.hist(bins=30)
    plt.xlabel("Relative Humidity")
    plt.ylabel("Number of Months")
    plt.title("Relative Humidity by Month in Bejaia Region")
    plt.show()
    print('\n')

    # bar graph to show the maximum anount of rain in a day per month for Bejaia
    # region

    rain = bejaia_df.groupby('month')['Rain '].max()

    df_rain = rain.to_frame()
    df_rain.plot.bar(legend=False)
    plt.xlabel('Month')
    plt.ylabel('Max Rain')
    plt.title('Max Rain vs. Month in Bejaia Region')
    plt.show()
    print('\n')

    # histogram to show the wind speed distribution in 5 bins for Sidi

    sidi_Ws = sidi_df.loc[sidi_df['month'] == 6]['Ws']
    df_sidi_Ws = sidi_Ws.to_frame()
    df_sidi_Ws.hist(bins=5)
    plt.xlabel("Wind Speed")
    plt.ylabel("Number of Days")
    plt.title("Wind Speed In the Sidi-Bel Abbes Region")
    plt.show()
    print('\n')

    # line graph to show the correlation between temperature and relative humidity for Sidi in July

    sidi_temp_rh = pd.DataFrame(sidi_df.loc[sidi_df['month'] == 7])
    july_rh = sorted(sidi_temp_rh['RH'])
    july_temp = sorted(sidi_temp_rh['Temperature'])

    plt.plot(july_temp, july_rh)
    plt.xlabel("Temperature")
    plt.ylabel("Relative Humidity")
    plt.title("Temperature vs. Relative Humidity in Sidi-Bel Abbes Region")
    plt.show()
    print('\n')

    # a  bar  figure  to  show  the  distribution  of  Relative  Humidity  (RH)  for  the  "Bejaia 
    # Region Dataset"

    bejaia_df.loc[:,'RH_decile'] = pd.cut(bejaia_df['RH'], bins=range(20, 110, 10), labels=[f'{i}s' for i in range(20, 100, 10)])
    rh_counts = bejaia_df.groupby('RH_decile')['day'].count()
    rh_counts.plot(kind='bar')
    plt.xlabel('RH Decile') 
    plt.ylabel('Number of Days')
    plt.title('Distribution of Relative Humidity in Bejaia Region')
    plt.show()
    print('\n')

    # average temperature for each month when there is "no fire" and there is "fire" for 
    # the "Bejaia Region Dataset" 

    bejaia_fire = bejaia_df.loc[bejaia_df['Classes  '] == "fire"]
    bejaia_not_fire = bejaia_df.loc[bejaia_df['Classes  '] == "not fire"]

    fire_avg = (bejaia_fire.groupby('month')['Temperature'].mean()).to_frame()
    not_fire_avg = (bejaia_not_fire.groupby('month')['Temperature'].mean()).to_frame()

    combined_df = fire_avg.merge(not_fire_avg, on='month', how='outer', suffixes=('_Fire', '_NotFIre'))
    combined_df.plot.bar()
    plt.xlabel('Month')
    plt.ylabel('Temperature')
    plt.title('Average Temperature Per Month on Fire vs. Not Fire Day in Bejaia Region')
    plt.show()
    print('\n')
