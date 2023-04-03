import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import stats 

def trim(data):
  trim = lambda x: x.strip() if type(x) is str else x
  return data.applymap(trim)

if __name__ == "__main__": 

    plt.style.use('ggplot')

    # read in the CSV as a dataframe 

    bejaia_df = trim(pd.read_csv('Bejaia_Region.csv', skipinitialspace=True))
    sidi_df = trim(pd.read_csv('Sidi-Bel_Abbes_Region.csv', skipinitialspace=True))

    bejaia_fire = bejaia_df.loc[bejaia_df['Classes  '] == "fire"]
    bejaia_not_fire = bejaia_df.loc[bejaia_df['Classes  '] == "not fire"]

    temp_fire_mean = stats.mean(bejaia_fire['Temperature'].to_numpy())
    temp_not_fire_mean = stats.mean(bejaia_not_fire['Temperature'].to_numpy())
    RH_fire_mean = stats.mean(bejaia_fire['RH'].to_numpy())
    RH_not_fire_mean = stats.mean(bejaia_not_fire['RH'].to_numpy())
    windspeed_fire_mean = stats.mean(bejaia_fire['Ws'].to_numpy())
    windspeed_not_fire_mean = stats.mean(bejaia_not_fire['Ws'].to_numpy())
    rain_fire_mean = stats.mean(bejaia_fire['Rain '].to_numpy())
    rain_not_fire_mean = stats.mean(bejaia_not_fire['Rain '].to_numpy())

    df_means = pd.DataFrame(zip([temp_fire_mean, RH_fire_mean, windspeed_fire_mean, rain_fire_mean], [temp_not_fire_mean, RH_not_fire_mean, windspeed_not_fire_mean, rain_not_fire_mean]), columns=['fire','no fire'], index=['temperature', 'RH', 'windspeed', 'rain'])

    df_means.plot.bar()
    plt.xlabel('Columns')
    plt.ylabel('Mean')
    plt.title('Means of Fire and Not Fire Classes in Bejaia Dataset')
    plt.show()
    print()

    print("Fire classes: ") 
    print(f'Temperature: {temp_fire_mean:.2f}.')
    print(f'RH: {RH_fire_mean:.2f}.')
    print(f'Windspeed: {windspeed_fire_mean:.2f}.')
    print(f'Temperature: {rain_fire_mean:.2f}.')

    print("Not fire classes: ") 
    print(f'Temperature: {temp_not_fire_mean:.2f}.')
    print(f'RH: {RH_not_fire_mean:.2f}.')
    print(f'Windspeed: {windspeed_not_fire_mean:.2f}.')
    print(f'Temperature: {rain_not_fire_mean:.2f}.')

    # task 2

    FFMC_median = stats.median(sidi_df['FFMC'])
    DMC_median = stats.median(sidi_df['DMC'])
    DC_median = stats.median(sidi_df['DC'])
    ISI_median = stats.median(sidi_df['ISI'])

    df_medians = pd.DataFrame({'medians':[FFMC_median, DMC_median, DC_median, ISI_median]}, index=['FFMC', 'DMC', 'DC', 'ISI'])
    df_medians.plot.bar()
    plt.xlabel('Columns')
    plt.ylabel('Medians')
    plt.title('Medians for Columns in the Sidi Dataset')
    plt.show()
    print()

    # task 3
    # "Temperature", "RH", "Ws" and "Rain",
    # 25-percent, 60-percent, and 75-percent

    temp_25, temp_60, temp_75 = stats.quantile(bejaia_df['Temperature'], .25), stats.quantile(bejaia_df['Temperature'], .60), stats.quantile(bejaia_df['Temperature'], .75)
    print(f'25% quartile for temperature: {temp_25}.')
    print(f'60% quartile for temperature: {temp_60}.')
    print(f'75% quartile for temperature: {temp_75}.')
    print()

    RH_25, RH_60, RH_75 = stats.quantile(bejaia_df['RH'], .25), stats.quantile(bejaia_df['RH'], .60), stats.quantile(bejaia_df['RH'], .75)
    print(f'25% quartile for RH: {RH_25}.')
    print(f'60% quartile for RH: {RH_60}.')
    print(f'75% quartile for RH: {RH_75}.')
    print()

    WS_25, WS_60, WS_75 = stats.quantile(bejaia_df['Ws'], .25), stats.quantile(bejaia_df['Ws'], .60), stats.quantile(bejaia_df['Ws'], .75)
    print(f'25% quartile for WS: {WS_25}.')
    print(f'60% quartile for WS: {WS_60}.')
    print(f'75% quartile for WS: {WS_75}.')
    print()

    rain_25, rain_60, rain_75 = stats.quantile(bejaia_df['Rain '], .25), stats.quantile(bejaia_df['Rain '], .60), stats.quantile(bejaia_df['Rain '], .75)
    print(f'25% quartile for rain: {rain_25}.')
    print(f'60% quartile for rain: {rain_60}.')
    print(f'75% quartile for rain: {rain_75}.')
    print()

    # task 4 
    # "Temperature", " Rain", "BUI" and "FWI"
    # standard deviation

    temp_std = stats.std(bejaia_df['Temperature'])
    rain_std = stats.std(bejaia_df['Rain '])
    BUI_std = stats.std(bejaia_df['BUI'])
    FWI_std = stats.std(bejaia_df['FWI'])

    print(f'Temperature standard deviation: {temp_std:.2f}.')
    print(f'Rain standard deviation: {rain_std:.2f}.')
    print(f'BUI standard deviation: {BUI_std:.2f}.')
    print(f'FWI standard deviation: {FWI_std:.2f}.')
    print()

    # task 5
    # correlation between RH and Temperature", "Ws", "Rain","FFMC", "DMC", "DC", "ISI", "BUI" and "FWI"

    cor_RH_temperature = stats.correlation(bejaia_df['RH'], bejaia_df['Temperature'])
    cor_RH_WS = stats.correlation(bejaia_df['RH'], bejaia_df['Ws'])
    cor_RH_rain = stats.correlation(bejaia_df['RH'], bejaia_df['Rain '])
    cor_RH_FFMC = stats.correlation(bejaia_df['RH'], bejaia_df['FFMC'])
    cor_RH_DMC = stats.correlation(bejaia_df['RH'], bejaia_df['DMC'])
    cor_RH_DC = stats.correlation(bejaia_df['RH'], bejaia_df['DC'])
    cor_RH_ISI = stats.correlation(bejaia_df['RH'], bejaia_df['ISI'])
    cor_RH_BUI = stats.correlation(bejaia_df['RH'], bejaia_df['BUI'])
    cor_RH_FWI = stats.correlation(bejaia_df['RH'], bejaia_df['FWI'])


    print(f'Correlation between RH and temperature {cor_RH_temperature}.')
    print(f'Correlation between RH and windspeed {cor_RH_WS}.')
    print(f'Correlation between RH and rain {cor_RH_rain}.')
    print(f'Correlation between RH and FFMC {cor_RH_FFMC}.')
    print(f'Correlation between RH and DMC {cor_RH_DMC}.')
    print(f'Correlation between RH and DC {cor_RH_DC}.')
    print(f'Correlation between RH and ISI {cor_RH_ISI}.')
    print(f'Correlation between RH and BUI {cor_RH_BUI}.')
    print(f'Correlation between RH and FWI {cor_RH_FWI}.')

    # task 6
    # finding the attributes that have the strongest correlation for fire and not fire 

    cor_fire_temperature = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['Temperature'])
    cor_fire_WS = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['Ws'])
    cor_fire_rain = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['Rain '])
    cor_fire_FFMC = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['FFMC'])
    cor_fire_DMC = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['DMC'])
    cor_fire_DC = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['DC'])
    cor_fire_ISI = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['ISI'])
    cor_fire_BUI = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['BUI'])
    cor_fire_FWI = stats.correlation(bejaia_df['Classes  '] == 'fire', bejaia_df['FWI'])

    cor_not_fire_temperature = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['Temperature'])
    cor_not_fire_WS = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['Ws'])
    cor_not_fire_rain = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['Rain '])
    cor_not_fire_FFMC = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['FFMC'])
    cor_not_fire_DMC = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['DMC'])
    cor_not_fire_DC = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['DC'])
    cor_not_fire_ISI = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['ISI'])
    cor_not_fire_BUI = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['BUI'])
    cor_not_fire_FWI = stats.correlation(bejaia_df['Classes  '] == 'not fire', bejaia_df['FWI'])

    print(f'Correlation between fire, not fire and temperature: (fire) {cor_fire_temperature} (not fire) {cor_not_fire_temperature}.')
    print(f'Correlation between fire, not fire and windspeed: (fire) {cor_fire_WS} (not fire) {cor_not_fire_WS}.')
    print(f'Correlation between fire, not fire and rain: (fire) {cor_fire_rain} (not fire) {cor_not_fire_rain}.')
    print(f'Correlation between fire, not fire and FFMC: (fire) {cor_fire_FFMC} (not fire) {cor_not_fire_FFMC}.')
    print(f'Correlation between fire, not fire and DMC: (fire) {cor_fire_DMC} (not fire) {cor_not_fire_DMC}.')
    print(f'Correlation between fire, not fire and DC: (fire) {cor_fire_DC} (not fire) {cor_not_fire_DC}.')
    print(f'Correlation between fire, not fire and ISI: (fire) {cor_fire_ISI} (not fire) {cor_not_fire_ISI}.')
    print(f'Correlation between fire, not fire and BUI: (fire) {cor_fire_BUI} (not fire) {cor_not_fire_BUI}.')
    print(f'Correlation between fire, not fire and FWI: (fire) {cor_fire_FWI} (not fire) {cor_not_fire_FWI}.')