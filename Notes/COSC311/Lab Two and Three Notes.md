### Lab Two 

##### Question One

Use the pandas package to: 
1. Use info() function to shoow the collumn information of each of the datasets separately 
2. Use the describe() function to show the statistics of the two datasets separately 
3. Show the unique values of the wind speed column of thee two datasets separately 
4. Count how many samples are in each of the two datasets separately 

**Read in data** 

```python3 
BejaiaData = pd.read_csv('Bejaia_Region.csv')
SidiData = pd.read_csv('Sidi-Bel_Abbes_Region.csv')
```

**Using the .info() function** 

```python3 
print(BejaiaData.info())
print(SidiData.info())
```

**Using the .describe() function** 

```python3 
print(BejaiaData.describe())
print(SidiData.describe())
```

**Finding the unique values of a specific column in a dataframe** 

```python3 
BejaiaData_Ws = BejaiaData[' Ws'].unique()
SidiData_Ws = SidiData[' Ws'].unique()
```

**Counting how many samples are in the dataframe** 

```python3 
BejaiaData_samples = len(BejaiaData)
SidiData_samples = len(SidiData)
```

##### Question Two

Draw a line graph to show the temperature changing with time for the "Bejaia Region Dataset"

**Getting the data** 

```python3
# slice the dataframe to just get the Temperature column 
Temp_Bejaia = BejaiaData['Temperature']

# get the number of days by finding the range based on the days in the temperature slice 
NumOfDay = np.arange(len(Temp_Bejaia))
```

**Plotting the data** 

```python3
plt.plot(NumOfDay, Temp_Bejaia, color='green', marker='o',
	linestyle='dashed', linewidth=1, markersize=4)
plt.title("Bejaia Region Temperature over time")
plt.ylabel("Temperature in C")
```

**Creating the x-labels** 

```python3 
Xlabel_month_loc = [i for i in NumOfDay if BejaiaData['day'][i] == 1]
Xlabels = ["Jun 2012", "Jul 2012", "Aug 2012", "Sep 2012"]
plt.xticks(Xlabel_month_loc, Xlabels)
plt.show()
```
![[Screenshot 2023-04-18 at 10.15.08.png]]

##### Question Three

Draw a scatterplot figure to show the relationship between the temperature and the fire weather index for the "Sidi-Bel Abbes Region Dataset"

```python3
plt.scatter(SidiData['Temperature'], SidiData['FWI'], marker='^')
plt.title('Relationship between Temperature and FWI in Sidi-Bel Abbes Region')
plt.xlabel('Temperature in C')
plt.ylabel('Fire Weather Index')
plt.show()
```
![[Screenshot 2023-04-18 at 10.14.53.png]]

##### Question Four 

Draw a histogram to show the average relative humidity for each month in the "Bejaia Region Dataset"

**Find the average of each month** 

```python3
# group by month and then take the mean 
RH_avgPerMonth = BejaiaData.groupby('month')[' RH'].mean()
```

**Plot the data** 

```python3
# you can use the range() function to get an iterable 
plt.bar(range(len(RH_avgPerMonth )), RH_avgPerMonth,
	width = 0.7, align='center')
plt.title("Average RH for each month in Bejaia Region")
plt.ylabel("Average Relative Humidity")
Xlabels = ["Jun 2012", "Jul 2012", "Aug 2012", "Sep 2012"]
plt.xticks(range(len(RH_avgPerMonth)), Xlabels)
plt.show()
```
![[Screenshot 2023-04-18 at 10.26.12.png]]

##### Question Five

Draw a bar figure to show the maximum rain amount in a day for each month for the "Bejaia Region Dataset"

**Get the max rain per month**

```python3 
# group by month and then index into the Rain column 
Rain_MaxPerMonth = BejaiaData.groupby('month')['Rain '].max()
```

**Plot data** 

```python3
plt.bar(range(len(Rain_MaxPerMonth)), Rain_MaxPerMonth ,
	width = 0.7, align='center')
plt.title("Max rain amount for each month in Bejaia Region")
plt.ylabel("Max rain amount")
Xlabels = ["Jun 2012", "Jul 2012", "Aug 2012", "Sep 2012"]
plt.xticks(range(len(Rain_MaxPerMonth)), Xlabels)
plt.show()
```
![[Screenshot 2023-04-18 at 10.32.43.png]]

##### Question Six

Draw a histogram to show the wind speed distribution in 5 bins for the "Sidi-Bel Abbes Region Dataset" in June, 2012

**Slice the data and plot the histogram**

```python3 
SidiData[' Ws'][SidiData['month'] == 6].hist(bins = 5, grid = True, figsize = (5,3))
plt.title("Wind speed distribution for Sidi-Bel in June, 2012")
plt.ylabel("# of days")
plt.xlabel("Wind speed")
plt.show()
```
![[Screenshot 2023-04-18 at 10.38.04.png]]

##### Question Seven 

Draw a line figure to show the correlation between teemperature and relative humidity for the "Sidi-Bel Abbes Region Dataset" in July, 2012

**Slice the data** 

```python3 
# slice to get Temperature and RH, get just the data froom July, then group by the mean temperature
Temp_avgRH = SidiData[['Temperature', ' RH']][SidiData['month'] == 7]\
.groupby('Temperature').mean()
```

**Plot the data** 

```python3 
Temp_avgRH.plot.line()
plt.title("Correlation between temperature and average RH")
plt.xlabel("Temperature in C")
plt.ylabel("Average relative humidity")
plt.show()
```

![[Screenshot 2023-04-18 at 10.43.13.png]]

##### Question EIght

Draw a bar figure to show the distribution of relative humidity for the "Bejaia Region Dataset". The x-axis is the declinee of the RH (20s, 30s, ..., 90s) and the y-axis is the number of days. 

**Creates the declie** 

```python3 
from collections import Counter
RH_decile = Counter([(rh // 10 * 10) for rh in BejaiaData[' RH']])
```

**Plot the data**

```python3 
plt.bar([x + 5 for x in RH_decile.keys()], RH_decile.values(), 10, edgecolor=(0, 0, 0))
plt.xticks(sorted(RH_decile.keys()))
plt.xlabel("Relative Humidity Decile")
plt.ylabel("# of days")
plt.title("Distribution of Relative Humitity in the Bejaia Region")
plt.show()
```

![[Screenshot 2023-04-18 at 10.55.15.png]]

##### Question Nine

Draw a figure to show the average temperature for each month where there is "no fire" and "fire" for the "Bejaia Region Dataset" 

**Slice the data** 

```python3 
fire = BejaiaData[BejaiaData['Classes '].str.startswith('fire')].\ groupby('month')['Temperature'].mean()
notFire = BejaiaData[BejaiaData['Classes '].str.startswith('not fire')].\
groupby('month')['Temperature'].mean()
```

**Plot the data** 

```python3 
Xloc = np.arange(len(BejaiaData['month'].unique()))
Xlabels = ["Jun 2012", "Jul 2012", "Aug 2012", "Sep 2012"]

plt.bar(Xloc - 0.15, fire, 0.3, label = 'Fire', edgecolor = (0,0,0), align = 'center', color = 'blue')
plt.bar(Xloc + 0.15, notFire, 0.3, label = 'Not fire', edgecolor = (0,0,0), align = 'center', color = 'gray')
plt.title('Average temperature for each month')
plt.ylabel('Average temperature')
plt.xticks(Xloc, Xlabels)
plt.legend()
plt.show()
```

![[Screenshot 2023-04-18 at 11.36.42.png]]
  
### Lab Three

##### Question One

Using the "Bejaia Region Dataset":  
1. Calculate and show the mean values of four attributes ("Temperature", "RH" (Relative Humidity), "Ws" (Wind speed) and "Rain", respectively) for each class (i.e. for “not fire” and “fire”, respectively). 
2. Draw a bar figure to show the mean values of these attributes for each class. 
3. Describe your observations in the above figure.

**Read in the data** 

```python3 
BejaiaData = pd.read_csv('Bejaia_Region.csv')
SidiData = pd.read_csv('Sidi-Bel_Abbes_Region.csv')
```


**Find the mean temperatures for the fire and not fire classes** 

```python3
fire = BejaiaData[BejaiaData['Classes '].str.startswith('fire')]
print('Mean temp for fire:', fire['Temperature'].mean())
print('Mean RH for fire:', fire[' RH'].mean())
print('Mean Ws for fire:', fire[' Ws'].mean())
print('Mean Rain for fire:', fire['Rain '].mean())

print()

notFire = BejaiaData[BejaiaData['Classes '].str.startswith('not fire')]
print('Mean temp for not fire:', notFire['Temperature'].mean())
print('Mean RH for not fire:', notFire[' RH'].mean())
print('Mean Ws for not fire:', notFire[' Ws'].mean())
print('Mean Rain for not fire:', notFire['Rain '].mean())
```

##### Question Two

Using the "Sidi-Bel Abbes Region Dataset", calculate and show the median values of four attributes ("FFMC", "DMC", "DC" and "ISI", respectively)

**Finding the medians** 

```python3
print('Median FFMC: ',stats.median(SidiData['FFMC']))
print('Median DMC: ',stats.median(SidiData['DMC']))
print('Median DC: ',stats.median(SidiData['DC']))
print('Median ISI: ',stats.median(SidiData['ISI']))
```

##### Question Three

Using the "Bejaia Region Dataset", calculate and show the 25-percent, 60-percent, and 75-percent quantiles of four attributes ("Temperature", "RH", "Ws" and "Rain", respectively)

```python3
print('25% temperature: ', stats.quantile(BejaiaData['Temperature'], 0.25))
print('60% temperature: ', stats.quantile(BejaiaData['Temperature'], 0.60))
print('75% temperature: ', stats.quantile(BejaiaData['Temperature'], 0.75))

print()

print('25% RH: ', stats.quantile(BejaiaData[' RH'], 0.25))
print('60% RH: ', stats.quantile(BejaiaData[' RH'], 0.60))
print('75% RH: ', stats.quantile(BejaiaData[' RH'], 0.75))

print()

print('25% Ws: ', stats.quantile(BejaiaData[' Ws'], 0.25))
print('60% Ws: ', stats.quantile(BejaiaData[' Ws'], 0.60))
print('75% Ws: ', stats.quantile(BejaiaData[' Ws'], 0.75))

print()

print('25% rain: ', stats.quantile(BejaiaData['Rain '], 0.25))
print('60% rain: ', stats.quantile(BejaiaData['Rain '], 0.60))
print('75% rain: ', stats.quantile(BejaiaData['Rain '], 0.75))

print()
```

##### Question Four 

Using the "Sidi-Bel Abbes Region Dataset", calculate and show the standard deviation values of four attributes ("Temperature", " Rain", "BUI" and "FWI", respectively)

```python3
print('Temperature standard deviation:', stats.std(SidiData['Temperature']))
print('Rain standard deviation:', stats.std(SidiData['Rain ']))
print('BUI standard deviation:', stats.std(SidiData['BUI']))
print('FWI standard deviation:', stats.std(SidiData['FWI']))  
```

##### Question Five

Using the "Bejaia Region Dataset", calculate and show the “correlation coefficient” between “RH” and each of the following attributes ("Temperature", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI" and "FWI"), respectively

```python3
print("Correlation between RH and : ")
print("Temperature:", stats.correlation(BejaiaData[" RH"], BejaiaData["Temperature"]))
print("Windspeed:", stats.correlation(BejaiaData[" RH"], BejaiaData[" Ws"]))
print("Rain:", stats.correlation(BejaiaData[" RH"], BejaiaData["Rain "]))
print("FFMC:", stats.correlation(BejaiaData[" RH"], BejaiaData["FFMC"]))
print("DMC:", stats.correlation(BejaiaData[" RH"], BejaiaData["DMC"]))
print("DC:", stats.correlation(BejaiaData[" RH"], BejaiaData["DC"]))
print("ISI:", stats.correlation(BejaiaData[" RH"], BejaiaData["ISI"]))
print("BUI:", stats.correlation(BejaiaData[" RH"], BejaiaData["BUI"]))
print("FWI:", stats.correlation(BejaiaData[" RH"], BejaiaData["FWI"]))
```