# Bike Rental Prediction using Gradient Descent Algorithm

### Goal
Implementing a linear regression model on the dataset to predict the total number of bike rentals during a given hour.

### Scope of the Project / Analysis 
Based on the variables, there are several supervised and unsupervised techniques that can could be performed on the above dataset to throw several insights on customer preferences. However, I would limit the scope of this analysis to using regression extensively to analyze the bike rentals. Additionally to learn the interal working of a regression algorithm - I would not be using any available implementation of regression model, but will implement the gradient descent algorithm programatically and ascertain the sum of squares that yield the least value of the cost function. 

### About the Dataset 
Bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions, precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors. The core data set is related to   the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA which is publicly available in http://capitalbikeshare.com/system-data. We aggregated the data on two hourly and daily basis and then extracted and added the corresponding weather and seasonal information. Weather information are extracted from http://www.freemeteo.com. 

### Dataset Characteristics
Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
	
	- instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered

### High Level Approach 

- A. Download the dataset and partition it randomly into train and test set using a 70/30 split.
- B. Design a linear regression model to predict the count of bike rentals.
- C. Implement the gradient descent algorithm with batch update rule.Report initial parameter values.

### Experimentation:

1. I am going to experiment with various values of learning rate ∝ and report the findings as how the error varies for train and test sets with varying ∝.  lastly I will report my best ∝ and why it was picked.
2. I am going to experiment with various thresholds for convergence and Plot error results for train and test sets as a function of threshold and describe how varying the threshold affected the error. In the end, I will pick my best threshold and plot train and test error as a function of number of gradient descent iterations.
3. I am going to Pick three features randomly and retrain the model only on these 3 features. This is to compare train and test error results for the case of using all features to using three random features.
4. Lastly, I will pick three features that I think are best suited to predict the output, and retrain my model using these three features. Additionally I will compare to the case of using all features and to random.

## Solution Report

### Part A: Download the dataset and partition it randomly into train and test set using a 70/30 split.
Accordingly the data was downloaded using “setwd” command and then the data was split into train and tests using “sample” function. 
Train = 70% of the data 
Test = 30% of the data


### Part B: Design a linear regression model to predict the count of bike rentals.

#### Model 
```cnt ~ beta0X0 + season_2 + season_3 + season_4 + mnth_2 + mnth_3 + mnth_4 + mnth_5 + mnth_6 + mnth_7 + mnth_8 + mnth_9 + mnth_10 + mnth_11 + mnth_12 + weathersit_2 + weathersit_3 + weathersit_4 + holiday_1 + weekday_1 + weekday_2 + weekday_3 + weekday_4 + weekday_5 + weekday_6 + workingday_1 + temp + hum + windspeed
```

#### Model Rationale:
- Have omitted variable “atemp” as it would be a directly related to “temp” as both are variable of temperature but calculated differently.
- Have omitted variables “Casual” & “registered” as both of them add up to the Y variable and thus if we include them into our regression then the regression will have a problem of Exact Multicollinearity
- As some of the X’s were categorical, dummy variables for the same have been included.
- The factors “temp , hum, windspeed” were already normalized thus they have been used as is.

### Part C: Implement the gradient descent algorithm with batch update rule.Report initial parameter values.

#### Parameter Values of Final model:

```Alpha = 0.001 
Iterations = 100000
Convergence Threshold = 0.01
Sum of error at alpha = 0.01 (train) -> 23690.8994931682
Sum of error at alpha = 0.01 (test) -> 23868.6
```

#### Final Model & Equation (Based on train Data):

```CNT ~ 129.34 + 40.46(SEASON_2) + 12.64(SEASON_3) + 72.60(SEASON_4) + 10.03(MNTH_2) 
+ 12.79(MNTH_3) – 2.55(MNTH_4) + 27.89(MNTH_5) – 6.92(MNTH_6) – 15.10(MNTH_7) 
+ 16.27 (MNTH_8) + 53.99 (MNTH_9) + 21.15 (MNTH_10) – 1.07 (MNTH_11) + 13.20 (MNTH_12) 
+ 6.45(WEATHERSIT_2) – 26.47(WEATHERSIT_3) + 1.15(WEATHERSIT_4) – 3.13(HOLIDAY_1) 
– 4.26 (WEEKDAY_1) + 2.56(WEEKDAY_2) + 5.53(WEEKDAY_3) – 3.91(WEEKDAY_4) +8.48(WEEKDAY_5)
 + 17.42 (WEEKDAY_6) + 10.25(WORKINGDAY_1) + 321.85(TEMP) – 265.19(HUM) + 75.28(WINDSPEED)
```

### Experiment 1

 - Varying learning rate on training data learning rate -> 0.1, 0.05 
```Sum of error at alpha = 0.1 & iterations = 6252 -> 23138.1587365457
Sum of error at alpha = 0.05 & iterations = 9017 -> 23163.1135189178
``` 
- Testing the test data using the new beta's from varying learning rate -> 0.1 & 0.05
```Sum of error at alpha = 0.1 -> 23059.558883839
Sum of error at alpha = 0.05 -> 23086.8130155647
```
```
Learning Rate	Training Data	Test Data
0.1	23138.1587365457	23244.98
0.05	23163.1135189178	24709.9
```

#### Best learning rate = 0.1  -> This is so as it computes the MSE faster and has comparatively low Total error.

### Experiment 2
- Varying threshold limit on training data 
```
Sum of error at alpha = 0.1, iterations = 742 & threshold limit = 1 -> 23689.7604669981
Sum of error at alpha = 0.1, iterations = 348 & threshold limit = 5-> 24611.7314502619 
```

- Varying threshold limit on test data
```
Sum of error at alpha = 0.1, iterations = 760 & threshold limit = 1-> 23687.4832562364
Sum of error at alpha = 0.1, iterations = 429 & threshold limit = 5-> 25572.3354264115
```
```
Threshold Limit	Training Data(error)	    Test Data (error)
1	              23689.7604669981	        23687.4832562364
5	              24611.7314502619          25572.3354264115
```

#### In both train and test data sets – varying the threshold limit has inversely affected the total sum of error.Best threshold = 1

### Experiment 3
```
Data	Error (All Features)	Error(Random 3 Features)
Train	23690.8994931682	    14192.672422653
Test	23868.6	              14005.67
```
                                       			                                                                       
- The Three features randomly picked are: Season, Hour & Holiday

### Experiment 4 

```
Comparison
Data	Error (All features)	Error(Random 3 features) Season/hr/Holiday	Error (3 Selected features) mnth/temp/hum
Train 	23690.8994931682	  14192.672422653	                            23235.0163219682
Test	  23868.6	            14005.67	                                  23446.58
```

Q - Did your choice of features provide better results than picking random features? why?
A – No, my choice of features did not provide better results than picking random features. This could have happened due to presence of collinearity or due to omitted variable bias – as I could have omitted some very important variables thus increasing the error.
Q - Did your choice of features provide better results than using all features? Why?
A – The result of using all features vs selected 3 features doesn’t differ much from each other.

### Discussion:
````
Final Model & Equation (Based on train Data):
CNT ~ 129.34 + 40.46(SEASON_2) + 12.64(SEASON_3) + 72.60(SEASON_4) + 10.03(MNTH_2) 
+ 12.79(MNTH_3) – 2.55(MNTH_4) + 27.89(MNTH_5) – 6.92(MNTH_6) – 15.10(MNTH_7) 
+ 16.27 (MNTH_8) + 53.99 (MNTH_9) + 21.15 (MNTH_10) – 1.07 (MNTH_11) + 13.20 (MNTH_12) 
+ 6.45(WEATHERSIT_2) – 26.47(WEATHERSIT_3) + 1.15(WEATHERSIT_4) – 3.13(HOLIDAY_1) 
– 4.26 (WEEKDAY_1) + 2.56(WEEKDAY_2) + 5.53(WEEKDAY_3) – 3.91(WEEKDAY_4) +8.48(WEEKDAY_5)
 + 17.42 (WEEKDAY_6) + 10.25(WORKINGDAY_1) + 321.85(TEMP) – 265.19(HUM) + 75.28(WINDSPEED)
````

## Interpretation: 

- Overall the fit of the model seems good. The error, after performing gradient descent, reduced from around 68K to 23K – which is a good performance. 
- Considering the produced model – it looks like bike rentals is a seasonal business and thus “season” has a large effect of the no. of bikes being rented.
- Other steps that could have been taken with regards to modelling to get better results are:
1. Checking to see if there is any interaction effect prevalent in the independent variables
2. Checking to see if there is any synergies present in the independent variables 
3. Also we could have checked if there is any non linear effect 

