
# This is a long code R studio code - it runs properly but takes some time to run fully - please bare with the time

##-----Part 1 Download the dataset and partition it randomly into train and test set using a 70/30 split.

# reading the dataset - 1.setting WD / 2.reading data  

#setwd("C:\\Mydata\\dellstudio\\US_Stuff\\wrk-study\\Study\\Sem 3\\Machine Learning\\projects")

setwd("/Users/asinghalibm.com/Downloads/Machine Learning/projects/1") 

Bkshardata <- read.csv("hour.csv")

# bifocating the data into training (70) and test (30)

ind <- sample(2, nrow(Bkshardata), replace = TRUE, prob = c(0.7, 0.3))

trdata <- Bkshardata[ind==1,]
tsdata <- Bkshardata[ind==2,]

rm(ind)


trdata$yr <- NULL # Removing the year coloum as the data is only for only 1 year.
tsdata$yr <- NULL # Removing the year coloum as the data is only for only 1 year.

## Part 2 Design a linear regression model to model the count of bike rentals. 
## Include your regression model equation in the report.

# ------------doing operations on the trdata to be able to apply multiple regression on it

library(dummies)

#-----trdata - making all the characterstic variables into dummy matrices with seperate coloums for each level  
seasondummy <- dummy(trdata$season, sep = "_")
mnthdummy <- dummy(trdata$mnth, sep = "_")
hrdummy <- dummy(trdata$hr, sep = "_")
weathersitdummy <- dummy(trdata$weathersit, sep = "_")
holidaydummy <- dummy(trdata$holiday, sep = "_")
weekdaydummy <- dummy(trdata$weekday, sep = "_")
workingdaydummy <- dummy(trdata$workingday, sep = "_")

#Making the dummy matrix
dummymatrix <- cbind(seasondummy, mnthdummy, hrdummy, weathersitdummy, holidaydummy, weekdaydummy, workingdaydummy)

# removing the individual dummy matrices 
rm(seasondummy, mnthdummy, hrdummy, weathersitdummy, holidaydummy, weekdaydummy, workingdaydummy)

# getting integer variables from the trdata to form a matrix
trdataint <- trdata[,c(10:13)]

trdataintmat <- as.matrix(trdataint)

# removing trdataint df for convineience 
rm(trdataint)

# Combinig trdataintmat with dummymatrix to form final matrix
Fdummymatrix_tr <- cbind(1, dummymatrix, trdataintmat)

# removing the dummy matrix and int mat for convenience
rm(dummymatrix, trdataintmat)

#making the matrix of X's for multiple regression and transposing it for  convenience
xmat_tr <- t(Fdummymatrix_tr[,c(1, 3:5, 7:17, 19:41, 43:45, 47, 49:54, 56, 57, 59:60)])

#removing Fdummy matrix for convenience
rm(Fdummymatrix_tr)

# making df and subsequent matrix of y coloum
trdata_Y <- trdata[,16]

ymat_tr<- t(as.matrix(trdata_Y))

# removing the trdata_y for convenience
rm(trdata_Y)

xmat_tr # matrix of all X's 
ymat_tr # matrix of all y's


# making a matrix of desired variables
xmat_tr_1 <- xmat_tr[c(1:15, 39:52), ]  

xmat_tr_1 # keeping the x's which have got best fit in lm model 


#----tsdata operations  <- making all the characterstic variables into dummy matrices with seperate coloums for each level  
seasondummy <- dummy(tsdata$season, sep = "_")
mnthdummy <- dummy(tsdata$mnth, sep = "_")
hrdummy <- dummy(tsdata$hr, sep = "_")
weathersitdummy <- dummy(tsdata$weathersit, sep = "_")
holidaydummy <- dummy(tsdata$holiday, sep = "_")
weekdaydummy <- dummy(tsdata$weekday, sep = "_")
workingdaydummy <- dummy(tsdata$workingday, sep = "_")

#Making the dummy matrix
dummymatrix_ts <- cbind(seasondummy, mnthdummy, hrdummy, weathersitdummy, holidaydummy, weekdaydummy, workingdaydummy)

# removing the individual dummy matrices 
rm(seasondummy, mnthdummy, hrdummy, weathersitdummy, holidaydummy, weekdaydummy, workingdaydummy)

# getting integer variables from the tsdata to form a matrix
tsdataint <- tsdata[,c(10:13)]

tsdataintmat <- as.matrix(tsdataint)

# removing tsdataint df for convineience 
rm(tsdataint)

# Combinig tsdataintmat with dummymatrix to form final matrix
Fdummymatrix_ts <- cbind(1, dummymatrix_ts, tsdataintmat)

# removing the dummy matrix and int mat for convenience
rm(dummymatrix_ts, tsdataintmat)

#making the matrix of X's for multiple regression and transposing it for  convenience
xmat_ts <- t(Fdummymatrix_ts[,(c(1, 3:5, 7:17, 19:41, 43:45, 47, 49:54, 56, 57, 59))])

#removing Fdummy matrix for convenience
rm(Fdummymatrix_ts)

# making df and subsequent matrix of y coloum

tsdata_Y <- tsdata[,16]

ymat_ts<- t(as.matrix(tsdata_Y))

# removing the tsdata_y for convenience
rm(tsdata_Y)

xmat_ts # matrix of all X's 
ymat_ts # matrix of all y's

# making a matrix of desired variables
xmat_ts_1 <- xmat_ts[c(1:15, 39:51), ]  

xmat_ts_1 # keeping the x's which have got best fit in lm model 


#------multiple linear regression using gradient descent with xmat_tr_1 (training data) and ymat_tr

gradientDesc <- function(x, y, learn_rate, conv_threshold, n, max_iter) {
  theta <- as.matrix(runif(29,0,1))
  theta<-t(theta) # converting theta to the ideal form for multiplication
  yhat <- theta %*%  x 
  # finding out the error
  MSE <- sum((yhat - y) ^ 2) / n
  converged = F
  iterations = 0
  print("here")
  while(converged == F) {
    ## Implement the gradient descent algorithm
    theta_new <- theta - learn_rate*(1/n)*((yhat - y)%*%t(x))
    theta <- theta_new
    yhat <- theta %*% x
    MSE_new <- sum((yhat - y) ^ 2) / n
    print(iterations)
    print(paste("MSE_new",MSE_new))
    if(abs(MSE - MSE_new) <= conv_threshold) {
      converged = T
      print(paste("iterations", iterations, "MSE:", MSE, "MSE_new:", MSE_new))
      return(theta)
    }
    iterations = iterations + 1
    if(iterations > max_iter) { 
      converged = T
      print(paste("iterations", iterations, "MSE:", MSE, "MSE_new:", MSE_new))
      return(theta)
    }
    MSE <- MSE_new
  }
}

Model_1 <- gradientDesc(xmat_tr_1, ymat_tr, 0.001, 0.01, 12163, 100000) # error: 23690.8994931682 / iterations 74136

Beta_tr_1 <- Model_1

#Making Test function

Testfunction <- function(x, y,n,theta) {
  yhat <- theta %*%  x 
  # finding out the error
  MSE <- sum((yhat - y) ^ 2) / n
  return(MSE)
}

# Experiment 1

# Varying learning rate on training data learning rate -> 0.1 & 0.05

model_2 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 0.01, 12163, 10000)
Beta_tr_2 <- model_2

model_3 <- gradientDesc(xmat_tr_1, ymat_tr, 0.05, 0.01, 12163, 15000)
Beta_tr_3 <- model_3

# plot 

library(ggplot2)

learningrate <- c(0.1, 0.05)
error_trdata <- c(23138.1587365457, 23163.1135189178)

plot( x=learningrate, y = error_trdata, type="l")

# Testing the test data using the new beta's from varying learning rate -> 0.1 & 0.05 

TestError_1 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_2)
TestError_1 # accuracy or error on the test data - 23244.98

TestError_2 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_3)
TestError_2 # accuracy or error on the test data - 24709.9

# plot 

error_tsdata <- c(23244.98, 24709.9)

plot( x=learningrate, y = error_tsdata, type="l")

# Experiment 2

# Varying threshhold limts on training data 

model_6 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 1, 12163, 1000) #iterations 742 / MSE_new: 23689.7604669981
Beta_tr_4 <- model_6

model_7 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 5, 12163, 1500) #iterations 348 / MSE_new: 24611.7314502619
Beta_tr_5 <- model_7

# plot 

thersholdlmt <- c(1,5)
Terror_trdata <- c(23689.7604669981, 24611.7314502619)

plot( x=thersholdlmt, y = Terror_trdata, type="l")

# Testing the test data using the new beta's from varying thershold limit 

TestError_3 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_4)
TestError_3 # accuracy or error on the test data - 23868.14

TestError_4 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_5)
TestError_4 # accuracy or error on the test data - 24707.74

# plot 
Terror_tsdata <- c(23868.14, 24707.74)
plot( x=thersholdlmt, y = Terror_tsdata, type="l")


# best thershold -> 1 

model_8 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 1, 12163, 500) #MSE_new: 24073.3338987333
Beta_tr_6 <- model_8

model_9 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 1, 12163, 600) #MSE_new: 23872.5286719485
Beta_tr_7 <- model_9

model_10 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 1, 12163, 700) #MSE_new: 23734.9751801838
Beta_tr_8 <- model_10

model_11 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 1, 12163, 725) #MSE_new: 23706.6632632971
Beta_tr_9 <- model_11

model_12 <- gradientDesc(xmat_tr_1, ymat_tr, 0.1, 1, 12163, 750) #MSE_new: 23690.8670314558
Beta_tr_10 <- model_12

# Finding error on test data using the above beta's 

TestError_5 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_6)
TestError_5 # accuracy or error on the test data - 24221.03

TestError_6 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_7)
TestError_6 # accuracy or error on the test data - 24037.24

TestError_7 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_8)
TestError_7 # accuracy or error on the test data - 23910.34

TestError_8 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_9)
TestError_8 # accuracy or error on the test data - 23883.26

TestError_9 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_10)
TestError_9 # accuracy or error on the test data - 23869.01

# Pick your best threshold and plot train and test error (in one figure) as a function of number of gradient descent iterations

exprmnt3_iterations <- c(500,600,700,725,750)

exprmnt3_tr_errors <- c(24073.3338987333, 23872.5286719485, 23734.9751801838, 23706.6632632971, 23690.8670314558)

exprmnt3_ts_errors <- c(24221.03, 24037.24, 23910.34, 23883.26, 23869.01)

# plot

plot(x = exprmnt3_iterations, y = exprmnt3_tr_errors ,type="o", col="red")
lines(x = exprmnt3_iterations,y = exprmnt3_ts_errors , type="o", col="green")

plot(x = exprmnt3_iterations, y = exprmnt3_tr_errors ,type="o", col="red")
par(new=TRUE)
plot( x = exprmnt3_iterations,y = exprmnt3_ts_errors , type="o", col="green" )

df <- data.frame(exprmnt3_iterations, exprmnt3_tr_errors, exprmnt3_ts_errors)

ggplot(df, aes(exprmnt3_iterations)) +                    
  geom_line(aes(y=exprmnt3_tr_errors), colour="red") +  
  geom_line(aes(y=exprmnt3_ts_errors), colour="green") 
ylab("Error") + xlab("Iterations") 

# Experiment 3

# making a matrix of 3 random variables
xmat_tr_2 <- xmat_tr[c(1:4, 16:38, 42), ] #28
xmat_ts_2 <- xmat_ts[c(1:4, 16:38, 42), ]

# remaking gradient function for lesser varaibels
gradientDesc_1 <- function(x, y, learn_rate, conv_threshold, n, max_iter) {
  theta <- as.matrix(runif(28,0,1))
  theta<-t(theta) # converting theta to the ideal form for multiplication
  yhat <- theta %*%  x 
  # finding out the error
  MSE <- sum((yhat - y) ^ 2) / n
  converged = F
  iterations = 0
  print("here")
  while(converged == F) {
    ## Implement the gradient descent algorithm
    theta_new <- theta - learn_rate*(1/n)*((yhat - y)%*%t(x))
    theta <- theta_new
    yhat <- theta %*% x
    MSE_new <- sum((yhat - y) ^ 2) / n
    print(iterations)
    print(paste("MSE_new",MSE_new))
    if(abs(MSE - MSE_new) <= conv_threshold) {
      converged = T
      print(paste("iterations", iterations, "MSE:", MSE, "MSE_new:", MSE_new))
      return(theta)
    }
    iterations = iterations + 1
    if(iterations > max_iter) { 
      converged = T
      print(paste("iterations", iterations, "MSE:", MSE, "MSE_new:", MSE_new))
      return(theta)
    }
    MSE <- MSE_new
  }
}



# retraining model with random 3 variables
model_13 <- gradientDesc_1(xmat_tr_2, ymat_tr, 0.01, 0.001, 12259, 100000) # Iterations =93839 / MSE_new:14192.672422653 
Beta_tr_11 <- model_13

# testing the data on the new beta's received from retraining  
TestError_10 <- Testfunction(xmat_ts_2, ymat_ts,5120, Beta_tr_11)
TestError_10 # accuracy or error on the test data - 14005.67

# testing the first model    
testerror_11 <- Testfunction(xmat_ts_1, ymat_ts,5120, Beta_tr_1)
testerror_11 # accuracy or error on the test data - 23868.6

# Experiment 4

# making a matrix of 3 selected variables - hum, temp, mnth
xmat_tr_3 <- xmat_tr[c(1, 5:15, 50:51), ]
xmat_ts_3 <- xmat_ts[c(1, 5:15, 50:51), ]

# remaking gradient function for different variables
gradientDesc_2 <- function(x, y, learn_rate, conv_threshold, n, max_iter) {
  theta <- as.matrix(runif(14,0,1))
  theta<-t(theta) # converting theta to the ideal form for multiplication
  yhat <- theta %*%  x 
  # finding out the error
  MSE <- sum((yhat - y) ^ 2) / n
  converged = F
  iterations = 0
  print("here")
  while(converged == F) {
    ## Implement the gradient descent algorithm
    theta_new <- theta - learn_rate*(1/n)*((yhat - y)%*%t(x))
    theta <- theta_new
    yhat <- theta %*% x
    MSE_new <- sum((yhat - y) ^ 2) / n
    print(iterations)
    print(paste("MSE_new",MSE_new))
    if(abs(MSE - MSE_new) <= conv_threshold) {
      converged = T
      print(paste("iterations", iterations, "MSE:", MSE, "MSE_new:", MSE_new))
      return(theta)
    }
    iterations = iterations + 1
    if(iterations > max_iter) { 
      converged = T
      print(paste("iterations", iterations, "MSE:", MSE, "MSE_new:", MSE_new))
      return(theta)
    }
    MSE <- MSE_new
  }
}

# retraining model with random 3 variables
model_14 <- gradientDesc_2(xmat_tr_3, ymat_tr, 0.01, 0.001, 12259, 100000) # Iterations = / MSE_new:23235.0163219682
Beta_tr_12 <- model_14


# testing the data on the new beta's received from retraining  
TestError_12 <- Testfunction(xmat_ts_3, ymat_ts,5120, Beta_tr_12)
TestError_12 # accuracy or error on the test data - 23446.58