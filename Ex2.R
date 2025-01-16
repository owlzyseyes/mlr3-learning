#Apply a repeated cross-validation resampling strategy on tsk("mtcars")
#and evaluate the performance of lrn("regr.rpart"). 
#Use five repeats of three folds each.
#Calculate the MSE for each iteration and visualize the result. 
#Finally, calculate the aggregated performance score.

library(mlr3)
library(mlr3viz)
library(data.table)
library(dplyr)

#Create mtcars task
mt_task = tsk("mtcars")

#create a learner
lrn_rpart = lrn("regr.rpart")
lrn_rpart

#Create a resampling strategy
rs = rsmp("repeated_cv", repeats = 5, folds = 3)

#modelling
rr = resample(task = mt_task, learner = lrn_rpart, resampling = rs)

#MSE for each iteration
mse = rr$score(msr("regr.mse"))
mse[, .(iteration, regr.mse)]

#Visualize the result
autoplot(rr, type = "boxplot", measure = msr("regr.mse"))
#aggregate performance score
rr$aggregate(msr("regr.mse"))
