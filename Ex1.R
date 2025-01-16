library(mlr3)
library(ggplot2)
library(mlbench)
library(data.table)
library(mlr3measures)
library(mlr3viz)

set.seed(1111)

# Load the Pima dataset
data(PimaIndiansDiabetes2, package = "mlbench")

# Convert it to a data table and remove rows with missing values
pima_data = as.data.table(na.omit(PimaIndiansDiabetes2))

# Create a classification task
pima_task = as_task_classif(pima_data, target = "diabetes", positive = "pos")

# Split the data into 80% training and 20% testing
split = partition(pima_task, ratio = 0.8)

# Set up the measure (classification error)
measure_ce = msr("classif.ce")

# Set up a baseline featureless learner
featureless_model = lrn("classif.featureless", predict_type = "prob")
# Train the featureless learner
featureless_model$train(pima_task, row_ids = split$train)
# Test the featureless learner
featureless_preds = featureless_model$predict(pima_task, row_ids = split$test)
# Set the threshold to 0.65
featureless_preds$set_threshold(0.5)
# Score it with the classification error measure
classification_error_featureless = featureless_preds$score(measure_ce)
print(paste("Classification Error (featureless):", classification_error_featureless))

# Set up the learner
model = lrn("classif.rpart", predict_type = "prob")
# Train the model
model$train(pima_task, row_ids = split$train)

# Test the model
predictions = model$predict(pima_task, row_ids = split$test)

#selected_threshold
predictions$set_threshold(0.5)
#predictions$set_threshold(0.3)

#check the fpr and fnr
predictions$score(msr("classif.fpr"))
predictions$score(msr("classif.fnr"))

#viusalize fpr with prob threshold
autoplot(predictions,  type = "threshold", measure = msr("classif.fnr"))
autoplot(predictions,  type = "threshold", measure = msr("classif.acc"))
