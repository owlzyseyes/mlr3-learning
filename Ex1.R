library(mlr3)
library(ggplot2)
library(mlbench)
library(data.table)

set.seed(1111)

# Load the Pima dataset
data(PimaIndiansDiabetes2, package = "mlbench")

# Convert it to a data table and remove rows with missing values
pima_data = as.data.table(na.omit(PimaIndiansDiabetes2))

# Create a classification task
pima_task = as_task_classif(pima_data, target = "diabetes", positive = "pos")

# Split the data into 80% training and 20% testing
split = partition(pima_task, ratio = 0.8)

# Set up the learner
model = lrn("classif.rpart", predict_type = "prob")

# Train the model
model = model$train(pima_task, row_ids = split$train)

# Test the model
predictions = model$predict(pima_task, row_ids = split$test)

# Function to calculate FPR and FNR
calculate_rates <- function(predictions) {
  conf_matrix <- predictions$confusion
  TN <- conf_matrix[1, 1]
  FP <- conf_matrix[1, 2]
  FN <- conf_matrix[2, 1]
  TP <- conf_matrix[2, 2]
  
  FPR <- FP / (FP + TN)
  FNR <- FN / (FN + TP)
  
  list(FPR = FPR, FNR = FNR)
}

# Initialize variables
thresholds <- seq(0.1, 0.9, by = 0.01)
selected_threshold <- NA

# Evaluate thresholds
for (threshold in thresholds) {
  predictions$set_threshold(threshold)
  rates <- calculate_rates(predictions)
  
  if (rates$FPR < rates$FNR) {
    selected_threshold <- threshold
    break
  }
}

# Set the selected threshold
predictions$set_threshold(selected_threshold)
rates <- calculate_rates(predictions)

# Print the final threshold and rates
print(paste("Selected threshold:", selected_threshold))
print(paste("False Positive Rate (FPR):", rates$FPR))
print(paste("False Negative Rate (FNR):", rates$FNR))

# Confusion matrix
print(predictions$confusion)

# Set up the measure (classification error)
measure1 = msr("classif.ce")

# Score the predictions
classification_error = predictions$score(measure1)
print(paste("Classification Error:", classification_error))