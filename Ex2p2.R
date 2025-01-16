library(mlr3)
library(mlr3viz)
library(mlr3learners)
library(ranger)
library(patchwork)
library(precrec)

set.seed(11111)

#Design the benchmark
spam_task = tsk("spam")
learners = lrns(c("classif.log_reg", "classif.ranger",
                  "classif.xgboost"), predict_type = "prob")

# Configure the xgboost learner with nrounds = 100
learners$classif.xgboost$param_set$values <- list(nrounds = 100)

rsmp_cv5 = rsmp("cv", folds = 5)

#Design the benchmark experiment
design = benchmark_grid(spam_task, learners, rsmp_cv5)
head(design)

#Run the benchmark experiment
bmr = benchmark(design)
bmr

#Score with AUC
bm_table = bmr$score(msr("classif.auc"))

#Compare the learners
autoplot(bmr, type = "roc") + autoplot(bmr, type = "prc") +
  plot_layout(guides = "collect")


#Compare the learners
autoplot(bmr, type = "boxplot", measure = msr("classif.auc"))