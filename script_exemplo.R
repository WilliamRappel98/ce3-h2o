# Packages
library(h2o)
library(microbenchmark)
library(glmnet)
library(randomForest)

# Initiate cluster H2O
h2o.init(
  nthreads=4,
  max_mem_size='10G'
)
# http://localhost:54321
# Tutorials: https://github.com/h2oai/h2o-tutorials/tree/master/h2o-open-tour-2016/chicago

# Data
loan_csv <- 'data/loan.csv'
data <- h2o.importFile(loan_csv)
dim(data)
str(as.data.frame(data))

# Data Preparation
data$bad_loan <- as.factor(data$bad_loan)
h2o.levels(data$bad_loan)

# Data splitting
splits <- h2o.splitFrame(
  data=data, 
  ratios=c(0.7, 0.15),
  seed=1
)
train <- splits[[1]]
valid <- splits[[2]]
train_cv <- h2o.rbind(train, valid)
test <- splits[[3]]
# Strategy 1: 70% Train    | 15% Valid | 15% Test
# Strategy 2: 85% Train-CV | --------- | 15% Test

# Sizes
nrow(train)
nrow(valid)
nrow(train_cv)
nrow(test)

# Identify response and predictor variables
y <- 'bad_loan'
x <- setdiff(names(data), c(y, 'int_rate'))
print(x)

# Available models
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html

# Help
?stats::glm
?h2o.glm

# GLM
glm1 <- glm(bad_loan ~ . - int_rate, data=as.data.frame(train), family=binomial(link='logit'))
glm2 <- h2o.glm(x=x, y=y, training_frame=train, model_id='glm_fit1', family='binomial')
microbenchmark(
  glm = glm(bad_loan ~ . - int_rate, data=as.data.frame(train), family=binomial(link='logit')),
  h2o = h2o.glm(x=x, y=y, training_frame=train, model_id='glm_fit1', family='binomial'),
  times = 3
)

# Random Forest
set.seed(1)
rf1 <- randomForest(bad_loan ~ . - int_rate, data=as.data.frame(train), ntree=50, na.action=na.omit)
rf2 <- h2o.randomForest(x=x, y=y, training_frame=train, model_id='rf_fit1', seed=1)
microbenchmark(
  randomForest = randomForest(bad_loan ~ . - int_rate, data=as.data.frame(train), ntree=50, na.action=na.omit),
  h2o = h2o.randomForest(x=x, y=y, training_frame=train, model_id='rf_fit1', seed=1),
  times = 1
)

# Comparison of some models
nb_fit1 <- h2o.naiveBayes(x=x, y=y, training_frame=train, model_id="nb_fit1")
nb_fit2 <- h2o.naiveBayes(x=x, y=y, training_frame=train_cv, model_id="nb_fit2", seed=1, nfolds=3)
glm_fit1 <- h2o.glm(x=x, y=y, training_frame=train, model_id="glm_fit1", family="binomial")
glm_fit2 <- h2o.glm(x=x, y=y, training_frame=train, model_id="glm_fit2", family="binomial", validation_frame=valid, lambda_search=TRUE)
rf_fit1 <- h2o.randomForest(x=x, y=y, training_frame=train, model_id="rf_fit1", seed=1)
rf_fit2 <- h2o.randomForest(x=x, y=y, training_frame=train_cv, model_id="rf_fit2", seed=1, nfolds=3)
gbm_fit1 <- h2o.gbm(x=x, y=y, training_frame=train, model_id="gbm_fit1", validation_frame=valid,
                    ntrees=500,
                    score_tree_interval=5,
                    stopping_rounds=3,
                    stopping_metric="AUC",
                    stopping_tolerance=0.0005,
                    seed=1)

# Performance
nb_perf1 <- h2o.performance(model = nb_fit1, newdata = test)
nb_perf2 <- h2o.performance(model = nb_fit2, newdata = test)
glm_perf1 <- h2o.performance(model = glm_fit1, newdata = test)
glm_perf2 <- h2o.performance(model = glm_fit2, newdata = test)
rf_perf1 <- h2o.performance(model = rf_fit1, newdata = test)
rf_perf2 <- h2o.performance(model = rf_fit2, newdata = test)
gbm_perf1 <- h2o.performance(model = gbm_fit1, newdata = test)

# AUC test set
h2o.auc(nb_perf1)
h2o.auc(nb_perf2)
h2o.auc(glm_perf1)
h2o.auc(glm_perf2)
h2o.auc(rf_perf1)
h2o.auc(rf_perf2)
h2o.auc(gbm_perf1)

# History
h2o.scoreHistory(gbm_fit1)
plot(gbm_fit1, timestep = "number_of_trees", metric = "AUC")

# Grid Search
params <- list(learn_rate = c(0.01, 0.1), max_depth = c(3, 5))
gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid1",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = params)
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1", sort_by = "auc", decreasing = TRUE)
print(gbm_gridperf1)

# Random Grid Search
gbm_params2 <- list(learn_rate = seq(0.1, 0.3, 0.1),
                    max_depth = seq(2, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria2 <- list(strategy = "RandomDiscrete", max_models = 5)
gbm_grid2 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params2,
                      search_criteria = search_criteria2)
gbm_gridperf2 <- h2o.getGrid(grid_id = "gbm_grid2", sort_by = "auc", decreasing = TRUE)
print(gbm_gridperf2)

# Random Grid Search using max_runtime
gbm_params <- list(learn_rate = seq(0.1, 0.3, 0.1),
                   max_depth = seq(2, 6, 1),
                   sample_rate = seq(0.8, 1.0, 0.05),
                   col_sample_rate = seq(0.5, 0.9, 0.1))
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 10)
gbm_grid <- h2o.grid("gbm", x = x, y = y,
                     grid_id = "gbm_grid2",
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 100,
                     seed = 1,
                     hyper_params = gbm_params,
                     search_criteria = search_criteria2)
gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid2", sort_by = "auc", decreasing = TRUE)
print(gbm_gridperf)
# Best model AUC
best_gbm_model_id <- gbm_gridperf@model_ids[[1]]
best_gbm <- h2o.getModel(best_gbm_model_id)
best_gbm_perf <- h2o.performance(model = best_gbm, newdata = test)
h2o.auc(best_gbm_perf)

# AutoML
# Max models in 60s
aml <- h2o.automl(
  x=x,
  y=y,
  training_frame=train_cv,
  max_runtime_secs=15,
  seed=1
)
lb <- aml@leaderboard
print(lb, n=nrow(lb))
# Prediction
(pred <- h2o.predict(aml, test))
(pred <- h2o.predict(aml@leader, test))
# Get leaderboard with all possible columns
lb <- h2o.get_leaderboard(object = aml, extra_columns = "ALL")
lb
# Leaderboard with test data
lb_test <- h2o.make_leaderboard(aml, test)
lb_test
# Get the best model using the metric
h2o.get_best_model(aml)@model_id[1]
h2o.get_best_model(aml, criterion = "logloss")@model_id[1]
h2o.get_best_model(aml, algorithm = "GBM")@model_id[1]
h2o.get_best_model(aml, algorithm = "GBM", criterion = "logloss")@model_id[1]
best <- h2o.get_best_model(aml)
# best@