library(h2o)
library(data.table)
library(h2oEnsemble)

h2o.removeAll()
h2o.init(nthreads=-1, max_mem_size="16g")

df_train <- read.csv("undersample/train_undersample.csv", header = T, sep=',', stringsAsFactors = FALSE)

df_test <- read.csv("test_preprocess_v1.csv", header = T, sep=',', stringsAsFactors = FALSE)

df_train$cible <- as.factor(df_train$cible)

train.h2o <- as.h2o(df_train)

test.h2o <- as.h2o(df_test)

colnames(train.h2o)

## Remove nb_recla_12, nb_recla_reco_12, type_d_offre


y <- 25
x <- c(1:11,14:20,22:24)


search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 120)
nfolds <- 5


# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.03) 
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt, 
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)


gbm_grid <- h2o.grid("gbm", x = x, y = y,
                     training_frame = train.h2o,
                     ntrees = 100,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

gbm_models <- lapply(gbm_grid@model_ids, function(model_id) h2o.getModel(model_id))


# RF Hyperparamters
mtries_opt <- 8:20 
max_depth_opt <- c(5, 10, 15, 20, 25)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_per_tree_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(mtries = mtries_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opt)

rf_grid <- h2o.grid("randomForest", x = x, y = y,
                    training_frame = train.h2o,
                    ntrees = 200,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,                    
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

rf_models <- lapply(rf_grid@model_ids, function(model_id) h2o.getModel(model_id))


# Deeplearning Hyperparamters
activation_opt <- c("Rectifier", "RectifierWithDropout", 
                    "Maxout", "MaxoutWithDropout") 
hidden_opt <- list(c(10,10), c(20,15), c(50,50,50))
l1_opt <- c(0, 1e-3, 1e-5)
l2_opt <- c(0, 1e-3, 1e-5)
hyper_params <- list(activation = activation_opt,
                     hidden = hidden_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    training_frame = train.h2o,
                    epochs = 15,
                    seed = 1,
                    nfolds = nfolds,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,                    
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_models <- lapply(dl_grid@model_ids, function(model_id) h2o.getModel(model_id))



# Create a list of all the base models
models <- c(gbm_models, rf_models, dl_models)


# Specify a defalt GBM as the metalearner
metalearner <- "h2o.gbm.wrapper"

# Stacking
stack <- h2o.stack(models = models, 
                   response_frame = train.h2o[,y],
                   metalearner = metalearner)


pred <- predict.h2o.ensemble(stack, test.h2o)

sub_ensemble <- as.data.frame(pred$pred)


write.csv(sub_ensemble,file="submission/submission_h2o_ensemble_stack_49.csv")
