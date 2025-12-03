################################################################################
# SVM classification
################################################################################

# --- Part I: Data preparation
# In this first part we extract raster values and prepare a dataframe that can be used in the future (to avoid to repeat this part)

#clean memory
rm(list= ls())
gc()

#check for packages and install if missing
if(!"terra" %in% installed.packages()){install.packages("terra")}
if(!"sf" %in% installed.packages()){install.packages("sf")}
if(!"ranger" %in% installed.packages()){install.packages("ranger")}
if(!"caret" %in% installed.packages()){install.packages("caret")}
if(!"Metrics" %in% installed.packages()){install.packages("Metrics")}
if(!"randomForest" %in% installed.packages()){install.packages("randomForest")}
if(!"tidyr" %in% installed.packages()){install.packages("tidyr")}
if(!"openxlsx" %in% installed.packages()){install.packages("openxlsx")}

#load library 
library(terra)
library(sf)
library(ranger)
library(caret)
library(dplyr)
library(Metrics)
library(randomForest)
library(tidyr)
library(broom)
library(openxlsx)
library(varSel)
library(exactextractr)
library(writexl)

#define wd
setwd("")

#load feature for classification
# MNF <- rast("Input/MNF_bands_mosaic_east.dat")
VI <- rast("") #.dat
Mosaic <- rast("") #.dat


ROI_poly <- vect("") #.shp

#print start time and go for the loop
old <- Sys.time() # get start time

#take out CRI1, CRI2 and VREI2 from VI rasters because they have too many NA values...
# VI <- subset(VI, c(2:7))


#load vegetation mask (if needed)
#veg_mask <- rast("Vegetation/0_Trees_mask/Ferrara_high_vegetation_mask.dat")

#apply mask to the different inputs (if needed)
#MNF <- mask(MNF, veg_mask, maskvalue=0)
#VI <- mask(VI, veg_mask, maskvalue=0)
#PCA <- mask(PCA, veg_mask, maskvalue=0)


# #here change band names 
# MNF_band_names <- c(1:40)
# MNF_band_names <- paste0("MNF_band", MNF_band_names)
# names(MNF_east) <- MNF_band_names
# names(MNF_west) <- MNF_band_names

#here change Mosaic bands names
mosaic_bands_names <- c(1:224)
mosaic_bands_names <- paste0("Band", mosaic_bands_names)
names(Mosaic) <- mosaic_bands_names


#check metadata
VI
Mosaic


#prepare a stack between MNF, VI and Mosaic : WEST area
features <- c(VI, Mosaic)


#check training 
ROI_poly
summary(ROI_poly$Class) #ROI_NAME represents the species to classify

#Extract pixel values below the points into a dataframe: both for west and east
trainingData <- terra::extract(features, ROI_poly) #maybe here is better to use exact_extract that is faster
df_all <- trainingData

df_all <- na.omit(df_all) #eliminate row with still NA value

#use SFFS to find out which band is more significant
number_bands_to_select <- 30
df_SFFS <- df_all[,c(1,8:231)]
df_SFFS <- df_SFFS[complete.cases(df_SFFS),]
se <- varSelSFFS(g = df_SFFS$ID, X = df_SFFS[,c(2:225)], strategy = "mean", n= number_bands_to_select)
#se$features[number_bands_to_select,]

#save the vector with the bands selected by JM, eliminate posssible NA values
bands_select <- se$features[13,] 
bands_select <- bands_select[complete.cases(bands_select)]
bands_select <- sort(bands_select)
bands_select <- subset(bands_select, bands_select < 232) #take off possible noisy bands that were selcted...

#subset the df all with the bands that have been selected
bands_selected_df_number <- bands_select + 7 
df_all <- df_all[, c(1:7, bands_selected_df_number)]

# add a column with class name and then take out ID column
df_all$classes <- as.factor(ROI_poly$Class[match(df_all$ID, seq(nrow(ROI_poly)))]) 
df_all <- df_all[-1]

#save table df_all for the future
write_xlsx(df_all,"Dataframe_ML.xlsx") #save as excell table
saveRDS(df_all, "Dataframe_ML.rds") #save as RDS object

#print final operation time
new <- Sys.time() - old # calculate difference
print(new) # print in nice format
#time for extract values and prepare data frame: 



# --- Part II: SVM training & tuning with caret

# clean memory and set wd
rm(list = ls()); gc()
setwd("")

# packages (installa se manca)
pkgs <- c("terra","sf","caret","dplyr","broom","openxlsx","writexl","kernlab")
for(p in pkgs) if(! p %in% installed.packages()) install.packages(p)

# load libraries
library(terra); library(sf); library(caret); library(dplyr)
library(broom); library(openxlsx); library(writexl); library(kernlab)

# re-import the value extracted (proveniente da Part I)
df_all <- readRDS("Dataframe_ML.rds")

# keep same variable naming as original
df_all_ml <- df_all
names(df_all_ml)[1:7] <- c("MCARI", "PRI", "RENDVI", "SR", "SAVI", "VARI", "WBI") # Make sure these indices are available
df_all_ml <- na.omit(df_all_ml)

# ensure class variable present and is factor (as in your original pipeline)
# if you already created 'classes' in Part I, keep it; otherwise create it here
if(!"classes" %in% names(df_all_ml)) {
  stop("Column 'classes' does not exist in df_all_ml. Make sure in Part I you created df_all$classes.")
}
df_all_ml$classes <- as.factor(df_all_ml$classes)

# make column names safe for formula / model usage
names(df_all_ml) <- make.names(names(df_all_ml))

# quick info
cat("Dataframe dimensions:", dim(df_all_ml), "\n")
cat("Class levels:", levels(df_all_ml$classes), "\n\n")

# ------------------------------
# Train/test split (80/20) stratified by class
# ------------------------------
set.seed(2024)
train_index <- createDataPartition(df_all_ml$classes, p = 0.8, list = FALSE)
train <- df_all_ml[train_index, ]
test  <- df_all_ml[-train_index, ]

# save train/test like in original
dir.create("ML_Model", showWarnings = FALSE)
saveRDS(test, "ML_Model/ML_test.rds")
saveRDS(train, "ML_Model/ML_train.rds")

# ------------------------------
# caret trainControl: cross-validation
# ------------------------------
ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 3,
                     summaryFunction = multiClassSummary, # for multiclass metrics
                     classProbs = TRUE,
                     savePredictions = "final",
                     allowParallel = TRUE,
                     verboseIter = TRUE)

# preprocess: SVM strongly benefits from centering + scaling
preprocess_steps <- c("center", "scale")

# ------------------------------
# Grid + training for SVM Linear
# ------------------------------
# Regularization parameter C: try a reasonable exponential grid
grid_linear <- expand.grid(C = 2^(-3:7))

set.seed(123)
svm_linear_fit <- train(classes ~ .,
                        data = train,
                        method = "svmLinear",
                        metric = "Accuracy",
                        preProcess = preprocess_steps,
                        trControl = ctrl,
                        tuneGrid = grid_linear)

# ------------------------------
# Grid + training for SVM Radial (RBF)
# We need sigma (kernel width) and C. Use kernlab::sigest to estimate sigma range.
# ------------------------------
# Estimate a reasonable sigma from a subset (sigest can be slow on huge datasets)
set.seed(123)
# use only numeric predictors (caret will handle predictions)
x_sample <- train %>% dplyr::select(-classes)
# if dataset is large, use a random subset to estimate sigma
if(nrow(x_sample) > 2000) x_sample_s <- x_sample[sample(nrow(x_sample), 2000), ] else x_sample_s <- x_sample

# compute suggested sigma; wrap in try in case of issues
sigma_est <- try(kernlab::sigest(as.matrix(x_sample_s), scaled = TRUE), silent = TRUE)
if(!inherits(sigma_est, "try-error")) {
  sigma_vals <- as.numeric(sigma_est)
  # choose a small, medium, large around the estimate
  sigma_grid <- c(sigma_vals[1], mean(sigma_vals), sigma_vals[3])
} else {
  # fallback defaults
  sigma_grid <- c(0.001, 0.01, 0.1)
}

# radial grid: combine sigma and C
grid_radial <- expand.grid(sigma = sigma_grid,
                           C = 2^(-3:7))

set.seed(123)
svm_radial_fit <- train(classes ~ .,
                        data = train,
                        method = "svmRadial",
                        metric = "Accuracy",
                        preProcess = preprocess_steps,
                        trControl = ctrl,
                        tuneGrid = grid_radial)

# ------------------------------
# Compare models and pick the best (by Accuracy)
# ------------------------------
resamps <- resamples(list(linear = svm_linear_fit, radial = svm_radial_fit))
summary_resamps <- summary(resamps)

# best models
best_linear_acc <- max(svm_linear_fit$results$Accuracy)
best_radial_acc <- max(svm_radial_fit$results$Accuracy)

cat("Best linear Accuracy:", best_linear_acc, "\n")
cat("Best radial Accuracy:", best_radial_acc, "\n")

if(best_radial_acc >= best_linear_acc) {
  svm_best <- svm_radial_fit
  selected_method <- "svmRadial"
} else {
  svm_best <- svm_linear_fit
  selected_method <- "svmLinear"
}

cat("Selected model:", selected_method, "\n")
print(svm_best$bestTune)

# save final caret model
saveRDS(svm_best, "ML_Model/SVM_model_final_caret.rds")

# ------------------------------
# Evaluate on held-out test set
# ------------------------------
pred_test <- predict(svm_best, newdata = test)
cm_svm <- confusionMatrix(pred_test, test$classes)
print(cm_svm)
capture.output(cm_svm, file = "ML_Model/Confusion_matrix_svm.txt")

# If you want class probabilities also:
# probs_test <- predict(svm_best, newdata = test, type = "prob")

# ------------------------------
# Optional: variable importance (for svmRadial caret uses varImp)
# ------------------------------
vi <- varImp(svm_best, scale = TRUE)
# top variables
print(head(vi$importance[order(-apply(vi$importance, 1, max)), , drop = FALSE], 25))
# save importance
write.xlsx(as.data.frame(vi$importance), "ML_Model/SVM_variable_importance.xlsx")

# ------------------------------
# Part III / IV: Predict raster and save as TIF
# ------------------------------

# load the model (just to be robust)
svm_final <- readRDS("ML_Model/SVM_model_final_caret.rds")

# load features used for mapping
VI <- rast("") #.dat
Mosaic <- rast("", #.dat
               lyrs = c(31, 45, 55, 65, 73, 90, 107, 119, 126, 136, 137, 138, 198))
names(VI)[1:7] <- c("MCARI", "PRI", "RENDVI","SR", "SAVI", "VARI", "WBI")

# name mosaic bands like your script
mosaic_bands_names <- c(31, 45, 55, 65, 73, 90, 107, 119, 126, 136, 137, 138, 198)
mosaic_bands_names <- paste0("Band", mosaic_bands_names)
names(Mosaic) <- mosaic_bands_names

# stack features
features <- c(VI, Mosaic)

# ensure column names of raster stack match the training dataframe predictors
# caret used make.names on df_all_ml; replicate that to be safe
names(features) <- make.names(names(features))

# If the feature stack contains more layers than the model expects, subset to the predictors
# caret model keeps the predictors names in svm_best$finalModel? Instead, use svm_best$trainingData column names
# caret stores trainingData with the outcome column named ".outcome" or the response name; we can extract predictors:
training_data_names <- colnames(svm_best$trainingData)
# caret::train by default includes the outcome column as ".outcome" in svm_best$trainingData (or the original name)
# Let's find predictor names by removing the outcome column:
outcome_name <- svm_best$finalModel@.Data # not reliable; we use the original data frame names
# The safer approach: use names from train: svm_best$finalModel doesn't contain original colnames reliably across methods.
# But caret stores predictors in svm_best$finalModel? To be robust, we'll extract predictors from the original 'train' data used earlier:
predictor_names <- setdiff(names(train), "classes")
# Ensure all predictor_names are present in features; if not, throw warning
missing_pred <- setdiff(predictor_names, names(features))
if(length(missing_pred) > 0) {
  warning("The following predictors weren't found: ", paste(missing_pred, collapse = ", "),
          "\nChek the bands/layer loading before prediction.")
}

# Subset features to the predictors present
common_preds <- intersect(predictor_names, names(features))
features_subset <- features[[common_preds]]

# Predict: terra::predict will call predict(svm_final, newdata) internally; caret::train objects support predict
out_tif_path <- "" #.tif
cat("Start raster prediction ->", out_tif_path, "\n")
# Important: terra::predict passes blocks of data as data.frame, caret will require columns in same order/names
predRaster <- terra::predict(features_subset, model = svm_final, filename = out_tif_path,
                             type = "raw", overwrite = TRUE, na.rm = TRUE, verbose = TRUE)

cat("Raster prediction finished. Output saved to:", out_tif_path, "\n")

# ------------------------------
# Optionally run across full east / west stacks (Part IV analog)
# ------------------------------
# If you have features_east and features_west raster stacks prepared like in original, use same approach:
# features_east <- rast("path_to_east_stack")
# names(features_east) <- make.names(names(features_east))
# features_east_subset <- features_east[[common_preds]]
# terra::predict(features_east_subset, model = svm_final, filename = "path_out_east.tif", overwrite = TRUE)

# features_west <- rast("path_to_west_stack")
# names(features_west) <- make.names(names(features_west))
# features_west_subset <- features_west[[common_preds]]
# terra::predict(features_west_subset, model = svm_final, filename = "path_out_west.tif", overwrite = TRUE)

# End of script
