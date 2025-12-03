################################################################################
# RandomForest classification
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


########################################################################################################################
# --- Part II
# Till here we have extracted and prepared the data to tune future models
# Now we will train RF models

#clean memory
rm(list= ls())
gc()

#define wd
setwd("")

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
#library(ROCit)
library(openxlsx)
library(varSel)
library(exactextractr)
library(writexl)

#re-import the value extracted
df_all <- readRDS("Dataframe_ML.rds")

#prepare dataset, remove possible NA values and MNF bands with strips effects
df_all_ml <- df_all

names(df_all_ml)[1:7] <- c("MCARI", "PRI", "RENDVI", # Make sure these indices are available
                              "SR", "SAVI", "VARI", "WBI")
# df_all_ml <- subset(df_all_ml, select = -c(MNF_band1, MNF_band2, MNF_band5, MNF_band6, MNF_band7, MNF_band9, MNF_band16, MNF_band20))#from mosaic west seems better to drop MNF 1, 2, 5, 6, 7, 9, 16 and 20
df_all_ml <- na.omit(df_all_ml)

#then prepare different dataset combination to test
# df_mnf <- df_all_ml[, c(1:32, 63)]
# df_mnf_vi <- df_all_ml[, c(1:48, 63)]
# df_mnf_bands <- df_all_ml[, c(1:32, 49:63)]
# df_vi <- df_all_ml[, c(33:48, 63)]
# df_bands <- df_all_ml[, c(49:63)]

#correlation plot
# corrplot::corrplot(cor(df_mnf %>% select_if(is.numeric),method = "spearman"), main = "Correlation ~ MNF")
corrplot::corrplot(cor(df_all_ml %>% select_if(is.numeric),method = "spearman"), main = "Correlation ~  VI-bands")
# corrplot::corrplot(cor(df_bands %>% select_if(is.numeric),method = "spearman"), main = "Correlation ~  Bands SFFS")

# for reproduciblity
set.seed(123)

#hyperparameter grid search
hyper_grid <- expand.grid(
  mtry       = seq(10, 19, by = 2),
  node_size  = seq(1, 10, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

# total number of combinations
nrow(hyper_grid)

ncol(df_all_ml)

for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = classes ~ ., 
    data            = df_all_ml, 
    num.trees       = 200,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

#the best random forest model we have found retains columnar categorical variables and uses mtry = 10, 
#terminal node size of 1 observations, and a sample size of 80%
#lets repeat this model to get a better expectation of our OBB error rate
#the out-of-bag (OOB) error is the model error in predicting the data left out of the training set 
#for that tree (P. Bruce and Bruce 2017). OOB is a very straightforward way to estimate the test error 
#of a bagged model, without the need to perform cross-validation or the validation set approach.

OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = classes ~ ., 
    data            = df_all_ml, 
    num.trees       = 200,
    mtry            = 19,
    min.node.size   = 1,
    sample.fraction = .8,
    importance      = "permutation"
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

#plot histogram of error rate
hist(OOB_RMSE, breaks = 20, main = "OBB RMSE - model with all variables")

#check most important features
optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 important variables ~ all variables")

#train RF model with ranger using the best tuning that we have found till now 
RF_ranger <- ranger(x = df_all_ml[, 1:ncol(df_all_ml)-1], y = df_all_ml$classes,
                    importance = "permutation", 
                    seed = 0xfedbeef,
                    num.trees = 200,
                    mtry            = 19,
                    min.node.size   = 1,
                    sample.fraction = .8)

# Inspect the structure and element names of the resulting RF model
RF_ranger
class(RF_ranger)
#str(RF_ranger)
#names(RF_ranger)

# Inspect the confusion matrix of the OOB error assessment of the RF model
#the matrix come from the internal sample fraction 
RF_ranger$confusion.matrix
ranger::importance(RF_ranger)

#now we want to test the model by splitting training into validation 
#and the calculate the final confusion matrix
#split data into 80 train and 20 test
#save the train and test
set.seed(2024)
train <- df_all_ml %>% group_by(classes) %>% sample_frac(0.8, replace = FALSE)
test <- setdiff(df_all_ml, train)

dir.create("ML_Model", showWarnings = FALSE)
saveRDS(test, "ML_Model/ML_test.rds")
saveRDS(train, "ML_Model/ML_train.rds") 

#Train RF model only with train data set
RF_ranger_train <- ranger(x = train[, 1:ncol(train)-1], y = train$classes,
                    importance = "permutation", 
                    seed = 0xfedbeef,
                    num.trees = 200,
                    mtry            = 19,
                    min.node.size   = 1
                    #sample.fraction = .8
                    )
RF_ranger_train

#check Confusion matrix
cm_rf <- confusionMatrix(data = predict(RF_ranger_train, test)$predictions,
                         test$classes)
cm_rf
capture.output(cm_rf, file = "ML_Model/Confusion_matrix_rf_allfeatures.txt")




#at the moment the best model  MNF and VI 
#train final model
RF_ranger_final <- ranger(x = df_all_ml[, 1:ncol(df_all_ml)-1], y = df_all_ml$classes,
                           importance = "permutation", 
                           seed = 0xfedbeef,
                           num.trees = 200,
                           mtry            = 19,
                           min.node.size   = 1)

#train RF model with ranger using the best tuning that we have found 
#RF_ranger_prob_mnf_vi <- ranger(x = df_mnf_vi[, 1:ncol(df_mnf_vi)-1], y = df_mnf_vi$Class,
#                                importance = "permutation",
#                                probability = T,
#                                seed = 0xfedbeef,
#                                num.trees = 200,
#                                mtry            = 16,
#                                min.node.size   = 1)

#save model RF as rds object
saveRDS(RF_ranger_final, "ML_Model/RF_ranger_model_final.rds")
#saveRDS(RF_ranger_prob_mnf_vi, "Vegetation/03_Model/RF_probability_ranger_model_mnf_vi.rds")


###############################################
# --- Part III
#here we test the trained model with a small area
#clean memory
rm(list= ls())
gc()

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
#library(ROCit)
library(openxlsx)
library(varSel)
library(exactextractr)
library(writexl)

#define wd
setwd("")

#predict raster map with the model trained
#test on a small sample
small_aoi <- vect("") #.shp

#re-open the saved file of RandomForest models
RF_ranger_final <- readRDS("ML_Model/RF_ranger_model_final.rds")
#RF_ranger_prob_mnf_vi <- readRDS("Vegetation/03_Model/RF_probability_ranger_model_mnf_vi.rds")

#import features needed for classification 
#load feature for classification
VI <- rast("") #.dat
Mosaic <- rast("", lyrs = c(31, 45, 55, 65, 73, 90, 107, 119, # you may need to change bands numbers, check the xlsx
                                                                                                                                                      126, 136, 137, 138, 198))
names(VI)[1:7] <- c("MCARI", "PRI", "RENDVI","SR", "SAVI", "VARI", "WBI") # Make sure these indices are available

# MNF_east <- rast("Input/MNF_bands_mosaic_east.dat")
# MNF_west <- rast("Input/MNF_bands_mosaic_west.dat")
# VI_east <- rast("Input/Mosaic_east_VI.tif")
# VI_west <- rast("Input/Mosaic_west_VI.tif")
# Mosaic_west <- rast("F:/32070_BF_2023/23-074_Bruneck_HS/50_ATM_corr/20230926/Mosaic_west_brefcor.bsq", lyrs = c(105, 108, 111, 117,
#                                                                                                                122, 127, 140, 148,
#                                                                                                                166, 175, 180, 181,
#                                                                                                                199, 222))
#Mosaic_east <- rast("F:/32070_BF_2023/23-074_Bruneck_HS/50_ATM_corr/20230925/Mosaic_east_final_UTM32N.bsq", lyrs = c(105, 108, 111, 117,
#                                                                                                                     122, 127, 140, 148,
#                                                                                                                     166, 175, 180, 181,
#                                                                                                                     199, 222))

#take out CRI1, CRI2 and VREI2 from VI rasters because they have too many NA values...
# VI_east <- subset(VI, c(1:3, 6:18))
# VI_west <- subset(VI_west, c(1:3, 6:18))
# 
# #here change band names 
# VI_col_names <- c(1:7)
# VI_col_names <- paste0("Index", VI_col_names)
# names(VI) <- VI_col_names
# names(MNF_west) <- MNF_band_names
# 
# #drop MNF bands excluded from the training model
# MNF_east <- subset(MNF_east, -c(1, 2, 5, 6, 7, 9, 16, 20)) 
# MNF_west <- subset(MNF_west, -c(1, 2, 5, 6, 7, 9, 16, 20)) 
# 
# #here change Mosaic bands names
mosaic_bands_names <- c(31, 45, 55, 65, 73, 90, 107, 119,
                        126, 136, 137, 138, 198) # you may need to change bands numbers, check the xlsx
mosaic_bands_names <- paste0("Band", mosaic_bands_names)
names(Mosaic) <- mosaic_bands_names


#crop respect to AOI test
features <- c(VI, Mosaic)


# features_subset_east <- crop(features_east, small_aoi)
# feature_subset_west <- crop(features_west, small_aoi)

#apply tree mask 
#features_subset_mask_east <- mask(features_subset, veg_mask_subset, maskvalue=0)
#features_subset_mask_west <- mask(features_subset, veg_mask_subset, maskvalue=0)

#run prediction on sample area 

predTreeSpec <- predict(features, RF_ranger_final, fun = function(...) predict(...)$predictions, 
                        filename = "", #.tif
                        verbose = T, overwrite = T, na.rm = T)



################################
#Part IV
#run it over all the area:east
features_east
predTreeSpec <- predict(features_east, RF_ranger_final, fun = function(...) predict(...)$predictions, 
                        filename = "", #.tif
                        verbose = T, overwrite = T, na.rm = T)


#run it over all the area:west
features_west
predTreeSpec <- predict(features_west, RF_ranger_final, fun = function(...) predict(...)$predictions, 
                        filename = "", #.tif
                        verbose = T, overwrite = T, na.rm = T)


