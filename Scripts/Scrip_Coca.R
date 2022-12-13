##---------------------------Proyecto Final --------------------------------

install.packages("rgdal")
install.packages("sp")
install.packages("sf")
install.packages("terra")
install.packages("spdplyr")
install.packages("osmdata")
install.packages("caret")
library(caret)
library(spdplyr)
library(sf)
library(sp)
library(rgdal)
library(terra)
library(leaflet)
require(sf)
library("dplyr") #for data wrangling
library("gamlr") #ML
install.packages("gamlr") #ML
##-------------------------
setwd("/Users/df.mendivelso10/Desktop/Bases_Final_Coca/Base_2020")
list.files("/Users/df.mendivelso10/Downloads/Base_2020")
shape_2020 <- read_sf(dsn = ".", layer="geo_export_790bc368-34f8-4472-86cd-1e7fd5791105")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2020$geometry, color = "red") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2013")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2013")
shape_2013 <- read_sf(dsn = ".", layer="geo_export_f2b10651-56a4-4d7d-91fa-58e5b3833581")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2013$geometry, color = "yellow") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2014")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2014")
shape_2014 <- read_sf(dsn = ".", layer="geo_export_6b789664-800f-4a06-be2f-b8a36f1a8297")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2014$geometry, color = "blue") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2015")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2015")
shape_2015 <- read_sf(dsn = ".", layer="geo_export_05120a1c-ccc7-42bf-b408-53617a19ce43")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2015$geometry, color = "red") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2016")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2016")
shape_2016 <- read_sf(dsn = ".", layer="geo_export_96bc209a-2392-489b-ad58-8ce9a130b77e")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2016$geometry, color = "yellow") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2017")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2017")
shape_2017 <- read_sf(dsn = ".", layer="geo_export_2239c3d4-d199-4a3d-949d-d991e95e9b16")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2017$geometry, color = "blue") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2018")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2018")
shape_2018 <- read_sf(dsn = ".", layer="geo_export_de40177b-be8d-4e0f-ae57-d8c2883e2da5")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2018$geometry, color = "red") 

setwd("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2019")
list.files("/Users/df.mendivelso10/Documents/GitHub/Trabajo_final/Datos/Base_2019")
shape_2019 <- read_sf(dsn = ".", layer="geo_export_d23f54c0-941a-41cc-ae97-6305684f25c5")
leaflet() %>% addTiles() %>% addPolygons(data=shape_2019$geometry, color = "yellow") 
#------------------------ Pegue de bases
shape_1_20 <-rbind(shape_2001, shape_2002, shape_2003, shape_2004, shape_2005, shape_2006,shape_2007,shape_2008,shape_2009,shape_2010,shape_2011,shape_2012,shape_2013,shape_2014,shape_2015,shape_2016,shape_2017,shape_2018,shape_2019,shape_2020)

shape_1_2 <-st_filter(shape_2001, shape_2002)
shape_1_3 <-st_filter(shape_1_2, shape_2003)
shape_1_4 <-st_filter(shape_1_3, shape_2004)
shape_1_5 <-st_filter(shape_1_4, shape_2005)
shape_1_6 <-st_filter(shape_1_5, shape_2006)
shape_1_7 <-st_filter(shape_1_6, shape_2007)
shape_1_8 <-st_filter(shape_1_7, shape_2008)
shape_1_9 <-st_filter(shape_1_8, shape_2009)
shape_1_10 <-st_filter(shape_1_9, shape_2010)
shape_1_11 <-st_filter(shape_1_10, shape_2011)
shape_1_12 <-st_filter(shape_1_11, shape_2012)
shape_1_13 <-st_filter(shape_1_12, shape_2013)
shape_1_14 <-st_filter(shape_1_13, shape_2014)
shape_1_15<-st_filter(shape_1_14, shape_2015)
shape_1_16<-st_filter(shape_1_15, shape_2016)
shape_1_17<-st_filter(shape_1_16, shape_2017)
shape_1_18<-st_filter(shape_1_17, shape_2018)
shape_1_19<-st_filter(shape_1_18, shape_2019)
shape_1_20_coca<-st_filter(shape_1_19, shape_2020)


    union2  <- shape_1_20 %>%
      distinct(grilla1, .keep_all = TRUE)
    final<- merge(union2, tratados_mas_1_ha_mas_10_anos, by="grilla1")
    write.csv(final, "union2.csv")

87099/113320 #0.768611
26221/113320 #0.231389

#-------Distancia vías principales
st_crs(final$centroide)
st_crs(final_municipios)
final$centroide <- st_transform(final$centroide, "EPSG:4326" )  ### Version Correcta
final$geometry <- st_transform(final$geometry, "EPSG:4326" )  ### Version Correcta

vias_col <- st_transform(vias_col, "EPSG:4326" )  ### Version Correcta
int 
#centroide
final$centroide<-st_centroid(final$geometry)
#unión características con zonas
final2<- st_join(final, final_municipios,join=st_nearest_feature)
table(final2$municipio)

write.csv(final2, "base_final")



#división muestra
set.seed(12345) 
final2 <- final2 %>%
  mutate(holdout= as.logical(1:nrow(final2) %in%
                               sample(nrow(final2), nrow(final2)*.3)))
test<-final2[final2$holdout==T,] #33996
train<-final2[final2$holdout==F,] #79324


#Modelos
x<- select(train,-holdout,-centroide,-geometry,-ano.y, -areacoca.x, -ano.x, -grilla1,-tratado, -codmpio, -depto, -municipio,-areacoca.y, -num_unique)
x=as.data.frame(x)
x=x[,-13]
y<-select(train, tratado)
y=as.data.frame(y)
y=y[,-2]
y=as.factor(y)


ols <- train(y~ .,# model to fit
             data = cbind(y,x),
             trControl = trainControl(method = "cv", number = 10),
             method = "lm")
ols$results
intercept      RMSE  Rsquared       MAE      RMSESD  RsquaredSD       MAESD
 TRUE      0.2198309 0.7266785 0.1738315 0.001613869 0.004137539 0.001529631

 intercept      RMSE   Rsquared       MAE      RMSESD  RsquaredSD       MAESD
 1      TRUE 0.4033107 0.08030272 0.3261649 0.003938673 0.009801354 0.002890016
 
 logit <- train(y~ .,# model to fit
              data = cbind(y,x),
              trControl = trainControl(method = "cv", number = 10),
              method = "glm",
              family= "binomial")
 
logit$results

 y_hat_insample <- predict(ols, train)
 y_hat_outsample <- predict(ols, test)
 
 y_hat_insample <- ifelse(y_hat_insample>=0.5,1,0)
 y_hat_outsample <- ifelse(y_hat_outsample>=0.5,1,0)
 y_hat_insample <- as.matrix(y_hat_insample)
 y<-as.matrix(y)
 y_hat_outsample <- as.matrix(y_hat_outsample)

  cm_insample<-confusionMatrix(data=factor(y_hat_insample) , 
                              reference=factor(y) , 
                              mode="sens_spec" , positive="1")$table
 
 
 cm_outsample<-confusionMatrix(data=factor(y_hat_outsample) , 
                               reference=factor(test$tratado) , 
                               mode="sens_spec" , positive="1")$table
 
 # Confusion Matrix insample
 cm_insample
 # Confusion Matrix outsample
 cm_outsample
 
 #metricas
 install.packages("pacman")
 library(pacman)
 p_load(tidyverse, ggplot2, doParallel, rattle, MLmetrics,
        janitor, fastDummies, tidymodels, caret)
 
acc_in <- Accuracy(y_true = y, y_pred = y_hat_insample)
 acc_in <- round(100*acc_in, 2)
 pre_in <- Precision(y_true = y, y_pred = y_hat_insample)
 pre_in <- round(100*pre_in, 2)
 recall_in <- Recall(y_true = y, y_pred = y_hat_insample)
 recall_in <- round(100*recall_in, 2)
 f1_in <- F1_Score(y_true = y, y_pred = y_hat_insample)
 f1_in <- round(100*f1_in, 2)
 
 acc_out <- Accuracy(y_true = test$tratado, y_pred = y_hat_outsample)
 acc_out <- round(100*acc_out, 2)
 pre_out <- Precision(y_true = test$tratado, y_pred = y_hat_outsample)
 pre_out <- round(100*pre_out, 2)
 recall_out <- Recall(y_true = test$tratado, y_pred = y_hat_outsample)
 recall_out <- round(100*recall_out, 2)
 f1_out <- F1_Score(y_true = test$tratado, y_pred = y_hat_outsample)
 f1_out <- round(100*f1_out, 2)
 
 resultados2 <- data.frame(Modelo = "Modelo 2: Grid search", Base = c("Train", "Test"), 
                           Accuracy = c(acc_in, acc_out), 
                           Precision = c(pre_in, pre_out),
                           Recall = c(recall_in, recall_out),
                           F1 = c(f1_in, f1_out))
 
 resultados2
 
 
 #xgBOOTS
 
 
 #----3.R F ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 install.packages("ranger")
 library(ranger)
 
 #------ Creamos una grilla para tunear el random forest
 set.seed(12345)
 cv3 <- trainControl(number = 3, method = "cv")
 tunegrid_rf <- expand.grid(mtry = c(3, 5, 10), 
                            min.node.size = c(10, 30, 50,
                                              70, 100),
                            splitrule="gini"
 )
 
 cv3 <- trainControl(number = 3, method = "cv")
 tunegrid_rf <- expand.grid(mtry = c(3), 
                            min.node.size = c(10),
                            splitrule="variance"
 )
 
 #------ Modelo Random forest
 
 modeloRF <- train( y~ .,
                    data = cbind(y,x), 
                    method = "ranger", 
                    trControl = cv3,
                    metric = 'Acurracy', 
                    verbose = TRUE,
                    preProcess= c("center", "scale"),
                    tuneGrid = tunegrid_rf)
 #-------Resultados
 
 RMSE       Rsquared   MAE      
 0.3617734  0.2599572  0.2610678
 
 RMSE was used to select the optimal model using the smallest value.
 The final values used for the model were mtry = 10, splitrule = variance and min.node.size = 30.
 
 #--------------## Visualize variable importance 
 
 plot(modeloRF)

 install.packages("ranger")
 library(ranger)
 
 predicciones <- predict(
   modeloRF,
   data = test
 )
 
 # Get variable importance from the model fi
 importancia_pred <- modelo$variable.importance %>%
   enframe(name = "predictor", value = "importancia")
 
 # Gráfico
 grafico1<- ggplot(
   data = importancia_pred,
   aes(x    = reorder(predictor, importancia),
       y    = importancia,
       fill = importancia)
 ) +
   labs(x = "predictor", title = "Importancia predictores (permutación)") +
   geom_col() +
   coord_flip() +
   theme_bw() +
   theme(legend.position = "none")
 
 grafico2<-ggplot(
   data = importancia_pred,
   aes(x    = reorder(predictor, importancia),
       y    = importancia,
       fill = importancia)
 ) +
   labs(x = "predictor", title = "Importancia predictores (pureza de nodos)") +
   geom_col() +
   coord_flip() +
   theme_bw() +
   theme(legend.position = "none")
 
 plot_grid(grafico1, grafico2, nrow  = 1, ncol=2, labels="AUTO")

 #--------3.3 Resultados en test 
 y_hat_insample_rf <- predict(modeloRF, train)
 y_hat_outsample_rf <- predict(modeloRF, test)
 y_hat_insample_rf <- ifelse(y_hat_insample_rf>=0.5,1,0)
 y_hat_outsample_rf <- ifelse(y_hat_outsample_rf>=0.5,1,0)
 y_hat_insample_rf <- as.matrix(y_hat_insample_rf)
 y<-as.matrix(y)
 y_hat_outsample_rf <- as.matrix(y_hat_outsample_rf)
 
 cm_insample<-confusionMatrix(data=factor(y_hat_insample_rf) , 
                              reference=factor(y) , 
                              mode="sens_spec" , positive="1")$table
 
 cm_outsample<-confusionMatrix(data=factor(y_hat_outsample) , 
                               reference=factor(test$tratado) , 
                               mode="sens_spec" , positive="1")$table
 
 # Confusion Matrix insample
 cm_insample
 # Confusion Matrix outsample
 cm_outsample
 
 #metricas

 acc_in <- Accuracy(y_true = y, y_pred = y_hat_insample_rf)
 acc_in <- round(100*acc_in, 2)
 pre_in <- Precision(y_true = y, y_pred = y_hat_insample_rf)
 pre_in <- round(100*pre_in, 2)
 recall_in <- Recall(y_true = y, y_pred = y_hat_insample_rf)
 recall_in <- round(100*recall_in, 2)
 f1_in <- F1_Score(y_true = y, y_pred = y_hat_insample_rf)
 f1_in <- round(100*f1_in, 2)
 
 acc_out <- Accuracy(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 acc_out <- round(100*acc_out, 2)
 pre_out <- Precision(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 pre_out <- round(100*pre_out, 2)
 recall_out <- Recall(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 recall_out <- round(100*recall_out, 2)
 f1_out <- F1_Score(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 f1_out <- round(100*f1_out, 2)
 
 resultados2 <- data.frame(Modelo = "Modelo 2: Grid search", Base = c("Train", "Test"), 
                           Accuracy = c(acc_in, acc_out), 
                           Precision = c(pre_in, pre_out),
                           Recall = c(recall_in, recall_out),
                           F1 = c(f1_in, f1_out))
 
 resultados2
 
 #------ 4. XGBOOTS ---------------------------------------------------------------------------------------------------------------
 
 install.packages("xgboost")
 library(xgboost)
 grid_default <- expand.grid(nrounds = c(250,500),
                             max_depth = c(4,6,8),
                             eta = c(0.01,0.3,0.5),
                             gamma = c(0,1),
                             min_child_weight = c(10, 25,50),
                             colsample_bytree = c(0.7),
                             subsample = c(0.6))
 
 ctrl<- trainControl(method = "cv",
                     number = 5,
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     verbose=FALSE,
                     savePredictions = T)
 set.seed(1410)
 xgboost <- train(
   y ~.,
   data=cbind(x,y),
   method = "xgbTree",
   trControl = ctrl,
   metric = "RMSE",
   tuneGrid = grid_default,
   preProcess = c("center", "scale")
 )
 
 Tuning parameter 'colsample_bytree' was held constant at a value of 0.7
 Tuning parameter 'subsample' was held constant at a value of 0.6
 RMSE was used to select the optimal model using the smallest value.
 The final values used for the model were nrounds = 500, max_depth = 8, eta = 0.01, gamma = 0, colsample_bytree =
   0.7, min_child_weight = 10 and subsample = 0.6.
 xgboost$results
   
   
 min(xgboost$results$RMSE) #0.3617508
 min(xgboost$results$MAE) #0.2612553
 max(xgboost$results$Rsquared) #0.2600363
 
 #--------3.3 Resultados en test 
 y_xg_test<-predict(xgboost,newdata=test)
 RMSE1<-rmse(test$tratado, y_xg_test)#0.3398501
 MAE1<-mae(test$tratado, y_xg_test)#0.2392797
 RSQUARE(test$tratado, y_xg_test) #0.7647703

 
 #--------3.3 Resultados en test 
 y_hat_insample_xg <- predict(xgboost, train)
 y_hat_outsample_xg <- predict(xgboost, test)
 y_hat_insample_xg <- ifelse(y_hat_insample_xg>=0.5,1,0)
 y_hat_outsample_xg <- ifelse(y_hat_outsample_xg>=0.5,1,0)
 y_hat_insample_xg <- as.matrix(y_hat_insample_xg)
 y<-as.matrix(y)
 y_hat_outsample_xg<- as.matrix(y_hat_outsample_xg)
 
 cm_insample<-confusionMatrix(data=factor(y_hat_insample_xg) , 
                              reference=factor(y) , 
                              mode="sens_spec" , positive="1")$table
 
 cm_outsample<-confusionMatrix(data=factor(y_hat_outsample_xg) , 
                               reference=factor(test$tratado) , 
                               mode="sens_spec" , positive="1")$table
 
 # Confusion Matrix insample
 cm_insample
 # Confusion Matrix outsample
 cm_outsample
 
 #metricas
 
 acc_in <- Accuracy(y_true = y, y_pred = y_hat_insample_xg)
 acc_in <- round(100*acc_in, 2)
 pre_in <- Precision(y_true = y, y_pred = y_hat_insample_xg)
 pre_in <- round(100*pre_in, 2)
 recall_in <- Recall(y_true = y, y_pred = y_hat_insample_xg)
 recall_in <- round(100*recall_in, 2)
 f1_in <- F1_Score(y_true = y, y_pred = y_hat_insample_xg)
 f1_in <- round(100*f1_in, 2)
 
 acc_out <- Accuracy(y_true = test$tratado, y_pred = y_hat_outsample_xg)
 acc_out <- round(100*acc_out, 2)
 pre_out <- Precision(y_true = test$tratado, y_pred = y_hat_outsample_xg)
 pre_out <- round(100*pre_out, 2)
 recall_out <- Recall(y_true = test$tratado, y_pred = y_hat_outsample_xg)
 recall_out <- round(100*recall_out, 2)
 f1_out <- F1_Score(y_true = test$tratado, y_pred = y_hat_outsample_xg)
 f1_out <- round(100*f1_out, 2)
 
 resultados2 <- data.frame(Modelo = "Modelo 2: Grid search", Base = c("Train", "Test"), 
                           Accuracy = c(acc_in, acc_out), 
                           Precision = c(pre_in, pre_out),
                           Recall = c(recall_in, recall_out),
                           F1 = c(f1_in, f1_out))
 
 resultados2
 
 
 #Balancear 
 #---------------------- Oversamplig 
x_train<-cbind(x,y)
 x_train<- cbind(x,y)
 y_train<-y 
 x_train$y<-as.factor(x_train$y)
 train$tratado<-as.factor(train$tratado)
 y_train<-as.data.frame(y_train)

 #---------------------- Oversamplig
 table(y)
 set.seed(1103)
 upSampledTrain <- upSample(x = train,
                            y = train$tratado,
                            ## keep the class variable name the same:
                            yname = "Default") 
 
 #Modelos
 x2<- select(upSampledTrain,-holdout,-Default,-centroide,-geometry,-ano.y, -areacoca.x, -ano.x, -grilla1,-tratado, -codmpio, -depto, -municipio,-areacoca.y, -num_unique)
 y2<-select(upSampledTrain, tratado)
 y2=as.data.frame(y2)
 y2=as.factor(y2)
 
 table(train$tratado)
 #----3.R F ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 install.packages("ranger")
 library(ranger)
 train2<-upSampledTrain
 table(y)
 #------ Creamos una grilla para tunear el random forest
 set.seed(12345)
 cv3 <- trainControl(number = 3, method = "cv")
 tunegrid_rf <- expand.grid(mtry = c(3, 5, 10), 
                            min.node.size = c(10, 30, 50,
                                              70, 100),
                            splitrule="gini"
 )
 
 cv3 <- trainControl(number = 3, method = "cv")
 tunegrid_rf <- expand.grid(mtry = c(3), 
                            min.node.size = c(10),
                            splitrule="variance"
 )
 
 #------ Modelo Random forest

 modeloRF <- train( tratado~.,
                    data = cbind(x2,y2), 
                    method = "ranger", 
                    trControl = cv3,
                    metric = 'Acuracy', 
                    verbose = TRUE,
                    preProcess= c("center", "scale"),
                    tuneGrid = tunegrid_rf)
 
 
 #-------Resultados
 modeloRF$results$min.node.size
 
 RMSE       Rsquared   MAE      
 0.3617734  0.2599572  0.2610678
 
 RMSE was used to select the optimal model using the smallest value.
 The final values used for the model were mtry = 10, splitrule = variance and min.node.size = 30.
 
 #--------------## Visualize variable importance 
 
 plot(modeloRF)
 
 install.packages("ranger")
 library(ranger)
 x_test<-select(test,-holdout,-centroide,-geometry,-ano.y, -areacoca.x, -ano.x, -grilla1, -codmpio, -depto, -municipio,-areacoca.y, -num_unique)
 predicciones <- predict(
   modeloRF,
   data = x_test
 )
 
 tratado<-as.data.frame(tratado)
 modelo  <- ranger(
   formula   = tratado~ .,
   data      = cbind(y2,x2),
   num.trees = 10,
   seed      = 12345,
   importance= "impurity"
 )
 
 # Get variable importance from the model fi
 importancia_pred <- modelo$variable.importance %>%
   enframe(name = "predictor", value = "importancia")
 
 # Gráfico
 grafico1<- ggplot(
   data = importancia_pred,
   aes(x    = reorder(predictor, importancia),
       y    = importancia,
       fill = importancia)
 ) +
   labs(x = "predictor", title = "Importancia predictores (permutación)") +
   geom_col() +
   coord_flip() +
   theme_bw() +
   theme(legend.position = "none")
 
 grafico2<-ggplot(
   data = importancia_pred,
   aes(x    = reorder(predictor, importancia),
       y    = importancia,
       fill = importancia)
 ) +
   labs(x = "predictor", title = "Importancia predictores (pureza de nodos)") +
   geom_col() +
   coord_flip() +
   theme_bw() +
   theme(legend.position = "none")
 
 plot_grid(grafico1, grafico2, nrow  = 1, ncol=2, labels="AUTO")
 
 #--------3.3 Resultados en test 
 y_hat_insample_rf <- predict(modeloRF, upSampledTrain)
 y_hat_outsample_rf <- predict(modeloRF, test)
 y_hat_insample_rf <- as.matrix(y_hat_insample_rf)
tratado<-as.matrix(upSampledTrain$tratado)
 y_hat_outsample_rf <- as.matrix(y_hat_outsample_rf)
 
 cm_insample<-confusionMatrix(data=factor(y_hat_insample_rf) , 
                              reference=factor(tratado) , 
                              mode="sens_spec" , positive="1")$table
 
 cm_outsample<-confusionMatrix(data=factor(y_hat_outsample) , 
                               reference=factor(test$tratado) , 
                               mode="sens_spec" , positive="1")$table
 
 # Confusion Matrix insample
 cm_insample=as.data.frame(cm_insample)
 cm_outsample=as.data.frame(cm_outsample)
 
 # Confusion Matrix outsample
 cm_outsample
 
 #metricas
 
 acc_in <- Accuracy(y_true = tratado, y_pred = y_hat_insample_rf)
 acc_in <- round(100*acc_in, 2)
 pre_in <- Precision(y_true = tratado, y_pred = y_hat_insample_rf)
 pre_in <- round(100*pre_in, 2)
 recall_in <- Recall(y_true = tratado, y_pred = y_hat_insample_rf)
 recall_in <- round(100*recall_in, 2)
 f1_in <- F1_Score(y_true = tratado, y_pred = y_hat_insample_rf)
 f1_in <- round(100*f1_in, 2)
 
 acc_out <- Accuracy(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 acc_out <- round(100*acc_out, 2)
 pre_out <- Precision(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 pre_out <- round(100*pre_out, 2)
 recall_out <- Recall(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 recall_out <- round(100*recall_out, 2)
 f1_out <- F1_Score(y_true = test$tratado, y_pred = y_hat_outsample_rf)
 f1_out <- round(100*f1_out, 2)
 
 resultados2 <- data.frame(Modelo = "Modelo 2: Grid search", Base = c("Train", "Test"), 
                           Accuracy = c(acc_in, acc_out), 
                           Precision = c(pre_in, pre_out),
                           Recall = c(recall_in, recall_out),
                           F1 = c(f1_in, f1_out))
 
 resultados2
 
 #Resultados 
 y_hat_outsample_xg<-as.data.frame(y_hat_outsample_xg)
 test<-cbind(test,y_hat_outsample_xg)
 'predicho'->names(y_hat_outsample_xg)[names(y_hat_outsample_xg)=='V1']
 
test$predicho.1<- ifelse(test$predicho.1 == 0, NA, test$predicho.1)
test_final = subset(x = test, subset = is.na(predicho.1)==FALSE) 

pl <- ggplot(test_final, aes(x=areacoca.y))
p3<-pl + geom_histogram( aes(fill=..count..), col='black')+ labs(title = 'Promedio hectareas de coca en zonas cocaleras',
                                                                 
                                                                 x = 'Hectáreas',
                                                                 y = 'conteos',
                                                                 subtitle = 'Distrinución',
                                                                 caption = 'Resultados Modelo Random Forest')

hist(test$areacoca.y)

Colombia <- getbb(place_name = "Colombia", 
                  featuretype = "boundary:administrative", 
                  format_out = "sf_polygon") %>% .$multipolygon
Colombia<-Colombia[1,]
leaflet() %>% addTiles() %>% addPolygons(data=Colombia)

table(test$municipio) 
cero1<- ggplot()+geom_sf(data=Colombia) +
  geom_sf(data=test_final$geometry, color="cadetblue")+
  theme_minimal()+
labs(x = "Latitud", y = "Longitud")+labs(title = 'Zonas cocaleras predichas',
                                        x = 'Longitud',
                                        y = 'Latitud',
                                        subtitle = 'Solo para test',
                                        caption = 'Resultados Modelo Random Forest')


cero2<- ggplot()+
  geom_bar(aes(x=test_final$municipio),data=test_final, color="cadetblue")+
  theme_minimal()+theme(axis.text = element_text(angle = 90))+
  labs(x = "Latitud", y = "Longitud")+labs(title = 'Principales Municipios Zonas cocaleras predichas',
                                           x = 'conteo',
                                           y = 'Municipio',
                                           las=2,
                                           subtitle = 'Solo para test',
                                           caption = 'Resultados Modelo Random Forest')

cero3<- ggplot()+
  geom_bar(aes(x=test_final$depto),data=test_final, color="cadetblue")+
  theme_minimal()+
  labs(x = "Latitud", y = "Longitud")+labs(title = 'Principales Departamentos Zonas cocaleras predichas',
                                           x = 'conteo',
                                           y = 'Municipio',
                                           subtitle = 'Solo para test',
                                           caption = 'Resultados Modelo Random Forest')

sumary<-summary(final2)
sumary<-as.data.frame(sumary)
table(final2$tratado)
ingreso <- (as.data.frame(summary(base2))) ; ingreso
output <- capture.output(sumary, file=NULL, append =FALSE)
output_ad <-as.data.frame(output) #convertir summary en tabla
write.table(x = output_ad, file = "summary.xlsx", sep = " ", 
            row.names = FALSE, col.names = TRUE)

