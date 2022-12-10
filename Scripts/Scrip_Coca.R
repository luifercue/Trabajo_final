##---------------------------Proyecto Final --------------------------------

install.packages("rgdal")
install.packages("sp")
install.packages("sf")
install.packages("terra")
library(sf)
library(sp)
library(rgdal)
library(terra)
library(leaflet)
require(sf)
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


