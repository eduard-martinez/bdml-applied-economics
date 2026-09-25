## Big Data y Machine Learning para Economia Aplicada
## week-05: arboles, bosques y boosting (partir, promediar y corregir)
## Last run: Sep 25, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , rpart , randomForest , xgboost , glmnet , doParallel)

## cluster: los bosques y el boosting se ajustan en paralelo, cada tarea con su semilla (mismo resultado, menos tiempo)
cores <- parallel::detectCores()
cl <- makeCluster(max(1 , cores - 2))
registerDoParallel(cl)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-05/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo <- import("input/censo_puestos_2022.rds" , trust=T)

## 1.2. la ficha del censo de la semana 3: de conteos a proporciones (una categoria por bloque queda por fuera)
ficha <- censo %>%
         mutate(con_nivel = educ_ninguna + educ_basica + educ_tecnica + educ_superior,
                con_etnia = etnia_ninguna + indigena + rom + raizal + palenquero + afro) %>%
         transmute(puesto,
                   educ_ninguna = educ_ninguna/con_nivel*100 , educ_tecnica = educ_tecnica/con_nivel*100 ,
                   educ_superior = educ_superior/con_nivel*100 ,
                   edad_0_19 = edad_0_19/personas*100 , edad_20_29 = edad_20_29/personas*100 ,
                   edad_45_59 = edad_45_59/personas*100 , edad_60_74 = edad_60_74/personas*100 ,
                   edad_75 = edad_75/personas*100 ,
                   mujeres = mujeres/personas*100 , edad_votar = edad_votar/personas*100 ,
                   indigena = indigena/con_etnia*100 , rom = rom/con_etnia*100 , raizal = raizal/con_etnia*100 ,
                   palenquero = palenquero/con_etnia*100 , afro = afro/con_etnia*100 ,
                   serv_agua = serv_agua/viviendas*100 , serv_alcantarillado = serv_alcantarillado/viviendas*100 ,
                   serv_energia = serv_energia/viviendas*100 , serv_gas = serv_gas/viviendas*100 ,
                   serv_internet = serv_internet/viviendas*100 ,
                   estrato_0 = estrato_0/viviendas*100 , estrato_1 = estrato_1/viviendas*100 ,
                   estrato_2 = estrato_2/viviendas*100 , estrato_3 = estrato_3/viviendas*100 ,
                   estrato_4 = estrato_4/viviendas*100 , estrato_5 = estrato_5/viviendas*100 ,
                   estrato_6 = estrato_6/viviendas*100 ,
                   hogares_por_vivienda = hogares/viviendas , personas_por_hogar = personas/hogares ,
                   log_poblacion = log(personas) , log_viviendas = log(viviendas) , log_manzanas = log(manzanas) ,
                   manzanas_compartidas = manzanas_compartidas/manzanas*100 ,
                   log_distancia = log(1 + distancia_media) ,
                   desfase = pmin(pmax(desfase , -100) , 300))

## 1.3. la base: el puesto, los dos targets de siempre y la ficha; el departamento como una sola variable
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , cod_municipio , zona , lon , lat , registrados ,
             voto_petro , gana_petro , muestra) %>%
      inner_join(ficha , by="puesto") %>%
      mutate(log_registrados = log(registrados) , rural = as.integer(zona=="rural") ,
             dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
nrow(db)

## 1.4. los ingredientes de hoy: la ficha de 37, la ficha con el departamento, y todo con las coordenadas
x_ficha <- c(setdiff(names(ficha) , "puesto") , "log_registrados" , "rural")
x_dpto <- c(x_ficha , "dpto")
x_todo <- c(x_ficha , "dpto" , "lon" , "lat")
length(x_ficha)

## 1.5. cali y su area metropolitana: la particion de la semana 2 (semilla 2026, 25% a la prueba)
df <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
set.seed(2026)
sample <- sample(x = nrow(df) , round(nrow(df)*0.25))
test <- df[sample,]
train <- df[-sample,]
c(train = nrow(train) , test = nrow(test))

## 1.6. el pais: la particion fija del curso y los 10 pliegues de la semana 3 (semilla 2026)
train_p <- db %>% subset(muestra=="entrenamiento")
test_p <- db %>% subset(muestra=="prueba")
set.seed(2026)
pliegue_p <- sample(rep(1:10 , length.out=nrow(train_p)))
c(train = nrow(train_p) , test = nrow(test_p))

##==: 2. el primer corte, a mano :==##

## 2.1. diez puestos de entrenamiento de cali y su area, al azar (semilla 2026), ordenados por educacion superior
set.seed(2026)
diez <- train[sample(nrow(train) , 10),] %>%
        select(nombre , municipio , educ_superior , voto_petro) %>%
        arrange(educ_superior)
diez

## 2.2. sin cortar: la mejor constante es la media, y lo que falta por explicar es su rss
rss_diez <- sum((diez$voto_petro - mean(diez$voto_petro))^2)
rss_diez

## 2.3. los nueve cortes posibles (el punto medio entre dos puestos seguidos): cada lado con su media
cortes <- data.frame(corte = (diez$educ_superior[1:9] + diez$educ_superior[2:10])/2 ,
                     media_izq = NA , media_der = NA , rss = NA)
for (i in 1:9){
     izq <- diez$voto_petro[1:i]
     der <- diez$voto_petro[(i+1):10]
     cortes$media_izq[i] <- mean(izq)
     cortes$media_der[i] <- mean(der)
     cortes$rss[i] <- sum((izq - mean(izq))^2) + sum((der - mean(der))^2)
}
round(cortes , 1)

## 2.4. el mejor corte; el rss tiene dos valles
cortes[which.min(cortes$rss),]
plot(cortes$corte , cortes$rss , type="b" , pch=16 , col="red" , ylim=c(0,700) ,
     xlab="donde se corta: educacion superior (%)" , ylab="rss de las dos mitades")
abline(h=rss_diez , lty=2)

## 2.5. la misma cuenta con los 257 puestos de entrenamiento: todos los cortes posibles
train_o <- train %>% arrange(educ_superior)
cortes_257 <- data.frame(corte = (train_o$educ_superior[-nrow(train_o)] + train_o$educ_superior[-1])/2 , rss = NA)
for (i in 1:nrow(cortes_257)){
     izq <- train_o$voto_petro[1:i]
     der <- train_o$voto_petro[(i+1):nrow(train_o)]
     cortes_257$rss[i] <- sum((izq - mean(izq))^2) + sum((der - mean(der))^2)
}

## entre dos puestos con la misma educacion no hay corte posible
cortes_257$rss[diff(train_o$educ_superior)==0] <- NA
mejor <- which.min(cortes_257$rss)
c(corte = cortes_257$corte[mejor] , rss = cortes_257$rss[mejor] ,
  sin_cortar = sum((train$voto_petro - mean(train$voto_petro))^2))

## la recta de la semana 2 deja menos: con una relacion casi recta, un corte no alcanza
modelo_simple <- lm(voto_petro ~ educ_superior , data=train)
sum(residuals(modelo_simple)^2)

## el rss de cada corte: un valle ancho alrededor del mejor
plot(cortes_257$corte , cortes_257$rss/1000 , type="l" , col="red" , lwd=2 ,
     xlab="donde se corta: educacion superior (%)" , ylab="rss (miles)")
abline(h=sum(residuals(modelo_simple)^2)/1000 , lty=2 , col="blue")

## 2.6. rpart hace exactamente esta cuenta: un arbol de un solo corte, con las dos medias
toco <- rpart(voto_petro ~ educ_superior , data=train ,
              control=rpart.control(maxdepth=1 , cp=0 , minsplit=2 , minbucket=1 , xval=0))
toco

## la nube, la recta y el corte (type="s" dibuja escalones)
grilla <- data.frame(educ_superior = seq(0,60,0.1))
plot(train$educ_superior , train$voto_petro , pch=16 , col="gray50" ,
     xlab="educacion superior (%)" , ylab="voto por petro (%)")
abline(modelo_simple , col="blue" , lwd=2 , lty=2)
lines(grilla$educ_superior , predict(toco , grilla) , type="s" , col="red" , lwd=3)

##==: 3. cortar otra vez: la escalera de cali :==##

## 3.1. crecer el arbol hasta hojas de 3 puestos; el juez con 10 pliegues del train (semilla 2026)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))
arbol_m <- rpart(voto_petro ~ educ_superior , data=train , control=rpart.control(cp=0 , minsplit=10 , xval=pliegue))

## 3.2. la secuencia de subarboles: rmse dentro, de cv y de prueba segun el numero de hojas
escalera <- data.frame(hojas = arbol_m$cptable[,"nsplit"] + 1 ,
                       cp = arbol_m$cptable[,"CP"] ,
                       rmse_train = NA ,
                       rmse_cv = sqrt(arbol_m$cptable[,"xerror"]*mean((train$voto_petro - mean(train$voto_petro))^2)) ,
                       rmse_test = NA)
for (i in 1:nrow(escalera)){
     podado <- prune(arbol_m , cp=escalera$cp[i])
     escalera$rmse_train[i] <- sqrt(mean((train$voto_petro - predict(podado))^2))
     escalera$rmse_test[i] <- sqrt(mean((test$voto_petro - predict(podado , test))^2))
}

## 2, 8 y 17 hojas: dentro siempre baja; la prueba baja y vuelve a subir
escalera %>% subset(hojas %in% c(2,8,17)) %>% select(hojas , rmse_train , rmse_cv , rmse_test)

## 3.3. el tamano que elige el juez, y la recta en la misma prueba: en cali gana la recta
escalera[which.min(escalera$rmse_cv),c("hojas","rmse_cv","rmse_test")]
sqrt(mean((test$voto_petro - predict(modelo_simple , test))^2))

## 3.4. tres escaleras contra la recta
par(mfrow=c(1,3))
for (h in c(2,8,17)){
     podado <- prune(arbol_m , cp=escalera$cp[escalera$hojas==h])
     plot(train$educ_superior , train$voto_petro , pch=16 , col="gray60" , main=paste(h , "hojas") ,
          xlab="educacion superior (%)" , ylab="voto por petro (%)")
     abline(modelo_simple , lty=2)
     lines(grilla$educ_superior , predict(podado , grilla) , type="s" , col="red" , lwd=2)
}
par(mfrow=c(1,1))

##==: 4. partir el mapa: un arbol sobre la longitud y la latitud :==##

## 4.1. el arbol del pais sobre las coordenadas: hojas de al menos 7 puestos, el juez con los pliegues de la semana 3
arbol_mapa <- rpart(voto_petro ~ lon + lat , data=train_p , control=rpart.control(cp=0 , minsplit=20 , xval=pliegue_p))

## 4.2. cinco hojas: cinco rectangulos y cinco promedios
mapa_5 <- prune(arbol_mapa , cp=arbol_mapa$cptable[arbol_mapa$cptable[,"nsplit"]==4,"CP"])
mapa_5

## target predicho vs target original: cada puesto recibe el promedio de su rectangulo
test_p$voto_pred <- predict(mapa_5 , test_p)
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
sqrt(mean((test_p$voto_petro - test_p$voto_pred)^2))

## los rectangulos en el mapa: cada puesto con el color de su hoja
hoja <- factor(round(predict(mapa_5) , 1))
plot(train_p$lon , train_p$lat , pch=16 , cex=0.3 , col=as.integer(hoja) , xlim=c(-79.5,-66.5) , ylim=c(-4.5,13) ,
     xlab="longitud" , ylab="latitud")
legend("bottomright" , legend=levels(hoja) , col=1:nlevels(hoja) , pch=16 , title="voto medio")

## 4.3. podado por el juez: muchos rectangulos pequenos
fila_mapa <- which.min(arbol_mapa$cptable[,"xerror"])
mapa_cv <- prune(arbol_mapa , cp=arbol_mapa$cptable[fila_mapa,"CP"])
c(hojas = arbol_mapa$cptable[fila_mapa,"nsplit"] + 1 ,
  rmse_test = sqrt(mean((test_p$voto_petro - predict(mapa_cv , test_p))^2)))

##==: 5. un corte dentro de otro: el arbol de la ficha :==##

## 5.1. la formula con las 37 variables; en la raiz hay miles de cortes candidatos, y se prueban todos
f_ficha <- as.formula(paste0("voto_petro ~ " , paste(x_ficha , collapse=" + ")))
sum(sapply(train_p[,x_ficha] , function(x) length(unique(x)) - 1))

## 5.2. el arbol grande (hojas de al menos 7 puestos) y su version de seis hojas
arbol_g <- rpart(f_ficha , data=train_p , control=rpart.control(cp=0 , minsplit=20 , xval=pliegue_p))
arbol_6 <- prune(arbol_g , cp=arbol_g$cptable[arbol_g$cptable[,"nsplit"]==5,"CP"])
arbol_6

## el gas y el estrato 2 solo aparecen en una rama: eso es una interaccion que nadie escribio
plot(arbol_6 , uniform=T , margin=0.05)
text(arbol_6 , use.n=T , cex=0.7)

## target predicho vs target original: cada puesto recibe el promedio de su hoja
test_p$voto_pred <- predict(arbol_6 , test_p)
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

## 5.3. el departamento entra entero, sin dummies: el primer corte junta a los 33 en dos bloques
f_dpto <- update(f_ficha , . ~ . + dpto)
arbol_d <- rpart(f_dpto , data=train_p , control=rpart.control(cp=0 , minsplit=20 , xval=pliegue_p))
prune(arbol_d , cp=arbol_d$cptable[2,"CP"])

##==: 6. gana_petro: la hoja da una probabilidad :==##

## 6.1. el arbol de clasificacion: el mismo corte, con el indice de gini
f_ficha_c <- as.formula(paste0("factor(gana_petro) ~ " , paste(x_ficha , collapse=" + ")))
arbol_c <- rpart(f_ficha_c , data=train_p , method="class" , control=rpart.control(cp=0.01 , xval=0))
arbol_c

## 6.2. la cuenta a mano en la raiz: la proporcion que gana, el gini y el error
p_raiz <- mean(train_p$gana_petro)
c(p = p_raiz , gini = 2*p_raiz*(1 - p_raiz) , error = min(p_raiz , 1 - p_raiz))

## 6.3. el primer corte: las dos mitades y el promedio ponderado de su impureza
var_raiz <- rownames(arbol_c$splits)[1]
umbral <- arbol_c$splits[1,"index"]
izq <- train_p$gana_petro[train_p[[var_raiz]] < umbral]
der <- train_p$gana_petro[train_p[[var_raiz]] >= umbral]
peso <- length(izq)/nrow(train_p)
c(variable = var_raiz , umbral = round(umbral , 2))
c(p_izq = mean(izq) , p_der = mean(der) ,
  gini = peso*2*mean(izq)*(1 - mean(izq)) + (1 - peso)*2*mean(der)*(1 - mean(der)) ,
  error = peso*min(mean(izq) , 1 - mean(izq)) + (1 - peso)*min(mean(der) , 1 - mean(der)))

##==: 7. hasta donde crecer: podar con el juez :==##

## 7.1. el camino de subarboles del arbol grande: rmse dentro, de cv y de prueba (tarda cerca de medio minuto)
poda <- data.frame(hojas = arbol_g$cptable[,"nsplit"] + 1 ,
                   cp = arbol_g$cptable[,"CP"] ,
                   rmse_train = sqrt(arbol_g$cptable[,"rel error"]*mean((train_p$voto_petro - mean(train_p$voto_petro))^2)) ,
                   rmse_cv = sqrt(arbol_g$cptable[,"xerror"]*mean((train_p$voto_petro - mean(train_p$voto_petro))^2)) ,
                   rmse_test = NA)
for (i in 1:nrow(poda)){
     poda$rmse_test[i] <- sqrt(mean((test_p$voto_petro - predict(prune(arbol_g , cp=poda$cp[i]) , test_p))^2))
}

## 7.2. el arbol completo, el minimo del juez y la regla de un error estandar
fila_min <- which.min(arbol_g$cptable[,"xerror"])
fila_1se <- which(arbol_g$cptable[,"xerror"] <= arbol_g$cptable[fila_min,"xerror"] + arbol_g$cptable[fila_min,"xstd"])[1]
poda[c(nrow(poda) , fila_min , fila_1se),c("hojas","rmse_train","rmse_cv","rmse_test")]

## dentro de muestra (gris) siempre pide mas hojas; el juez (azul) y la prueba (rojo), no
plot(poda$hojas , poda$rmse_train , type="l" , log="x" , col="gray50" , lwd=2 , ylim=c(10,28) ,
     xlab="numero de hojas" , ylab="rmse")
lines(poda$hojas , poda$rmse_cv , col="blue" , lwd=2)
lines(poda$hojas , poda$rmse_test , col="red" , lwd=2 , lty=2)
abline(v=poda$hojas[c(fila_min , fila_1se)] , lty=3)

##==: 8. arboles contra rectas :==##

## 8.1. las rectas de la semana 3 con los mismos 10 pliegues: la media, la ficha y la ficha con departamentos
f_ols_dpto <- update(f_ficha , . ~ . + departamento)
lineales <- data.frame(modelo = c("media del train" , "ols, la ficha de 37" , "ols, ficha + departamentos") ,
                       rmse_cv = NA , rmse_test = NA)
formulas <- list(voto_petro ~ 1 , f_ficha , f_ols_dpto)
for (i in 1:3){
     error2 <- rep(NA , nrow(train_p))
     for (j in 1:10){
          modelo_j <- lm(formulas[[i]] , data=train_p[pliegue_p!=j,])
          error2[pliegue_p==j] <- (train_p$voto_petro[pliegue_p==j] - predict(modelo_j , train_p[pliegue_p==j,]))^2
     }
     lineales$rmse_cv[i] <- sqrt(mean(error2))
     lineales$rmse_test[i] <- sqrt(mean((test_p$voto_petro - predict(lm(formulas[[i]] , data=train_p) , test_p))^2))
}
lineales

## 8.2. el lasso de la semana 3: las 1.291 columnas escritas a mano (37 nacionales + 33 constantes + 37 x 33 desviaciones)
dptos <- levels(train_p$dpto)
nombres <- c(x_ficha , paste0("dpto:" , dptos) , paste0(rep(x_ficha , each=length(dptos)) , ":" , dptos))
x_larga <- matrix(0 , nrow(train_p) , length(nombres) , dimnames=list(NULL , nombres))
x_larga_test <- matrix(0 , nrow(test_p) , length(nombres) , dimnames=list(NULL , nombres))
x_larga[,x_ficha] <- as.matrix(train_p[,x_ficha])
x_larga_test[,x_ficha] <- as.matrix(test_p[,x_ficha])
for (d in dptos){
     x_larga[,paste0("dpto:",d)] <- as.numeric(train_p$departamento==d)
     x_larga_test[,paste0("dpto:",d)] <- as.numeric(test_p$departamento==d)
     for (v in x_ficha){
          x_larga[,paste0(v,":",d)] <- train_p[[v]]*(train_p$departamento==d)
          x_larga_test[,paste0(v,":",d)] <- test_p[[v]]*(test_p$departamento==d)
     }
}

## el juez elige lambda con los mismos pliegues (en paralelo: cerca de un minuto)
lasso_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=1 , foldid=pliegue_p , parallel=T)
c(rmse_cv = sqrt(min(lasso_p$cvm)) ,
  rmse_test = sqrt(mean((test_p$voto_petro - predict(lasso_p , x_larga_test , s="lambda.min"))^2)))

## 8.3. un arbol podado por el juez con cada juego de ingredientes, y el del mapa
f_todo <- update(f_dpto , . ~ . + lon + lat)
arbol_t <- rpart(f_todo , data=train_p , control=rpart.control(cp=0 , minsplit=20 , xval=pliegue_p))
arboles <- data.frame(modelo = c("arbol, la ficha" , "arbol, ficha + departamento" ,
                                 "arbol, ficha + departamento + coordenadas" , "arbol, el mapa") ,
                      hojas = NA , rmse_cv = NA , rmse_test = NA)
lista_arboles <- list(arbol_g , arbol_d , arbol_t , arbol_mapa)
for (i in 1:4){
     arbol_i <- lista_arboles[[i]]
     fila_i <- which.min(arbol_i$cptable[,"xerror"])
     arboles$hojas[i] <- arbol_i$cptable[fila_i,"nsplit"] + 1
     arboles$rmse_cv[i] <- sqrt(arbol_i$cptable[fila_i,"xerror"]*mean((train_p$voto_petro - mean(train_p$voto_petro))^2))
     arboles$rmse_test[i] <- sqrt(mean((test_p$voto_petro - predict(prune(arbol_i , cp=arbol_i$cptable[fila_i,"CP"]) , test_p))^2))
}
arboles

## en cali gana la recta; en el pais, un arbol solo apenas empata con ella
lineales[2:3,]

##==: 9. el problema del arbol: su varianza :==##

## 9.1. veinte remuestras bootstrap del entrenamiento, con el precio por hoja que eligio el juez
cp_min <- arbol_g$cptable[fila_min,"CP"]
boot <- data.frame(b = 1:20 , raiz = NA , segundo = NA , rmse_test = NA)
pred_boot <- matrix(NA , nrow(test_p) , 20)
set.seed(2026)
for (b in 1:20){
     i <- sample(nrow(train_p) , replace=T)
     arbol_b <- rpart(f_ficha , data=train_p[i,] , control=rpart.control(cp=cp_min , minsplit=20 , xval=0))
     boot$raiz[b] <- as.character(arbol_b$frame$var[1])
     boot$segundo[b] <- paste(arbol_b$frame$var[rownames(arbol_b$frame)=="2"] ,
                              arbol_b$frame$var[rownames(arbol_b$frame)=="3"] , sep=" / ")
     pred_boot[,b] <- predict(arbol_b , test_p)
     boot$rmse_test[b] <- sqrt(mean((test_p$voto_petro - pred_boot[,b])^2))
}

## el primer corte es estable; lo de abajo cambia; cada arbol solo predice mal
table(boot$raiz)
table(boot$segundo)
range(boot$rmse_test)

## 9.2. cuanto se mueve la prediccion de un puesto de un arbol a otro, y el promedio de los veinte
median(apply(pred_boot , 1 , sd))
sqrt(mean((test_p$voto_petro - rowMeans(pred_boot))^2))

##==: 10. el bosque: promediar arboles hechos sobre remuestras :==##

## 10.1. bagging (m = 37) y bosques con otros m: 300 arboles hondos cada uno (en paralelo: un par de minutos)
## cada bosque guarda su error contra el numero de arboles, fuera de la bolsa y en la prueba
grid_m <- data.frame(m = c(1,2,4,6,9,12,18,25,37) , rmse_oob = NA , rmse_test = NA , cor_arboles = NA , rmse_arbol = NA)
bosques <- foreach(i = 1:nrow(grid_m) , .packages="randomForest") %dopar% {
           set.seed(2026)
           bosque <- randomForest(x=train_p[,x_ficha] , y=train_p$voto_petro , ntree=300 , mtry=grid_m$m[i] ,
                                  xtest=test_p[,x_ficha] , ytest=test_p$voto_petro , keep.forest=T)
           arbol_a_arbol <- predict(bosque , test_p[,x_ficha] , predict.all=T)$individual
           correlacion <- cor(arbol_a_arbol)
           list(curva_oob = sqrt(bosque$mse) ,
                curva_test = sqrt(bosque$test$mse) ,
                pred_test = bosque$test$predicted ,
                veces_fuera = mean(bosque$oob.times) ,
                cor_arboles = mean(correlacion[upper.tri(correlacion)]) ,
                rmse_arbol = median(apply(arbol_a_arbol , 2 , function(p) sqrt(mean((test_p$voto_petro - p)^2)))))
}
for (i in 1:nrow(grid_m)){
     grid_m$rmse_oob[i] <- tail(bosques[[i]]$curva_oob , 1)
     grid_m$rmse_test[i] <- tail(bosques[[i]]$curva_test , 1)
     grid_m$cor_arboles[i] <- bosques[[i]]$cor_arboles
     grid_m$rmse_arbol[i] <- bosques[[i]]$rmse_arbol
}

## 10.2. arboles malos, promedio bueno: bagging (m = 37) contra el bosque (m = 12), y el arbol podado
bagging <- bosques[[which(grid_m$m==37)]]
bosque_f <- bosques[[which(grid_m$m==12)]]
data.frame(arboles = c(1,10,50,100,300) ,
           bagging = bagging$curva_test[c(1,10,50,100,300)] ,
           bosque = bosque_f$curva_test[c(1,10,50,100,300)])
plot(1:300 , bagging$curva_test , type="l" , log="x" , col="blue" , lwd=2 , ylim=c(14,26) ,
     xlab="numero de arboles promediados" , ylab="rmse")
lines(1:300 , bosque_f$curva_test , col="red" , lwd=2)
lines(1:300 , bosque_f$curva_oob , col="red" , lty=3)
abline(h=arboles$rmse_test[1] , lty=2 , col="gray50")

## target predicho vs target original: el promedio de 300 arboles
test_p$voto_pred <- bosque_f$pred_test
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

## 10.3. el juez gratis: cada puesto queda fuera de la bolsa en cerca de un tercio de los arboles
c(veces_fuera = bosque_f$veces_fuera , rmse_oob = tail(bosque_f$curva_oob , 1) , rmse_test = tail(bosque_f$curva_test , 1))

## 10.4. la perilla m: con m chico los arboles se parecen menos pero cada uno es peor
grid_m %>% mutate(across(-m , ~ round(.x , 3)))
par(mfrow=c(1,2))
plot(grid_m$m , grid_m$rmse_test , type="b" , pch=16 , col="red" , xlab="m: variables por corte" , ylab="rmse del bosque")
lines(grid_m$m , grid_m$rmse_oob , type="b" , col="gray50" , lty=3)
plot(grid_m$m , grid_m$cor_arboles , type="b" , pch=16 , col="blue" , xlab="m: variables por corte" , ylab="correlacion entre dos arboles")
par(mfrow=c(1,1))

##==: 11. lo que el bosque no hace: salirse de los datos :==##

## 11.1. un bosque de cali y su area con una sola variable, la educacion superior
set.seed(2026)
bosque_m <- randomForest(x=train[,"educ_superior",drop=F] , y=train$voto_petro , ntree=500)

## 11.2. la recta, la cubica y el bosque hasta 80 % de educacion superior (el maximo de los datos es 58 %)
modelo_cubica <- lm(voto_petro ~ poly(educ_superior , 3 , raw=T) , data=train)
extrapola <- data.frame(educ_superior = seq(0,80,0.5))
extrapola <- extrapola %>%
             mutate(recta = predict(modelo_simple , extrapola) ,
                    cubica = predict(modelo_cubica , extrapola) ,
                    bosque = predict(bosque_m , extrapola))
extrapola %>% subset(educ_superior %in% c(20,40,60,80))
c(educ_max = max(train$educ_superior) , voto_min = min(train$voto_petro) , voto_max = max(train$voto_petro))

## el bosque nunca sale del rango de los votos de entrenamiento: mas alla de los datos, se queda plano
plot(train$educ_superior , train$voto_petro , pch=16 , col="gray60" , xlim=c(0,80) , ylim=c(-40,100) ,
     xlab="educacion superior (%)" , ylab="voto predicho (%)")
lines(extrapola$educ_superior , extrapola$recta , col="blue" , lwd=2)
lines(extrapola$educ_superior , extrapola$cubica , col="orange" , lwd=2)
lines(extrapola$educ_superior , extrapola$bosque , col="red" , lwd=2)
abline(v=max(train$educ_superior) , lty=2)

##==: 12. el boosting a mano: aprender despacio en cali :==##

## 12.1. el algoritmo 8.2 de isl con tocones (un corte): arrancar en la media, ajustar el residuo, sumarlo encogido
lambda <- 0.1
f_train <- rep(mean(train$voto_petro) , nrow(train))
f_test <- rep(mean(train$voto_petro) , nrow(test))
f_grilla <- rep(mean(train$voto_petro) , nrow(grilla))
aprende <- data.frame(b = 1:500 , rmse_train = NA , rmse_test = NA)
curvas <- list()
for (b in 1:500){
     residuo <- train$voto_petro - f_train
     toco_b <- rpart(residuo ~ educ_superior , data=data.frame(residuo = residuo , educ_superior = train$educ_superior) ,
                     control=rpart.control(maxdepth=1 , cp=0 , minsplit=10 , minbucket=5 , xval=0))
     f_train <- f_train + lambda*predict(toco_b)
     f_test <- f_test + lambda*predict(toco_b , test)
     f_grilla <- f_grilla + lambda*predict(toco_b , grilla)
     aprende$rmse_train[b] <- sqrt(mean((train$voto_petro - f_train)^2))
     aprende$rmse_test[b] <- sqrt(mean((test$voto_petro - f_test)^2))
     if (b %in% c(1,10,50,500)) curvas[[as.character(b)]] <- f_grilla
}

## 12.2. dentro de muestra el error solo baja; en la prueba toca fondo y vuelve a subir
aprende[c(1,10,50,100,200,500),]
aprende[which.min(aprende$rmse_test),]

## las cuatro escaleras: 1, 10, 50 y 500 tocones
plot(train$educ_superior , train$voto_petro , pch=16 , col="gray60" ,
     xlab="educacion superior (%)" , ylab="voto por petro (%)")
colores <- c("gray40" , "lightblue" , "blue" , "red")
for (i in 1:4){
     lines(grilla$educ_superior , curvas[[i]] , type="s" , col=colores[i] , lwd=2)
}
legend("topright" , legend=paste(names(curvas) , "tocones") , col=colores , lwd=2)

##==: 13. el boosting con xgboost: cuantas rondas y que tan rapido :==##

## 13.1. el juez de xgboost: validacion cruzada con los pliegues que se le pasen, y las predicciones fuera
## del pliegue en la mejor ronda (con parada temprana: se detiene tras 50 rondas sin mejorar). un hilo por
## tarea: el paralelo va por fuera, con foreach
pliegues_p <- lapply(1:10 , function(j) which(pliegue_p==j))
cv_xgb <- function(x , y , pliegues , objetivo , d , eta , rondas=10000 , parada=T){
          parametros <- xgb.params(objective=objetivo , eta=eta , max_depth=d , subsample=0.8 , colsample_bytree=0.8 ,
                                   eval_metric=ifelse(objetivo=="binary:logistic" , "logloss" , "rmse") , nthread=1)
          set.seed(2026)
          cv <- xgb.cv(params=parametros , data=xgb.DMatrix(x , label=y) , nrounds=rondas , folds=pliegues ,
                       early_stopping_rounds=if (parada) 50 else NULL , verbose=0 ,
                       callbacks=list(xgb.cb.cv.predict(save_models=T)))
          if (parada) mejor <- cv$early_stop$best_iteration else mejor <- which.min(cv$evaluation_log$test_rmse_mean)
          oof <- rep(NA , length(y))
          for (j in 1:length(pliegues)){
               oof[pliegues[[j]]] <- predict(cv$cv_predict$models[[j]] , xgb.DMatrix(x[pliegues[[j]],]) , iterationrange=c(1 , mejor))
          }
          return(list(rondas=mejor , oof=oof , log=cv$evaluation_log , parametros=parametros))
}

## 13.2. la ficha con d = 6 (el valor por defecto de xgboost): tasa 0,3 contra 0,05, 2.000 rondas sin parar
## (en paralelo: un par de minutos)
x_ficha_m <- as.matrix(train_p[,x_ficha])
x_ficha_t <- as.matrix(test_p[,x_ficha])
tasas <- foreach(eta = c(0.3 , 0.05) , .packages="xgboost") %dopar% {
         cv_xgb(x_ficha_m , train_p$voto_petro , pliegues_p , "reg:squarederror" , 6 , eta , rondas=2000 , parada=F)
}

## la mejor ronda, el error del juez en ella y en la ronda 2.000
data.frame(eta = c(0.3 , 0.05) ,
           mejor_ronda = c(tasas[[1]]$rondas , tasas[[2]]$rondas) ,
           rmse_cv = c(sqrt(mean((train_p$voto_petro - tasas[[1]]$oof)^2)) , sqrt(mean((train_p$voto_petro - tasas[[2]]$oof)^2))) ,
           rmse_cv_2000 = c(tasas[[1]]$log$test_rmse_mean[2000] , tasas[[2]]$log$test_rmse_mean[2000]))

## el juez (izquierda) tiene un minimo; dentro de muestra (derecha) el error siempre baja
par(mfrow=c(1,2))
plot(1:2000 , tasas[[1]]$log$test_rmse_mean , type="l" , log="x" , col="red" , lwd=2 , ylim=c(15,20) ,
     xlab="rondas" , ylab="rmse de cv")
lines(1:2000 , tasas[[2]]$log$test_rmse_mean , col="blue" , lwd=2)
plot(1:2000 , tasas[[1]]$log$train_rmse_mean , type="l" , log="x" , col="red" , lwd=2 , ylim=c(0,28) ,
     xlab="rondas" , ylab="rmse dentro de muestra")
lines(1:2000 , tasas[[2]]$log$train_rmse_mean , col="blue" , lwd=2)
par(mfrow=c(1,1))

## 13.3. el boosting de la ficha: tasa 0,05 con las rondas que eligio el juez
set.seed(2026)
boost_ficha <- xgb.train(params=tasas[[2]]$parametros , data=xgb.DMatrix(x_ficha_m , label=train_p$voto_petro) ,
                         nrounds=tasas[[2]]$rondas , verbose=0)
sqrt(mean((test_p$voto_petro - predict(boost_ficha , xgb.DMatrix(x_ficha_t)))^2))

##==: 14. cuanto valen las interacciones: la profundidad de cada arbol :==##

## 14.1. los ingredientes de la semana 3: la ficha y el departamento (33 columnas de ceros y unos para xgboost)
x_dpto_m <- model.matrix(~ . - 1 , data=train_p[,x_dpto])
x_dpto_t <- model.matrix(~ . - 1 , data=test_p[,x_dpto])
dim(x_dpto_m)

## 14.2. el mismo boosting con d = 1 (sin cruces) hasta d = 8: tasa 0,05 y parada temprana (en paralelo: un par de minutos)
grid_d <- data.frame(d = c(1,2,3,4,6,8) , rondas = NA , rmse_cv = NA)
jueces_d <- foreach(i = 1:nrow(grid_d) , .packages="xgboost") %dopar% {
            cv_xgb(x_dpto_m , train_p$voto_petro , pliegues_p , "reg:squarederror" , grid_d$d[i] , 0.05)
}
for (i in 1:nrow(grid_d)){
     grid_d$rondas[i] <- jueces_d[[i]]$rondas
     grid_d$rmse_cv[i] <- sqrt(mean((train_p$voto_petro - jueces_d[[i]]$oof)^2))
}
grid_d

## con d = 1 el modelo es aditivo (una curva por variable): queda donde la recta con departamentos
plot(grid_d$d , grid_d$rmse_cv , type="b" , pch=16 , col="red" , ylim=c(12.5,16.5) ,
     xlab="d: profundidad de cada arbol" , ylab="rmse de cv")
abline(h=lineales$rmse_cv[3] , lty=2 , col="gray50")
abline(h=sqrt(min(lasso_p$cvm)) , lty=2 , col="blue")

## lo que valen los cruces de la maquina (d = 1 contra d = 6) y los que escribimos a mano (ols con departamentos contra el lasso)
c(maquina = grid_d$rmse_cv[1] - grid_d$rmse_cv[5] , a_mano = lineales$rmse_cv[3] - sqrt(min(lasso_p$cvm)))

## 14.3. el boosting con d = 6 y con d = 1, en la prueba
set.seed(2026)
boost_dpto <- xgb.train(params=jueces_d[[5]]$parametros , data=xgb.DMatrix(x_dpto_m , label=train_p$voto_petro) ,
                        nrounds=jueces_d[[5]]$rondas , verbose=0)
set.seed(2026)
boost_d1 <- xgb.train(params=jueces_d[[1]]$parametros , data=xgb.DMatrix(x_dpto_m , label=train_p$voto_petro) ,
                      nrounds=jueces_d[[1]]$rondas , verbose=0)
c(d_1 = sqrt(mean((test_p$voto_petro - predict(boost_d1 , xgb.DMatrix(x_dpto_t)))^2)) ,
  d_6 = sqrt(mean((test_p$voto_petro - predict(boost_dpto , xgb.DMatrix(x_dpto_t)))^2)))

## target predicho vs target original: el boosting con los ingredientes de la semana 3
test_p$voto_pred <- predict(boost_dpto , xgb.DMatrix(x_dpto_t))
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

##==: 15. la carrera: los ingredientes de la semana 3, el mapa y el promedio :==##

## 15.1. los bosques con ficha + departamento y con todo (500 arboles, m = p/3), y los mismos para gana_petro
## (m = raiz de p; la probabilidad es la fraccion de votos). en paralelo: un par de minutos
grid_b <- data.frame(ingredientes = c("dpto","todo","dpto","todo") , target = c("voto","voto","gana","gana"))
bosques_c <- foreach(i = 1:nrow(grid_b) , .packages="randomForest") %dopar% {
             if (grid_b$ingredientes[i]=="dpto") vars <- x_dpto else vars <- x_todo
             set.seed(2026)
             if (grid_b$target[i]=="voto"){
                 bosque <- randomForest(x=train_p[,vars] , y=train_p$voto_petro , ntree=500 , mtry=floor(length(vars)/3) ,
                                        xtest=test_p[,vars] , ytest=test_p$voto_petro)
                 salida <- list(oob = sqrt(tail(bosque$mse , 1)) , test = sqrt(tail(bosque$test$mse , 1)))
             } else {
                 bosque <- randomForest(x=train_p[,vars] , y=factor(train_p$gana_petro) , ntree=500 ,
                                        xtest=test_p[,vars] , ytest=factor(test_p$gana_petro))
                 salida <- list(oob = bosque$votes[,"1"] , test = bosque$test$votes[,"1"])
             }
             salida
}
c(dpto_oob = bosques_c[[1]]$oob , dpto_test = bosques_c[[1]]$test , todo_oob = bosques_c[[2]]$oob , todo_test = bosques_c[[2]]$test)

## 15.2. el boosting con coordenadas, y el juez por municipios enteros de la semana 3 (en paralelo: un par de minutos)
municipios <- unique(train_p$cod_municipio)
set.seed(2026)
grupo <- sample(rep(1:10 , length.out=length(municipios)))
pliegue_m <- grupo[match(train_p$cod_municipio , municipios)]
pliegues_m <- lapply(1:10 , function(j) which(pliegue_m==j))
x_todo_m <- model.matrix(~ . - 1 , data=train_p[,x_todo])
x_todo_t <- model.matrix(~ . - 1 , data=test_p[,x_todo])
grid_j <- data.frame(ingredientes = c("todo","dpto","todo") , pliegues = c("al azar","por municipio","por municipio"))
jueces <- foreach(i = 1:nrow(grid_j) , .packages="xgboost") %dopar% {
          if (grid_j$ingredientes[i]=="dpto") x <- x_dpto_m else x <- x_todo_m
          if (grid_j$pliegues[i]=="al azar") pl <- pliegues_p else pl <- pliegues_m
          cv_xgb(x , train_p$voto_petro , pl , "reg:squarederror" , 6 , 0.05)
}
set.seed(2026)
boost_todo <- xgb.train(params=jueces[[1]]$parametros , data=xgb.DMatrix(x_todo_m , label=train_p$voto_petro) ,
                        nrounds=jueces[[1]]$rondas , verbose=0)

## 15.3. k-nn en el mapa de la semana 3 (k = 7): la funcion de las semanas 2 y 3
knn_reg <- function(coord_train , y_train , coord_nueva , k){
           pred <- matrix(NA , nrow(coord_nueva) , length(k))
           for (j in 1:nrow(coord_nueva)){
                dist <- (coord_train[,1] - coord_nueva[j,1])^2 + (coord_train[,2] - coord_nueva[j,2])^2
                vecinos <- y_train[order(dist)[1:max(k)]]
                pred[j,] <- cumsum(vecinos)[k]/k
           }
           return(pred)
}
coord_p <- as.matrix(train_p[,c("lon","lat")])
coord_t <- as.matrix(test_p[,c("lon","lat")])
pred_knn <- knn_reg(coord_p , train_p$voto_petro , coord_t , 7)[,1]

## el k-nn fuera del pliegue, con los pliegues al azar y por municipio (tarda cerca de medio minuto)
oof_knn <- rep(NA , nrow(train_p))
oof_knn_m <- rep(NA , nrow(train_p))
for (j in 1:10){
     oof_knn[pliegue_p==j] <- knn_reg(coord_p[pliegue_p!=j,] , train_p$voto_petro[pliegue_p!=j] , coord_p[pliegue_p==j,] , 7)[,1]
     oof_knn_m[pliegue_m==j] <- knn_reg(coord_p[pliegue_m!=j,] , train_p$voto_petro[pliegue_m!=j] , coord_p[pliegue_m==j,] , 7)[,1]
}

## 15.4. promediar dos modelos que se equivocan distinto: el boosting sin mapa y el mapa (los pesos no se estiman)
pred_boost_dpto <- predict(boost_dpto , xgb.DMatrix(x_dpto_t))
pred_prom <- (pred_boost_dpto + pred_knn)/2
cor(test_p$voto_petro - pred_boost_dpto , test_p$voto_petro - pred_knn)

## target predicho vs target original: el promedio
test_p$voto_pred <- pred_prom
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

## 15.5. el juez al azar y por municipios: el mapa pierde mas cuando el municipio es nuevo
juez_municipio <- data.frame(modelo = c("k-nn en el mapa (k = 7)" , "boosting, ficha + departamento" ,
                                        "boosting, ficha + departamento + coordenadas" , "promedio: boosting sin mapa y k-nn") ,
                             al_azar = c(sqrt(mean((train_p$voto_petro - oof_knn)^2)) ,
                                         sqrt(mean((train_p$voto_petro - jueces_d[[5]]$oof)^2)) ,
                                         sqrt(mean((train_p$voto_petro - jueces[[1]]$oof)^2)) ,
                                         sqrt(mean((train_p$voto_petro - (jueces_d[[5]]$oof + oof_knn)/2)^2))) ,
                             por_municipio = c(sqrt(mean((train_p$voto_petro - oof_knn_m)^2)) ,
                                               sqrt(mean((train_p$voto_petro - jueces[[2]]$oof)^2)) ,
                                               sqrt(mean((train_p$voto_petro - jueces[[3]]$oof)^2)) ,
                                               sqrt(mean((train_p$voto_petro - (jueces[[2]]$oof + oof_knn_m)/2)^2))))
juez_municipio %>% mutate(aumento = round(por_municipio - al_azar,1) , al_azar = round(al_azar,1) , por_municipio = round(por_municipio,1))

## 15.6. la tabla de la carrera: rmse de cv (fuera de la bolsa en los bosques) y de prueba
carrera <- data.frame(modelo = c(lineales$modelo , "lasso, 1.291 columnas a mano (semana 3)" , arboles$modelo ,
                                 "bosque, la ficha (m = 12)" , "bosque, ficha + departamento" , "bosque, ficha + departamento + coordenadas" ,
                                 "boosting, la ficha" , "boosting, ficha + departamento" , "boosting, ficha + departamento + coordenadas" ,
                                 "k-nn en el mapa (k = 7)" , "promedio: boosting sin mapa y k-nn") ,
                      rmse_cv = c(lineales$rmse_cv , sqrt(min(lasso_p$cvm)) , arboles$rmse_cv ,
                                  tail(bosque_f$curva_oob , 1) , bosques_c[[1]]$oob , bosques_c[[2]]$oob ,
                                  sqrt(mean((train_p$voto_petro - tasas[[2]]$oof)^2)) , juez_municipio$al_azar[2] , juez_municipio$al_azar[3] ,
                                  juez_municipio$al_azar[1] , juez_municipio$al_azar[4]) ,
                      rmse_test = c(lineales$rmse_test ,
                                    sqrt(mean((test_p$voto_petro - predict(lasso_p , x_larga_test , s="lambda.min"))^2)) , arboles$rmse_test ,
                                    tail(bosque_f$curva_test , 1) , bosques_c[[1]]$test , bosques_c[[2]]$test ,
                                    sqrt(mean((test_p$voto_petro - predict(boost_ficha , xgb.DMatrix(x_ficha_t)))^2)) ,
                                    sqrt(mean((test_p$voto_petro - pred_boost_dpto)^2)) ,
                                    sqrt(mean((test_p$voto_petro - predict(boost_todo , xgb.DMatrix(x_todo_t)))^2)) ,
                                    sqrt(mean((test_p$voto_petro - pred_knn)^2)) ,
                                    sqrt(mean((test_p$voto_petro - pred_prom)^2))))
carrera %>% mutate(rmse_cv = round(rmse_cv,1) , rmse_test = round(rmse_test,1)) %>% arrange(rmse_test)

##==: 16. la otra pregunta: gana_petro :==##

## 16.1. la auc de la semana 4: la probabilidad de ordenar bien un par (un puesto que gana y uno que pierde)
auc <- function(y , p){
       r <- rank(p)
       n1 <- sum(y==1)
       n0 <- sum(y==0)
       return((sum(r[y==1]) - n1*(n1 + 1)/2)/(n1*n0))
}

## 16.2. un arbol de clasificacion con ficha + departamento, podado por el juez
f_dpto_c <- update(f_ficha_c , . ~ . + dpto)
arbol_cd <- rpart(f_dpto_c , data=train_p , method="class" , control=rpart.control(cp=0 , minsplit=20 , xval=pliegue_p))
arbol_cd <- prune(arbol_cd , cp=arbol_cd$cptable[which.min(arbol_cd$cptable[,"xerror"]),"CP"])

## 16.3. el boosting para gana_petro: la log-loss de la semana 4 como perdida y como juez (en paralelo: un minuto)
jueces_c <- foreach(ingredientes = c("dpto","todo") , .packages="xgboost") %dopar% {
            if (ingredientes=="dpto") x <- x_dpto_m else x <- x_todo_m
            cv_xgb(x , train_p$gana_petro , pliegues_p , "binary:logistic" , 6 , 0.05)
}
set.seed(2026)
boost_cd <- xgb.train(params=jueces_c[[1]]$parametros , data=xgb.DMatrix(x_dpto_m , label=train_p$gana_petro) ,
                      nrounds=jueces_c[[1]]$rondas , verbose=0)
set.seed(2026)
boost_ct <- xgb.train(params=jueces_c[[2]]$parametros , data=xgb.DMatrix(x_todo_m , label=train_p$gana_petro) ,
                      nrounds=jueces_c[[2]]$rondas , verbose=0)

## 16.4. k-nn de la semana 4 (k = 14): la fraccion de vecinos donde gano petro
p_knn <- knn_reg(coord_p , train_p$gana_petro , coord_t , 14)[,1]
oof_knn_c <- rep(NA , nrow(train_p))
for (j in 1:10){
     oof_knn_c[pliegue_p==j] <- knn_reg(coord_p[pliegue_p!=j,] , train_p$gana_petro[pliegue_p!=j] , coord_p[pliegue_p==j,] , 14)[,1]
}

## 16.5. las probabilidades de cada modelo, en la prueba y fuera del pliegue (o de la bolsa)
p_boost_cd <- predict(boost_cd , xgb.DMatrix(x_dpto_t))
probs_test <- list(arbol = predict(arbol_cd , test_p)[,"1"] , knn = p_knn ,
                   bosque_dpto = bosques_c[[3]]$test , boost_dpto = p_boost_cd ,
                   bosque_todo = bosques_c[[4]]$test , boost_todo = predict(boost_ct , xgb.DMatrix(x_todo_t)) ,
                   promedio = (p_boost_cd + p_knn)/2)
probs_cv <- list(arbol = NULL , knn = oof_knn_c ,
                 bosque_dpto = bosques_c[[3]]$oob , boost_dpto = jueces_c[[1]]$oof ,
                 bosque_todo = bosques_c[[4]]$oob , boost_todo = jueces_c[[2]]$oof ,
                 promedio = (jueces_c[[1]]$oof + oof_knn_c)/2)

## 16.6. ordenar (auc), cuantificar (brier) y decidir: el costo de la campana con los precios de la semana 4
## (dar por seguro un puesto que se pierde cuesta 5; trabajar uno ya ganado cuesta 1; umbral 5/6)
clasif <- data.frame(modelo = names(probs_test) , auc_cv = NA , auc_test = NA , brier = NA , costo = NA)
for (i in 1:nrow(clasif)){
     p <- probs_test[[i]]
     clasif$auc_test[i] <- auc(test_p$gana_petro , p)
     clasif$brier[i] <- mean((p - test_p$gana_petro)^2)
     clasif$costo[i] <- 5*sum(p>=5/6 & test_p$gana_petro==0) + 1*sum(p<5/6 & test_p$gana_petro==1)
     if (!is.null(probs_cv[[i]])) clasif$auc_cv[i] <- auc(train_p$gana_petro , probs_cv[[i]])
}
clasif %>% mutate(auc_cv = round(auc_cv,3) , auc_test = round(auc_test,3) , brier = round(brier,3)) %>% arrange(costo)

##==: 17. tabla final :==##

## el voto: rmse de prueba de todos los modelos de hoy y de la semana 3, ordenados
carrera %>% mutate(rmse_cv = round(rmse_cv,1) , rmse_test = round(rmse_test,1)) %>% arrange(rmse_test) %>% head(8)

## gana petro: auc, brier y costo de la campana
clasif %>% mutate(auc_cv = round(auc_cv,3) , auc_test = round(auc_test,3) , brier = round(brier,3)) %>% arrange(desc(auc_test))

## el mejor del curso hasta hoy: target predicho vs target original, y la diagonal
plot(test_p$voto_petro , pred_prom , pch=16 , col="blue" , cex=0.5 , xlim=c(0,100) , ylim=c(0,100) ,
     xlab="target original (%)" , ylab="target predicho (%)")
abline(a=0 , b=1 , lty=2)

## stop cluster
stopCluster(cl)
