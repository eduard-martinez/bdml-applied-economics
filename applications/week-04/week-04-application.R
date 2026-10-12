## Big Data y Machine Learning para Economia Aplicada
## week-04-application: la aplicacion de clase. Dos modelos de probabilidad (lpm y
## logit), el umbral tau y la matriz de confusion. La replica completa de las
## diapositivas es week-04-replication.R
## Last run: Oct 8, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , pROC)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-04/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)

## 1.2. la base de hoy: la y (gana_petro = 1 si Petro saco mas votos que Hernandez)
## y las 37 variables del censo
db <- puestos %>%
      select(puesto , nombre , municipio , departamento , gana_petro , muestra) %>%
      inner_join(censo37 , by="puesto")

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 bajo llave)
test <- db %>% subset(muestra=="prueba")
train <- db %>% subset(muestra=="entrenamiento")
c(train = nrow(train) , test = nrow(test))

## 1.4. el modelo de la parte 2 de las diapositivas: las 37 del censo + las dummies
## de departamento (R crea las dummies solo al ver la variable de texto)
x_censo <- setdiff(names(censo37) , "puesto")
f <- as.formula(paste0("gana_petro ~ " , paste(x_censo , collapse=" + ") , " + departamento"))

##==: 2. dos modelos, dos predicciones :==##

## 2.1. el lpm: ols sobre la dummy (un segundo). beta se lee en puntos de probabilidad
lpm <- lm(f , data=train)

## 2.2. el logit: la misma formula dentro de la S (unos segundos)
logit <- glm(f , family=binomial , data=train)

## 2.3. las predicciones en la prueba: una columna por modelo
## mira: dos numeros por puesto; parecidos en el medio, distintos en los extremos
test$p_lpm <- predict(lpm , test)
test$p_logit <- predict(logit , test , type="response")
test %>%
     select(nombre , municipio , gana_petro , p_lpm , p_logit) %>%
     mutate(p_lpm = round(p_lpm,3) , p_logit = round(p_logit,3)) %>%
     head(8)

## 2.4. el lpm se sale de [0, 1] (la advertencia de la lamina 6: 371 puestos,
## de -0,28 a 1,36); el logit queda siempre adentro
sum(test$p_lpm < 0 | test$p_lpm > 1)
range(test$p_lpm)
range(test$p_logit)

## 2.5. los dos, puesto por puesto
## mira: fuera de las lineas rojas el lpm no es una probabilidad; para ordenar
## sirve igual, para decidir con costos no
plot(test$p_lpm , test$p_logit , pch=16 , cex=0.4 , col="gray40" ,
     xlab="lpm: probabilidad 'lineal'" , ylab="logit: probabilidad")
abline(v=c(0,1) , lty=2 , col="red")

##==: 3. de la probabilidad a la clase: el umbral tau = 0,5 :==##

## 3.1. la regla: declarar "gana" si p >= 0,5 (la clase la ponemos nosotros)
test$clase_05 <- ifelse(test$p_logit >= 0.5 , 1 , 0)

## 3.2. la matriz de confusion (la de la lamina 12: TP 1.067, FN 201, FP 188, TN 942)
## filas: lo que paso; columnas: lo que dijo la regla
table(observado = test$gana_petro , predicho = test$clase_05)

## 3.3. los cuatro conteos, con nombre
tp <- sum(test$clase_05==1 & test$gana_petro==1)   ## dijo gana y gano
fn <- sum(test$clase_05==0 & test$gana_petro==1)   ## dijo pierde y gano
fp <- sum(test$clase_05==1 & test$gana_petro==0)   ## dijo gana y perdio
tn <- sum(test$clase_05==0 & test$gana_petro==0)   ## dijo pierde y perdio

## 3.4. las cuatro metricas de la lamina 12 (recall y sensitivity son la misma)
c(accuracy = (tp + tn)/nrow(test) ,
  sensitivity = tp/(tp + fn) ,
  specificity = tn/(tn + fp) ,
  precision = tp/(tp + fp))

##==: 4. mover tau: las mismas probabilidades, otra decision :==##

## 4.1. guardamos las probabilidades para comprobar que no se tocan
p_guardada <- test$p_logit

## 4.2. el tau de la campana (lamina 16): confiarse y perder (fp) cuesta 5, trabajar
## un puesto que igual ganaba (fn) cuesta 1 -> tau* = 5/(5 + 1) = 0,83
c_fp <- 5
c_fn <- 1
tau_estrella <- c_fp/(c_fp + c_fn)
tau_estrella

## 4.3. la unica linea nueva es la regla; el modelo no se reestima
test$clase_083 <- ifelse(test$p_logit >= tau_estrella , 1 , 0)

## mira: las probabilidades son identicas a las de la seccion 3
identical(test$p_logit , p_guardada)

## 4.4. la matriz se mueve entera (lamina 15: los FP caen de 188 a 45 y los FN
## suben de 201 a 470)
table(observado = test$gana_petro , predicho = test$clase_083)

## 4.5. quienes cambian de clase: los puestos con p entre 0,5 y 0,83
## mira: misma probabilidad, otra clase; cambio la exigencia, no el modelo
cambian <- test %>% subset(p_logit >= 0.5 & p_logit < tau_estrella)
nrow(cambian)
cambian %>%
     select(nombre , municipio , gana_petro , p_logit , clase_05 , clase_083) %>%
     mutate(p_logit = round(p_logit,3)) %>%
     head(8)

## 4.6. las metricas con cada tau, lado a lado
## mira: precision sube (0,85 -> 0,95), sensitivity baja (0,84 -> 0,63) y
## accuracy empeora (83,8 -> 78,5): ninguna regla gana en todo
grid <- data.frame(tau = c(0.5 , tau_estrella) , tp = NA , fn = NA , fp = NA , tn = NA)
for (i in 1:nrow(grid)){
     pred <- ifelse(test$p_logit >= grid$tau[i] , 1 , 0)
     grid$tp[i] <- sum(pred==1 & test$gana_petro==1)
     grid$fn[i] <- sum(pred==0 & test$gana_petro==1)
     grid$fp[i] <- sum(pred==1 & test$gana_petro==0)
     grid$tn[i] <- sum(pred==0 & test$gana_petro==0)
}
grid %>%
     mutate(accuracy = round((tp + tn)/nrow(test),3) ,
            sensitivity = round(tp/(tp + fn),3) ,
            specificity = round(tn/(tn + fp),3) ,
            precision = round(tp/(tp + fp),3))

##==: 5. ¿y cual tau conviene? el costo de cada umbral :==##

## 5.1. el costo total de la campana en la prueba, para cada tau de 0 a 1
costos <- data.frame(tau = seq(0 , 1 , by=0.01) , costo = NA)
for (i in 1:nrow(costos)){
     pred <- ifelse(test$p_logit >= costos$tau[i] , 1 , 0)
     costos$costo[i] <- sum(pred==1 & test$gana_petro==0)*c_fp + sum(pred==0 & test$gana_petro==1)*c_fn
}

## 5.2. la curva de costo (lamina 17)
## mira: en tau = 0 (todos seguros) cuesta 5.650; en tau = 1 (todos trabajados),
## 1.268; el minimo cae en 0,84, al lado del 0,83 de la formula
plot(costos$tau , costos$costo , type="l" , col="red" , lwd=2 ,
     xlab="umbral tau: dar por seguro si p >= tau" , ylab="costo total en la prueba")
abline(v=tau_estrella , lty=2 , col="blue")

## 5.3. el tau que minimiza el costo, y el costo de las dos reglas que ya usamos
## mira: la regla de 0,5 paga 1.141; la de 0,83 paga 695, con el mismo modelo
costos %>% arrange(costo) %>% mutate(ranking = row_number()) %>% subset(ranking==1)
c(costo_05 = sum(test$clase_05==1 & test$gana_petro==0)*c_fp + sum(test$clase_05==0 & test$gana_petro==1)*c_fn ,
  costo_083 = sum(test$clase_083==1 & test$gana_petro==0)*c_fp + sum(test$clase_083==0 & test$gana_petro==1)*c_fn)

##==: 6. todos los umbrales a la vez: la roc y el auc :==##

## 6.1. la curva roc de cada modelo (pROC recorre todos los tau posibles)
roc_logit <- roc(test$gana_petro , test$p_logit , levels=c(0,1) , direction="<")
roc_lpm <- roc(test$gana_petro , test$p_lpm , levels=c(0,1) , direction="<")

## 6.2. las dos curvas
## mira: cada punto de la curva es un tau; mover tau camina sobre la curva, no
## la cambia. El lpm ordena casi igual aunque sus p no sean probabilidades
plot(1 - roc_logit$specificities , roc_logit$sensitivities , type="l" , col="blue" , lwd=2 ,
     xlab="tasa de fp (1 - specificity)" , ylab="tasa de tp (sensitivity)")
lines(1 - roc_lpm$specificities , roc_lpm$sensitivities , col="red" , lwd=2)
abline(a=0 , b=1 , lty=2 , col="gray60")
legend("bottomright" , legend=c("logit" , "lpm") , col=c("blue","red") , lty=1 , bty="n")

## 6.3. el auc: el area bajo cada curva (la lamina 14: 0,916), que resume el
## ordenamiento sin obligarnos a elegir un tau
auc(roc_logit)
auc(roc_lpm)
