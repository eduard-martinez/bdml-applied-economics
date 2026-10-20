## Big Data y Machine Learning para Economia Aplicada
## week-04-application: lpm y logit, el umbral tau, la matriz de confusion y la roc

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

## 1.2. la base de hoy: gana_petro (1 si Petro saco mas votos que Hernandez) y las 37 del censo
db <- puestos %>%
      select(puesto , nombre , municipio , departamento , gana_petro , muestra) %>%
      inner_join(censo37 , by="puesto")

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 bajo llave)
test <- db %>% subset(muestra=="prueba")
train <- db %>% subset(muestra=="entrenamiento")
c(train = nrow(train) , test = nrow(test))

## 1.4. el modelo de hoy: las 37 del censo + las dummies de departamento
x_censo <- setdiff(names(censo37) , "puesto")
f <- as.formula(paste0("gana_petro ~ " , paste(x_censo , collapse=" + ") , " + departamento"))

##==: 2. dos modelos, dos predicciones :==##

## 2.1. el lpm: ols sobre la dummy (un segundo). beta se lee en puntos de probabilidad
lpm <- lm(f , data=train)

## 2.2. el logit: la misma formula dentro de la S (unos segundos)
logit <- glm(f , family=binomial , data=train)

## 2.3. las predicciones en la prueba: una columna por modelo
test$p_lpm <- predict(lpm , test)
test$p_logit <- predict(logit , test , type="response")
test %>%
     select(nombre , municipio , gana_petro , p_lpm , p_logit) %>%
     mutate(p_lpm = round(p_lpm,3) , p_logit = round(p_logit,3)) %>%
     head(8)

## 2.4. el lpm se sale de [0, 1]; el logit queda siempre adentro
sum(test$p_lpm < 0 | test$p_lpm > 1)
range(test$p_lpm)
range(test$p_logit)

## 2.5. los dos, puesto por puesto
plot(test$p_lpm , test$p_logit , pch=16 , cex=0.4 , col="gray40" ,
     xlab="lpm: probabilidad 'lineal'" , ylab="logit: probabilidad")
abline(v=c(0,1) , lty=2 , col="red")

##==: 3. de la probabilidad a la clase: el umbral tau = 0,5 :==##

## 3.1. la regla: declarar "gana" si p >= 0,5 (la clase la ponemos nosotros)
test$clase_05 <- ifelse(test$p_logit >= 0.5 , 1 , 0)

## 3.2. la matriz de confusion: filas lo que paso, columnas lo que dijo la regla
table(observado = test$gana_petro , predicho = test$clase_05)

## 3.3. los cuatro conteos, con nombre
tp <- sum(test$clase_05==1 & test$gana_petro==1)   ## dijo gana y gano
fn <- sum(test$clase_05==0 & test$gana_petro==1)   ## dijo pierde y gano
fp <- sum(test$clase_05==1 & test$gana_petro==0)   ## dijo gana y perdio
tn <- sum(test$clase_05==0 & test$gana_petro==0)   ## dijo pierde y perdio

## 3.4. las cuatro metricas (recall y sensitivity son la misma)
c(accuracy = (tp + tn)/nrow(test) ,
  sensitivity = tp/(tp + fn) ,
  specificity = tn/(tn + fp) ,
  precision = tp/(tp + fp))

##==: 4. mover tau: las mismas probabilidades, otra decision :==##

## 4.1. guardamos las probabilidades para comprobar que no se tocan
p_guardada <- test$p_logit

## 4.2. el tau de la campana: confiarse y perder (fp) cuesta 5, trabajar de mas (fn) cuesta 1
c_fp <- 5
c_fn <- 1
tau_estrella <- c_fp/(c_fp + c_fn)
tau_estrella

## 4.3. la unica linea nueva es la regla; el modelo no se reestima
test$clase_083 <- ifelse(test$p_logit >= tau_estrella , 1 , 0)

## las probabilidades no se tocaron
identical(test$p_logit , p_guardada)

## 4.4. la matriz se mueve entera
table(observado = test$gana_petro , predicho = test$clase_083)

## 4.5. quienes cambian de clase: los puestos con p entre 0,5 y 0,83
cambian <- test %>% subset(p_logit >= 0.5 & p_logit < tau_estrella)
nrow(cambian)
cambian %>%
     select(nombre , municipio , gana_petro , p_logit , clase_05 , clase_083) %>%
     mutate(p_logit = round(p_logit,3)) %>%
     head(8)

## 4.6. las metricas con cada tau, lado a lado
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

## 5.2. la curva de costo: el minimo cae al lado del 0,83 de la formula
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

## 6.2. las dos curvas: cada punto es un tau; mover tau camina sobre la curva
plot(1 - roc_logit$specificities , roc_logit$sensitivities , type="l" , col="blue" , lwd=2 ,
     xlab="tasa de fp (1 - specificity)" , ylab="tasa de tp (sensitivity)")
lines(1 - roc_lpm$specificities , roc_lpm$sensitivities , col="red" , lwd=2)
abline(a=0 , b=1 , lty=2 , col="gray60")
legend("bottomright" , legend=c("logit" , "lpm") , col=c("blue","red") , lty=1 , bty="n")

## 6.3. el auc: el area bajo cada curva, sin obligarnos a elegir un tau
auc(roc_logit)
auc(roc_lpm)
