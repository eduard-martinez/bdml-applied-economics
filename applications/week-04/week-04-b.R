## Big Data y Machine Learning para Economia Aplicada
## week-04-b
## Last run: Oct 6, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , glmnet , pROC)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-04/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo_dpto <- import("input/censo_dpto_puestos_2022.rds" , trust=T)

## 1.2. la base de hoy: la y (gana_petro = 1 si Petro saco mas votos que Hernandez) y todas las
## variables, el censo x departamento de la semana 3: las 37 del censo, las 33 dummies de
## departamento y las 37 x 33 desviaciones por departamento (1.291 columnas)
db <- puestos %>%
      select(puesto , nombre , municipio , departamento , gana_petro , muestra) %>%
      inner_join(censo_dpto , by="puesto")
x_vars <- setdiff(names(db) , c("puesto","nombre","municipio","departamento","muestra","gana_petro"))
x_train <- as.matrix(train[,x_vars])
x_test <- as.matrix(test[,x_vars])

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 bajo llave)
test <- db %>% subset(muestra=="prueba")
train <- db %>% subset(muestra=="entrenamiento")
c(train = nrow(train) , test = nrow(test))

## 1.4. las matrices de glmnet y los 10 pliegues de la semana 3 (semilla 2026)
x_train <- as.matrix(select(train,-gana_petro))
x_test <- as.matrix(select(test,-gana_petro))
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))

##==: 2. el lasso por dentro: el camino de los coeficientes :==##

## 2.1. el lpm con lasso: la perilla de la semana 3, ahora sobre la y de 0/1
## glmnet ajusta un modelo por cada lambda (100 precios, del mas caro al mas barato)
camino <- glmnet(x=x_train , y=train$gana_petro , alpha=1)

## 2.2. el camino guarda una columna de coeficientes por cada lambda (1.291 filas x 100 columnas)
betas <- as.matrix(camino$beta)

## 2.3. tres paradas del camino: precio alto (la primera con 5 o mas activas), medio (20 o mas)
## y bajo (30 o mas)
parada_alta <- which(camino$df >= 5)[1]
parada_media <- which(camino$df >= 20)[1]
parada_baja <- which(camino$df >= 30)[1]
camino$lambda[c(parada_alta , parada_media , parada_baja)]
camino$df[c(parada_alta , parada_media , parada_baja)]

## precio alto: las primeras variables que compra el lasso, con su beta
b_alto <- betas[,parada_alta]
round(b_alto[b_alto!=0] , 4)

## precio medio: las 10 activas que mas pesan
b_medio <- betas[,parada_media]
b_medio <- b_medio[b_medio!=0]
round(head(b_medio[order(-abs(b_medio))] , 10) , 4)

## precio bajo: mas variables y coeficientes mas grandes (el peaje es mas barato)
b_bajo <- betas[,parada_baja]
b_bajo <- b_bajo[b_bajo!=0]
round(head(b_bajo[order(-abs(b_bajo))] , 10) , 4)

## 2.4. el grafico del camino: hacia la derecha sube lambda, los coeficientes se encogen
## y uno a uno mueren en cero exacto (arriba, cuantas variables siguen activas)
plot(camino , xvar="lambda")

##==: 3. el juez: lambda por validacion cruzada :==##

## 3.1. los mismos 10 pliegues de la semana 3 (tarda unos 3 minutos)
## el error del juez es el mse fuera del pliegue, aqui sobre la y de 0/1
cv_lpm <- cv.glmnet(x=x_train , y=train$gana_petro , alpha=1 , foldid=pliegue)

## 3.2. los dos lambda del juez: el que minimiza el error y el de la regla de 1 ee
i_min <- which(cv_lpm$lambda==cv_lpm$lambda.min)
i_1se <- which(cv_lpm$lambda==cv_lpm$lambda.1se)
data.frame(regla = c("lambda min","lambda 1se") ,
           lambda = c(cv_lpm$lambda.min , cv_lpm$lambda.1se) ,
           error_cv = c(cv_lpm$cvm[i_min] , cv_lpm$cvm[i_1se]) ,
           activas = c(cv_lpm$nzero[i_min] , cv_lpm$nzero[i_1se]))

## 3.3. los coeficientes del lambda min: cuales quedan distintos de cero y cuanto pesan
b_min <- coef(cv_lpm , s="lambda.min")
activas_min <- b_min[b_min[,1]!=0 , ][-1]
round(head(activas_min[order(-abs(activas_min))] , 35) , 4)

## 3.4. los dos objetos juntos, con el mismo eje (log lambda):
## arriba, que le pasa a los coeficientes; abajo, que le pasa al error de validacion
par(mfrow=c(2,1) , mar=c(4,4,3,1))
plot(camino , xvar="lambda")
abline(v=log(c(cv_lpm$lambda.min , cv_lpm$lambda.1se)) , lty=2)
plot(cv_lpm)
par(mfrow=c(1,1) , mar=c(5,4,4,2) + 0.1)

##==: 4. la prediccion fuera de muestra, a mano :==##

## 4.1. el vector de coeficientes del lambda min (el intercepto y los 1.291 beta) y la matriz X
## de la prueba con una columna de unos para el intercepto
## (glmnet estandariza por dentro pero devuelve los beta en las unidades originales)
beta <- as.numeric(coef(cv_lpm , s="lambda.min"))
x_test_1 <- cbind(1 , x_test)

## 4.2. y_hat = X beta: las caracteristicas de cada puesto por los coeficientes
test$score_lpm <- as.numeric(x_test_1 %*% beta)

## un puesto, la suma a mano: sum(x_ij * beta_j)
sum(x_test_1[1,]*beta)
test$score_lpm[1]

## 4.3. predict() hace exactamente esa multiplicacion (la diferencia es cero)
max(abs(test$score_lpm - as.numeric(predict(cv_lpm , x_test , s="lambda.min"))))

## el score contra lo que paso
test %>% select(nombre , municipio , gana_petro , score_lpm) %>% head(5)

##==: 5. del score a la decision: el umbral tau (lpm-lasso) :==##

## 5.1. la regla con tau = 0,5: "gana" si el score es 0,5 o mas
## la matriz de confusion: filas lo que paso, columnas lo que dijo la regla
test$score_lpm <- runif(min = 0,max = 1,n = nrow(test))
test$pred_lpm_05 <- ifelse(test$score_lpm >= 0.5 , 1 , 0)
table(observado = test$gana_petro , predicho = test$pred_lpm_05)

## los cuatro conteos y las metricas de la lamina 12 (recall y sensitivity son la misma)
tp <- sum(test$pred_lpm_05==1 & test$gana_petro==1)   ## dijo gana y gano
fn <- sum(test$pred_lpm_05==0 & test$gana_petro==1)   ## dijo pierde y gano
fp <- sum(test$pred_lpm_05==1 & test$gana_petro==0)   ## dijo gana y perdio
tn <- sum(test$pred_lpm_05==0 & test$gana_petro==0)   ## dijo pierde y perdio
c(accuracy = (tp + tn)/nrow(test) ,
  recall = tp/(tp + fn) ,
  specificity = tn/(tn + fp) ,
  precision = tp/(tp + fp))

## 5.2. la misma regla con un corte mas exigente: tau = 0,8
test$pred_lpm_08 <- ifelse(test$score_lpm >= 0.8 , 1 , 0)
table(observado = test$gana_petro , predicho = test$pred_lpm_08)
tp <- sum(test$pred_lpm_08==1 & test$gana_petro==1)
fn <- sum(test$pred_lpm_08==0 & test$gana_petro==1)
fp <- sum(test$pred_lpm_08==1 & test$gana_petro==0)
tn <- sum(test$pred_lpm_08==0 & test$gana_petro==0)
c(accuracy = (tp + tn)/nrow(test) ,
  recall = tp/(tp + fn) ,
  specificity = tn/(tn + fp) ,
  precision = tp/(tp + fp))

## 5.3. que cambio: el modelo es el mismo, solo cambio la regla. cambian de clase solo los
## puestos con score entre 0,5 y 0,8: los que ganaron pasan de tp a fn; los que perdieron, de fp a tn
cambian <- test %>% subset(score_lpm >= 0.5 & score_lpm < 0.8)
table(cambian$gana_petro)
cambian %>% select(nombre , municipio , score_lpm , gana_petro) %>% head(8)

##==: 6. el logit: como leer sus coeficientes (laminas 7 y 9) :==##

## 6.1. los mismos 257 puestos del area metropolitana de la semana 2 (semilla 2026, 25% a la prueba)
metro <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
set.seed(2026)
prueba_m <- sample(x = nrow(metro) , round(nrow(metro)*0.25))
train_m <- metro[-prueba_m,]

## 6.2. la logistica con una variable: la ventaja esperada x'beta = 6,09 - 0,136 x educ
logit_m <- glm(gana_petro ~ educ_superior , family=binomial , data=train_m)
coef(logit_m)

## 6.3. el -0,136 esta en unidades de la ventaja latente: solo el signo se lee directo
## (mas educacion superior, menos Petro). las chances: cada punto multiplica p/(1-p) por 0,87
exp(coef(logit_m)[2])

## la frontera p = 0,5: la educacion donde la ventaja esperada se anula (44,8 %)
-coef(logit_m)[1]/coef(logit_m)[2]

## la probabilidad: beta x p x (1-p), en puntos, por cada punto de educacion
coef(logit_m)[2]*0.5*0.5*100     ## en la frontera (p = 0,5): -3,4
coef(logit_m)[2]*0.99*0.01*100   ## en un puesto seguro (p = 0,99): -0,1

##==: 7. el logit con las variables que eligio el lasso :==##

## 7.1. la lista: las columnas que el lpm-lasso deja con lambda 1se (la de la lamina 21)
## (una logistica sin perilla aguanta mejor 140 columnas que 179)
b_1se <- coef(cv_lpm , s="lambda.1se")
seleccionadas <- rownames(b_1se)[b_1se[,1]!=0][-1]
length(seleccionadas)

## 7.2. la logistica con esas columnas (regla de oro #3: la lista se eligio con todo el
## entrenamiento; la prueba nunca la vio). el aviso "fitted probabilities numerically 0 or 1"
## es la separacion de la lamina 10
db_sel <- data.frame(gana_petro = train$gana_petro , x_train[,seleccionadas])
logit_sel <- glm(gana_petro ~ . , family=binomial , data=db_sel)

## 7.3. la probabilidad a mano: primero la ventaja esperada x'beta, despues la S de la lamina 7
## (p = e^xb/(1 + e^xb), escrita como 1/(1 + e^-xb), que no se desborda)
b_logit[is.na(b_logit)] <- 0
b_logit <- coef(logit_sel)
test$xb_logit <- as.numeric(cbind(1 , x_test[,seleccionadas]) %*% b_logit)
test$p_logit <- 1/(1 + exp(-test$xb_logit))

## 7.4. predict(type="response") hace exactamente lo mismo
max(abs(test$p_logit - predict(logit_sel , data.frame(x_test[,seleccionadas]) , type="response")))
test %>% select(nombre , municipio , gana_petro , xb_logit , p_logit) %>% head(5)

##==: 8. un score lineal contra una probabilidad :==##

## 8.1. el lpm-lasso se sale de [0, 1]; el logit no
sum(test$score_lpm < 0 | test$score_lpm > 1)
range(test$score_lpm)
range(test$p_logit)

## 8.2. los dos, puesto por puesto: fuera de las lineas rojas el lpm no es una probabilidad
plot(test$score_lpm , test$p_logit , pch=16 , cex=0.4 , col="gray40" ,
     xlab="score del lpm-lasso" , ylab="probabilidad del logit con las seleccionadas")
abline(v=c(0,1) , lty=2 , col="red")

##==: 9. del score a la decision: el umbral tau (logit) :==##

## 9.1. las mismas dos reglas con la probabilidad del logit
test$pred_logit_05 <- ifelse(test$p_logit >= 0.5 , 1 , 0)
test$pred_logit_08 <- ifelse(test$p_logit >= 0.8 , 1 , 0)
table(observado = test$gana_petro , predicho = test$pred_logit_05)
table(observado = test$gana_petro , predicho = test$pred_logit_08)

## 9.2. las cuatro reglas lado a lado: subir tau baja los fp y sube los fn
metricas <- data.frame(regla = c("pred_lpm_05","pred_lpm_08","pred_logit_05","pred_logit_08") ,
                       tp = NA , fn = NA , fp = NA , tn = NA)
for (i in 1:nrow(metricas)){
     pred <- test[[metricas$regla[i]]]
     metricas$tp[i] <- sum(pred==1 & test$gana_petro==1)
     metricas$fn[i] <- sum(pred==0 & test$gana_petro==1)
     metricas$fp[i] <- sum(pred==1 & test$gana_petro==0)
     metricas$tn[i] <- sum(pred==0 & test$gana_petro==0)
}
metricas %>% mutate(accuracy = round((tp + tn)/nrow(test) , 3) ,
                    recall = round(tp/(tp + fn) , 3) ,
                    specificity = round(tn/(tn + fp) , 3) ,
                    precision = round(tp/(tp + fp) , 3))

##==: 10. el umbral con costos :==##

## 10.1. los precios de la campana (lamina 16): confiarse y perder (fp) cuesta 5;
## trabajar un puesto que igual ganaba (fn) cuesta 1. la formula: tau* = c_fp/(c_fp + c_fn)
c_fp <- 5
c_fn <- 1
tau_estrella <- c_fp/(c_fp + c_fn)
tau_estrella

## 10.2. la grilla: 101 umbrales de 0 a 1 y, para cada uno, la matriz y su costo
grid <- data.frame(tau = seq(0 , 1 , by=0.01) , costo_lpm = NA , costo_logit = NA)
for (i in 1:nrow(grid)){
     pred_lpm <- ifelse(test$score_lpm >= grid$tau[i] , 1 , 0)
     pred_logit <- ifelse(test$p_logit >= grid$tau[i] , 1 , 0)
     fp_lpm <- sum(pred_lpm==1 & test$gana_petro==0)
     fn_lpm <- sum(pred_lpm==0 & test$gana_petro==1)
     fp_logit <- sum(pred_logit==1 & test$gana_petro==0)
     fn_logit <- sum(pred_logit==0 & test$gana_petro==1)
     grid$costo_lpm[i] <- fp_lpm*c_fp + fn_lpm*c_fn
     grid$costo_logit[i] <- fp_logit*c_fp + fn_logit*c_fn
}
head(grid)

## con tau = 1 el lpm todavia da puestos por seguros: su score pasa de 1
sum(test$score_lpm >= 1)

## 10.3. el costo frente a tau, con el tau* de la formula
plot(grid$tau , grid$costo_logit , type="l" , col="blue" , lwd=2 , ylim=c(0,6000) ,
     xlab="umbral tau: dar por seguro si el score es tau o mas" , ylab="costo total en la prueba")
lines(grid$tau , grid$costo_lpm , col="red" , lwd=2)
abline(v=tau_estrella , lty=2)
legend("topright" , legend=c("logit con las seleccionadas","lpm-lasso","tau* de la formula") ,
       col=c("blue","red","black") , lty=c(1,1,2) , bty="n")

## 10.4. el tau que minimiza el costo en cada modelo
min_logit <- grid %>%
             arrange(costo_logit) %>%
             mutate(ranking = row_number()) %>%
             subset(ranking==1)
min_lpm <- grid %>%
           arrange(costo_lpm) %>%
           mutate(ranking = row_number()) %>%
           subset(ranking==1)
min_logit
min_lpm

## 10.5. el costo con el tau* de la formula, en cada modelo
costo_logit_estrella <- sum(test$p_logit >= tau_estrella & test$gana_petro==0)*c_fp +
                        sum(test$p_logit < tau_estrella & test$gana_petro==1)*c_fn
costo_lpm_estrella <- sum(test$score_lpm >= tau_estrella & test$gana_petro==0)*c_fp +
                      sum(test$score_lpm < tau_estrella & test$gana_petro==1)*c_fn
c(logit = costo_logit_estrella , lpm = costo_lpm_estrella)

## la formula supone que p dice la verdad; si el modelo exagera o subestima su seguridad,
## el minimo se corre (en la lamina 17, la logistica con censo + dummies toca fondo en 0,84).
## ojo: el minimo se busca en la prueba: es un diagnostico, no una eleccion (regla de oro #2)

##==: 11. la curva roc y el auc :==##

## 11.1. la roc a mano: cada umbral da una matriz y la matriz da un punto
## (tasa de fp = 1 - specificity, tasa de tp = sensitivity); de tau alto a tau bajo
roc_mano <- data.frame(tau = c(1 , 0.95 , 0.9 , 0.8 , 0.7 , 0.5 , 0.3 , 0.1 , 0) , tpr = NA , fpr = NA)
for (i in 1:nrow(roc_mano)){
     pred <- ifelse(test$p_logit >= roc_mano$tau[i] , 1 , 0)
     tp <- sum(pred==1 & test$gana_petro==1)
     fn <- sum(pred==0 & test$gana_petro==1)
     fp <- sum(pred==1 & test$gana_petro==0)
     tn <- sum(pred==0 & test$gana_petro==0)
     roc_mano$tpr[i] <- tp/(tp + fn)
     roc_mano$fpr[i] <- fp/(fp + tn)
}
roc_mano %>% mutate(tpr = round(tpr,3) , fpr = round(fpr,3))

## pintar los nueve puntos: ya se ve la curva
plot(roc_mano$fpr , roc_mano$tpr , type="b" , pch=16 , col="blue" , xlim=c(0,1) , ylim=c(0,1) ,
     xlab="tasa de fp (1 - specificity)" , ylab="tasa de tp (sensitivity)")
text(roc_mano$fpr , roc_mano$tpr , labels=roc_mano$tau , pos=4 , cex=0.7)
abline(a=0 , b=1 , lty=2 , col="gray60")

## 11.2. con un paquete: pROC hace el mismo for con todos los umbrales posibles
## (uno por cada valor distinto del score; direction="<": mas score, mas probable que gane)
roc_logit <- roc(test$gana_petro , test$p_logit , levels=c(0,1) , direction="<")
roc_lpm <- roc(test$gana_petro , test$score_lpm , levels=c(0,1) , direction="<")
length(roc_logit$thresholds)

## las dos curvas en la misma figura, con los nueve puntos de la roc a mano encima
plot(1 - roc_logit$specificities , roc_logit$sensitivities , type="l" , col="blue" , lwd=2 ,
     xlab="tasa de fp (1 - specificity)" , ylab="tasa de tp (sensitivity)")
lines(1 - roc_lpm$specificities , roc_lpm$sensitivities , col="red" , lwd=2)
points(roc_mano$fpr , roc_mano$tpr , pch=16)
abline(a=0 , b=1 , lty=2 , col="gray60")
legend("bottomright" , legend=c("logit con las seleccionadas","lpm-lasso","puntos a mano (logit)") ,
       col=c("blue","red","black") , lty=c(1,1,NA) , pch=c(NA,NA,16) , bty="n")

## 11.3. el auc: el area bajo cada curva, que resume todos los umbrales a la vez
## (no depende de tau ni de la escala: al lpm-lasso le da igual salirse de [0, 1])
auc(roc_logit)
auc(roc_lpm)

## el auc como ordenamiento (lamina 14): se toma un puesto que gano y uno que perdio, y se
## pregunta si el que gano tiene el score mas alto; el auc es la fraccion de parejas bien ordenadas
gano <- test$p_logit[test$gana_petro==1]
perdio <- test$p_logit[test$gana_petro==0]
mean(outer(gano , perdio , ">"))
