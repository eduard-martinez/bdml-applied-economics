## Big Data y Machine Learning para Economia Aplicada
## week-04: clasificacion, de la probabilidad a la decision
## Last run: Oct 01, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , glmnet)

## la auc como probabilidad de ordenar bien un par (mann-whitney; empates a la mitad)
auc <- function(y , p){
       r <- rank(p)
       n1 <- sum(y==1)
       n0 <- sum(y==0)
       (sum(r[y==1]) - n1*(n1 + 1)/2)/(n1*n0)
}

## la matriz de confusion y sus metricas, con el umbral tau
matriz <- function(y , p , tau = 0.5){
          yhat <- as.integer(p >= tau)
          vp <- sum(yhat==1 & y==1) ; fn <- sum(yhat==0 & y==1)
          fp <- sum(yhat==1 & y==0) ; vn <- sum(yhat==0 & y==0)
          c(vp = vp , fn = fn , fp = fp , vn = vn ,
            exactitud = (vp + vn)/length(y) , sensibilidad = vp/(vp + fn) ,
            especificidad = vn/(vn + fp) , precision = vp/(vp + fp))
}

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-04/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)

## 1.2. la base de hoy: el puesto, las dos y (el voto y gana_petro), la regla
## vigente de 2018 y las 37 variables del censo (vienen listas en censo37)
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , zona , lon , lat , registrados ,
             voto_petro , gana_petro , voto_petro_2018 , estrato_bajo , estrato_alto , servicios , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")

## 1.3. split data (la particion fija del curso: 9.602 para aprender, 2.398 bajo llave)
test <- db %>% subset(muestra=="prueba")
train <- db %>% subset(muestra=="entrenamiento")
c(train = nrow(train) , test = nrow(test))

## 1.4. los 10 pliegues del pais de la semana 3 (semilla 2026)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))

## 1.5. el censo x departamento de la semana 3: 1.291 columnas para la perilla
dptos <- sort(unique(db$departamento))
nombres <- c(x_censo , paste0("dpto:" , dptos) , paste0(rep(x_censo , each=length(dptos)) , ":" , dptos))
x_larga <- matrix(0 , nrow(train) , length(nombres) , dimnames=list(NULL , nombres))
x_larga_test <- matrix(0 , nrow(test) , length(nombres) , dimnames=list(NULL , nombres))
x_larga[,x_censo] <- as.matrix(train[,x_censo])
x_larga_test[,x_censo] <- as.matrix(test[,x_censo])
for (d in dptos){
     x_larga[,paste0("dpto:",d)] <- as.numeric(train$departamento==d)
     x_larga_test[,paste0("dpto:",d)] <- as.numeric(test$departamento==d)
     for (v in x_censo){
          x_larga[,paste0(v,":",d)] <- train[[v]]*(train$departamento==d)
          x_larga_test[,paste0(v,":",d)] <- test[[v]]*(test$departamento==d)
     }
}
dim(x_larga)

##==: 2. la y, antes de cualquier modelo :==##

## 2.1. la y es un 0 o un 1: gana_petro vale 1 si Petro saco mas votos que Hernandez
table(train$gana_petro)
c(entrenamiento = mean(train$gana_petro) , prueba = mean(test$gana_petro))

## las clases estan casi parejas: la distribucion en barras
barplot(table(train$gana_petro)/nrow(train) , names.arg=c("gana Hernandez (y = 0)","gana Petro (y = 1)") ,
        col=c("gray70","lightblue") , ylab="fraccion de los puestos de entrenamiento")

## 2.2. referencia 1, sin un solo dato: decir "gana en todos" acierta la prevalencia
mean(test$gana_petro)

## 2.3. referencia 2, sin modelo: la regla vigente, "gana donde gano en 2018"
## (solo existe en los puestos con historia; en los nuevos no hay regla)
con_historia <- !is.na(test$voto_petro_2018)
sum(con_historia)
mean((test$voto_petro_2018[con_historia] >= 50)==(test$gana_petro[con_historia]==1))

##==: 3. del lpm a la logistica (una variable, en el area metropolitana) :==##

## 3.1. los mismos 257 puestos de la semana 2 (semilla 2026, 25% a la prueba)
df <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
set.seed(2026)
sample <- sample(x = nrow(df) , round(nrow(df)*0.25))
train_m <- df[-sample,]
nrow(train_m)

## 3.2. el lpm: ols sobre la dummy estima una probabilidad lineal...
lpm <- lm(gana_petro ~ educ_superior , data=train_m)
coef(lpm)

## ...que se sale de [0, 1]: sirve para un efecto promedio, no para decidir
sum(fitted(lpm) > 1 | fitted(lpm) < 0)

## 3.3. la logistica: una recta para los log-odds, una S para la probabilidad
logit_m <- glm(gana_petro ~ educ_superior , family=binomial , data=train_m)
coef(logit_m)

## la frontera p = 0,5 (donde el indice se anula) y el grafico de las dos curvas
-coef(logit_m)[1]/coef(logit_m)[2]
plot(train_m$educ_superior , jitter(train_m$gana_petro , 0.1) , pch="|" , col="orange" , cex=0.6 ,
     xlab="educacion superior (%)" , ylab="P(gana Petro)")
abline(lpm , col="red" , lwd=2)
rejilla <- data.frame(educ_superior = 0:60)
lines(rejilla$educ_superior , predict(logit_m , rejilla , type="response") , col="blue" , lwd=2)

## 3.4. como leer la logistica: el odds ratio y el efecto que depende del punto
exp(coef(logit_m)[2])
exp(10*coef(logit_m)[2])
c(en_p_05 = abs(coef(logit_m)[2])*0.5*0.5*100 , en_p_099 = abs(coef(logit_m)[2])*0.99*0.01*100)

##==: 4. la matriz de confusion y sus metricas (el pais) :==##

## 4.1. el modelo basico del pais: la logistica con censo + dummies de departamento
f_dpto <- as.formula(paste("gana_petro ~" , paste(x_censo , collapse=" + ") , "+ departamento"))
logit_dpto <- glm(f_dpto , family=binomial , data=train)
test$p_dpto <- predict(logit_dpto , test , type="response")

## el mismo modelo juzgado con los 10 pliegues (las p de cada puesto, fuera de su pliegue)
cv_dpto <- rep(NA , nrow(train))
for (j in 1:10){
     g <- glm(f_dpto , family=binomial , data=train[pliegue!=j,])
     cv_dpto[pliegue==j] <- predict(g , train[pliegue==j,] , type="response")
}
c(auc_cv = auc(train$gana_petro , cv_dpto) , auc_test = auc(test$gana_petro , test$p_dpto))

## 4.2. la matriz con el umbral 0,5 y sus cuatro lecturas
## (sensibilidad y recall son la misma metrica: de los que gano, cuantos encontre)
round(matriz(test$gana_petro , test$p_dpto , 0.5) , 2)

## 4.3. la exactitud engana cuando una clase domina: 2010, Santos contra Mockus
p2010 <- import("input/puestos_colombia_2010.rds" , trust=T) %>%
         subset(!is.na(voto_santos) & !is.na(educ_superior) & !is.na(afro) & !is.na(estrato_bajo)) %>%
         mutate(log_registrados = log(registrados) , gana = gana_santos)
p2022 <- puestos %>%
         subset(!is.na(educ_superior) & !is.na(afro)) %>%
         mutate(log_registrados = log(registrados) , gana = gana_petro)
f_bloques <- gana ~ educ_superior + estrato_bajo + estrato_alto + servicios + edad_0_19 + edad_20_29 +
             edad_45_59 + edad_60_74 + edad_75 + mujeres + afro + indigena + log(poblacion) +
             log_registrados + zona + departamento

## la misma logistica en las dos elecciones, cada una con su prueba
eleccion <- data.frame(anio = c(2010 , 2022) , prevalencia = NA , exactitud = NA , especificidad = NA , auc = NA)
p_mockus <- NULL
for (i in 1:2){
     d <- if (i==1) p2010 else p2022
     tr <- d %>% subset(muestra=="entrenamiento")
     te <- d %>% subset(muestra=="prueba")
     g <- glm(f_bloques , family=binomial , data=tr)
     p <- predict(g , te , type="response")
     m <- matriz(te$gana , p , 0.5)
     eleccion$prevalencia[i] <- mean(te$gana)
     eleccion$exactitud[i] <- m[["exactitud"]]
     eleccion$especificidad[i] <- m[["especificidad"]]
     eleccion$auc[i] <- auc(te$gana , p)
     if (i==1){p_mockus <- 1 - p ; y_mockus <- 1 - te$gana}
}

## en 2010 "Santos gana en todos" acierta el 88%: la exactitud premia la dominancia;
## la especificidad y la auc cuentan la historia completa
eleccion %>% mutate(across(prevalencia:auc , ~round(.x , 3)))

## 4.4. cuando los positivos son raros: buscar los puestos de Mockus
## (bajar el umbral encuentra mas, al precio de mas alarmas falsas)
mockus <- data.frame(umbral = c(0.5 , 0.3 , 0.1) , alarmas = NA , recall = NA , precision = NA)
for (i in 1:nrow(mockus)){
     u <- mockus$umbral[i]
     mockus$alarmas[i] <- sum(p_mockus >= u)
     mockus$recall[i] <- mean(p_mockus[y_mockus==1] >= u)
     mockus$precision[i] <- mean(y_mockus[p_mockus >= u])
}
mockus %>% mutate(recall = round(recall , 2) , precision = round(precision , 2))

##==: 5. el umbral es una decision economica :==##

## 5.1. los costos de la campana: confiarse y perder (FP) cuesta 5; trabajar
## un puesto que igual ganaba (FN) cuesta 1. La formula da el umbral optimo
c_fp <- 5
c_fn <- 1
c_fp/(c_fp + c_fn)

## 5.2. el mismo modelo con dos umbrales: la matriz y el costo total
m_05 <- matriz(test$gana_petro , test$p_dpto , 0.5)
m_tau <- matriz(test$gana_petro , test$p_dpto , 5/6)
round(rbind(umbral_05 = m_05 , umbral_tau = m_tau) , 2)
c(costo_05 = m_05[["fp"]]*c_fp + m_05[["fn"]]*c_fn , costo_tau = m_tau[["fp"]]*c_fp + m_tau[["fn"]]*c_fn)

## 5.3. la curva de costo: el minimo cae donde dijo la formula
taus <- seq(0 , 1 , by=0.01)
costo <- rep(NA , length(taus))
for (i in 1:length(taus)){
     m <- matriz(test$gana_petro , test$p_dpto , taus[i])
     costo[i] <- m[["fp"]]*c_fp + m[["fn"]]*c_fn
}
taus[which.min(costo)]
plot(taus , costo , type="l" , col="red" , lwd=2 , xlab="umbral tau" , ylab="costo total en la prueba")
abline(v = 5/6 , lty=2 , col="blue")

## los extremos sin modelo: darlos todos por seguros o trabajarlos todos
c(todos_seguros = sum(test$gana_petro==0)*c_fp , trabajar_todos = sum(test$gana_petro==1)*c_fn)

##==: 6. los competidores: la perilla, otra vez :==##

## 6.1. el lpm con lasso: la penalizacion gaussiana de la semana 3 sobre la y de 0/1
## (tarda cerca de un minuto)
lpm_lasso <- cv.glmnet(x=x_larga , y=train$gana_petro , alpha=1 , foldid=pliegue , keep=T)
i_min <- which.min(abs(lpm_lasso$lambda - lpm_lasso$lambda.min))
i_1se <- which.min(abs(lpm_lasso$lambda - lpm_lasso$lambda.1se))
c(columnas_min = unname(lpm_lasso$nzero[i_min]) , columnas_1se = unname(lpm_lasso$nzero[i_1se]))

## ordena bien, pero sus "probabilidades" no son probabilidades
test$p_lpm <- as.numeric(predict(lpm_lasso , x_larga_test , s="lambda.min"))
c(auc_cv = auc(train$gana_petro , lpm_lasso$fit.preval[,i_min]) , auc_test = auc(test$gana_petro , test$p_lpm))
c(fuera = sum(test$p_lpm < 0 | test$p_lpm > 1) , minimo = min(test$p_lpm) , maximo = max(test$p_lpm))

## 6.2. la logistica con las variables que el lpm-lasso dejo (regla 1-se)
## ojo, regla de oro #3: la seleccion uso todo el entrenamiento y esta validacion
## hereda esa eleccion (tarda cerca de un minuto)
b_sel <- coef(lpm_lasso , s="lambda.1se")
sel <- rownames(b_sel)[as.numeric(b_sel)!=0][-1]
length(sel)
df_sel <- data.frame(y = train$gana_petro , x_larga[,sel])
df_sel_test <- data.frame(x_larga_test[,sel])
logit_sel <- suppressWarnings(glm(y ~ . , family=binomial , data=df_sel))
test$p_sel <- suppressWarnings(predict(logit_sel , df_sel_test , type="response"))
cv_sel <- rep(NA , nrow(train))
for (j in 1:10){
     g <- suppressWarnings(glm(y ~ . , family=binomial , data=df_sel[pliegue!=j,]))
     cv_sel[pliegue==j] <- suppressWarnings(predict(g , df_sel[pliegue==j,] , type="response"))
}
c(auc_cv = auc(train$gana_petro , cv_sel) , auc_test = auc(test$gana_petro , test$p_sel))

## 6.3. el lasso directamente sobre el logit: penalizar la log-verosimilitud
## (el pipeline honesto completo; tarda varios minutos)
lasso_logit <- cv.glmnet(x=x_larga , y=train$gana_petro , family="binomial" , alpha=1 , foldid=pliegue , keep=T)
il_min <- which.min(abs(lasso_logit$lambda - lasso_logit$lambda.min))
test$p_lasso <- as.numeric(predict(lasso_logit , x_larga_test , s="lambda.min" , type="response"))
cv_lasso <- as.numeric(lasso_logit$fit.preval[,il_min])
c(columnas = unname(lasso_logit$nzero[il_min]) , auc_cv = auc(train$gana_petro , cv_lasso) ,
  auc_test = auc(test$gana_petro , test$p_lasso))

## 6.4. los vecinos votan: la fraccion de los k puestos mas cercanos donde gano Petro
knn_reg <- function(coord_train , y_train , coord_nueva , k){
           pred <- matrix(NA , nrow(coord_nueva) , length(k))
           for (j in 1:nrow(coord_nueva)){
                dist <- (coord_train[,1] - coord_nueva[j,1])^2 + (coord_train[,2] - coord_nueva[j,2])^2
                vecinos <- y_train[order(dist)[1:max(k)]]
                pred[j,] <- cumsum(vecinos)[k]/k
           }
           return(pred)
}

## k por cv con la auc (con k chico los vecinos dicen 0 o 1 exactos; tarda un par de minutos)
coord <- as.matrix(train[,c("lon","lat")])
ks <- c(1:30 , 35 , 40 , 50 , 60 , 75 , 100)
pred_cv <- matrix(NA , nrow(train) , length(ks))
for (j in 1:10){
     pred_cv[pliegue==j,] <- knn_reg(coord[pliegue!=j,] , train$gana_petro[pliegue!=j] , coord[pliegue==j,] , ks)
}
auc_k <- rep(NA , length(ks))
for (i in 1:length(ks)){
     auc_k[i] <- auc(train$gana_petro , pred_cv[,i])
}
k_elegido <- ks[which.max(auc_k)]
c(k = k_elegido , auc_cv = max(auc_k))

## los vecinos en la prueba
test$p_knn <- knn_reg(coord , train$gana_petro , as.matrix(test[,c("lon","lat")]) , k_elegido)[,1]
auc(test$gana_petro , test$p_knn)

## target predicho vs target original (el lasso logistico)
test %>% select(nombre , departamento , gana_petro , p_lasso) %>% mutate(p_lasso = round(p_lasso , 2)) %>% head(5)

##==: 7. evaluar y comparar :==##

## 7.1. la curva roc del lasso logistico y de los vecinos: todos los umbrales a la vez
umbrales <- sort(unique(c(-1 , quantile(test$p_lasso , seq(0 , 1 , length.out=100)) , 2)) , decreasing=T)
roc_lasso <- data.frame(fpr = rep(NA , length(umbrales)) , tpr = NA)
for (i in 1:length(umbrales)){
     roc_lasso$fpr[i] <- mean(test$p_lasso[test$gana_petro==0] >= umbrales[i])
     roc_lasso$tpr[i] <- mean(test$p_lasso[test$gana_petro==1] >= umbrales[i])
}
plot(roc_lasso$fpr , roc_lasso$tpr , type="l" , col="red" , lwd=2 ,
     xlab="tasa de falsos positivos (1 - especificidad)" , ylab="tasa de verdaderos positivos")
abline(a=0 , b=1 , lty=2 , col="gray")

## ordenar no es decidir: el voto de 2018 como puntaje ordena muy bien...
auc(test$gana_petro[con_historia] , test$voto_petro_2018[con_historia])

## ...pero su corte de 50 decide peor que el modelo (Petro crecio entre las dos elecciones)
crecio <- test %>% subset(con_historia & voto_petro_2018 >= 40 & voto_petro_2018 < 50)
c(puestos = nrow(crecio) , gana_ahora = mean(crecio$gana_petro))

## 7.2. calibracion por deciles: cuando el modelo dice 0,8, ¿gana el 80%?
calibra <- test %>%
           mutate(decil = ntile(p_lasso , 10)) %>%
           group_by(decil) %>%
           summarise(predicha = mean(p_lasso) , observada = mean(gana_petro))
plot(calibra$predicha , calibra$observada , type="b" , pch=16 , col="red" ,
     xlim=c(0,1) , ylim=c(0,1) , xlab="probabilidad predicha (media del decil)" , ylab="fraccion que gano Petro")
abline(a=0 , b=1 , lty=2 , col="gray")

## el brier es el mse de la semana 2 aplicado a probabilidades
c(lasso = mean((test$p_lasso - test$gana_petro)^2) ,
  knn = mean((test$p_knn - test$gana_petro)^2) ,
  prevalencia = mean((mean(train$gana_petro) - test$gana_petro)^2))

## 7.3. donde falla: los puestos renidos (el error de Bayes con datos)
test %>%
     mutate(acierto = as.integer((p_lasso >= 0.5)==(gana_petro==1)) ,
            tramo = cut(abs(voto_petro - 50) , c(0 , 5 , 10 , 20 , 30 , 50) , include.lowest=T , right=F)) %>%
     group_by(tramo) %>%
     summarise(puestos = n() , exactitud = round(mean(acierto) , 2))

##==: 8. tabla final :==##

## las logisticas simples del pais, para completar la escalera
f_educ <- gana_petro ~ educ_superior
f_nueve <- gana_petro ~ educ_superior + estrato_bajo + estrato_alto + edad_20_29 + edad_60_74 +
           mujeres + afro + servicios + log_registrados
f_censo <- as.formula(paste("gana_petro ~" , paste(x_censo , collapse=" + ")))
escalera <- data.frame(modelo = c("logistica, educacion superior","logistica, las nueve variables",
                                  "logistica, las 37 del censo","logistica, censo + dummies de departamento",
                                  "lpm con lasso, censo x departamento","logistica con las seleccionadas",
                                  "lasso logistico, censo x departamento","k-nn en el mapa"),
                       auc_test = NA , exactitud = NA , brier = NA)
p_lista <- list()
for (i in 1:3){
     f <- list(f_educ , f_nueve , f_censo)[[i]]
     g <- glm(f , family=binomial , data=train)
     p_lista[[i]] <- predict(g , test , type="response")
}
p_lista[[4]] <- test$p_dpto
p_lista[[5]] <- test$p_lpm
p_lista[[6]] <- test$p_sel
p_lista[[7]] <- test$p_lasso
p_lista[[8]] <- test$p_knn
for (i in 1:8){
     escalera$auc_test[i] <- auc(test$gana_petro , p_lista[[i]])
     escalera$exactitud[i] <- mean((p_lista[[i]] >= 0.5)==(test$gana_petro==1))
     escalera$brier[i] <- mean((p_lista[[i]] - test$gana_petro)^2)
}

## ordenar (auc), decidir (exactitud con 0,5) y cuantificar (brier): ninguna sola cuenta la historia
escalera %>% mutate(auc_test = round(auc_test , 3) , exactitud = round(exactitud , 3) , brier = round(brier , 3))
