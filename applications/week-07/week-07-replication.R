## Big Data y Machine Learning para Economia Aplicada
## week-07-replication: clasificacion con redes neuronales (reproduce todas las cifras del deck)

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , nnet , doParallel)

## cluster: las redes de la validacion cruzada corren en paralelo, cada tarea con su semilla
cores <- parallel::detectCores()
cl <- makeCluster(max(1 , cores - 2))
registerDoParallel(cl)

## el auc de la semana 4 (mann-whitney), usado muchas veces hoy
auc <- function(y , p){
       r <- rank(p)
       n1 <- sum(y==1)
       n0 <- length(y) - n1
       (sum(r[y==1]) - n1*(n1+1)/2)/(n1*n0)
}

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-07/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)
primera <- import("input/puestos_primera_2022.rds" , trust=T)
p10 <- import("input/puestos_colombia_2010.rds" , trust=T)

## 1.2. la base: los dos targets de la segunda vuelta, las 37 del censo y el departamento
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , voto_petro , gana_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      mutate(dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")

## 1.3. la primera vuelta: votos por candidato y quien gano el puesto (empates fuera)
db_r1 <- db %>%
         inner_join(primera , by="puesto") %>%
         mutate(gana_primera = factor(gana_primera , levels=c("petro","hernandez","gutierrez","otros")))
c(segunda = nrow(db) , primera = nrow(db_r1))

## 1.4. split data (la particion fija del curso) y los 10 pliegues de la semana 3
train <- db %>% subset(muestra=="entrenamiento")
test <- db %>% subset(muestra=="prueba")
train_r1 <- db_r1 %>% subset(muestra=="entrenamiento")
test_r1 <- db_r1 %>% subset(muestra=="prueba")
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))
c(train = nrow(train) , test = nrow(test) , train_r1 = nrow(train_r1) , test_r1 = nrow(test_r1))

## 1.5. las 37 como matriz, estandarizadas con el entrenamiento; el departamento en 33 dummies
X <- as.matrix(train[,x_censo])
X_t <- as.matrix(test[,x_censo])
S <- scale(X)
centro <- attr(S , "scaled:center")
escala <- attr(S , "scaled:scale")
S <- matrix(S , nrow(S))
S_t <- scale(X_t , centro , escala)
S_t <- matrix(S_t , nrow(S_t))
dptos <- sort(unique(db$departamento))
D <- sapply(dptos , function(d) as.numeric(train$departamento==d))
D_t <- sapply(dptos , function(d) as.numeric(test$departamento==d))
x_red <- cbind(S , D)
x_red_t <- cbind(S_t , D_t)

##==: 2. la y de hoy: tres elecciones, tres repartos :==##

## 2.1. la segunda vuelta de 2022 (train), la de 2010 y la primera vuelta de 2022 (train)
round(100*prop.table(table(train$gana_petro)) , 1)
round(100*prop.table(table(p10$gana_santos)) , 1)
round(100*prop.table(table(train_r1$gana_primera)) , 1)

## 2.2. las tres, en barras
par(mfrow=c(1,3))
barplot(100*prop.table(table(factor(train$gana_petro , 1:0 , c("petro","hdez")))) , col=c("blue","orange") , ylim=c(0,100) , main="2022-II" , ylab="% de los puestos que gana")
abline(h=50 , lty=2)
barplot(100*prop.table(table(factor(p10$gana_santos , 1:0 , c("santos","mockus")))) , col=c("gray60","darkgreen") , ylim=c(0,100) , main="2010-II")
abline(h=50 , lty=2)
barplot(100*prop.table(table(train_r1$gana_primera)) , col=c("blue","orange","red","gray") , ylim=c(0,100) , main="2022-I")
abline(h=50 , lty=2)
par(mfrow=c(1,1))

##==: 3. el desbalance en accion: gana fico el puesto? :==##

## 3.1. la clase rara y la referencia que no modela nada
train_r1$gana_fico <- as.integer(train_r1$gana_primera=="gutierrez")
test_r1$gana_fico <- as.integer(test_r1$gana_primera=="gutierrez")
c(prevalencia_train = mean(train_r1$gana_fico) , siempre_no = mean(test_r1$gana_fico==0))

## 3.2. la logistica de la semana 4 (censo + departamento)
modelo_fico <- glm(gana_fico ~ . , data=train_r1[,c("gana_fico" , x_censo , "dpto")] , family=binomial)
test_r1$p_fico <- predict(modelo_fico , test_r1 , type="response")

## 3.3. las metricas de una regla de decision, calculadas a mano
metricas <- function(y , p , tau){
            pred <- as.integer(p >= tau)
            c(exactitud = mean(pred==y) ,
              sensibilidad = sum(pred==1 & y==1)/sum(y==1) ,
              precision = sum(pred==1 & y==1)/sum(pred==1))
}

## 3.4. tal cual con 0,5, y bajando el umbral a la prevalencia: la roc no cambia, el punto si
round(c(auc = auc(test_r1$gana_fico , test_r1$p_fico) , metricas(test_r1$gana_fico , test_r1$p_fico , 0.5)) , 3)
round(metricas(test_r1$gana_fico , test_r1$p_fico , mean(train_r1$gana_fico)) , 3)

## 3.5. pesos x5 a la clase rara en la verosimilitud (cada puesto raro cuenta por cinco)
w <- ifelse(train_r1$gana_fico==1 , 5 , 1)
modelo_pesos <- suppressWarnings(glm(gana_fico ~ . , data=train_r1[,c("gana_fico" , x_censo , "dpto")] ,
                                     family=binomial , weights=w))
p_pesos <- predict(modelo_pesos , test_r1 , type="response")

## 3.6. sobremuestrear la rara (entra cinco veces) y submuestrear la comun (semilla 2026)
idx_sobre <- c(which(train_r1$gana_fico==0) , rep(which(train_r1$gana_fico==1) , 5))
modelo_sobre <- glm(gana_fico ~ . , data=train_r1[idx_sobre , c("gana_fico" , x_censo , "dpto")] , family=binomial)
p_sobre <- predict(modelo_sobre , test_r1 , type="response")
set.seed(2026)
idx_sub <- c(sample(which(train_r1$gana_fico==0) , sum(train_r1$gana_fico==1)) , which(train_r1$gana_fico==1))
modelo_sub <- glm(gana_fico ~ . , data=train_r1[idx_sub , c("gana_fico" , x_censo , "dpto")] , family=binomial)
p_sub <- predict(modelo_sub , test_r1 , type="response")

## 3.7. la tabla del deck: la auc casi no se mueve; la p media se infla (descalibra)
tabla_des <- rbind(tal_cual_05 = c(auc(test_r1$gana_fico , test_r1$p_fico) , metricas(test_r1$gana_fico , test_r1$p_fico , 0.5) , mean(test_r1$p_fico)) ,
                   umbral_prev = c(auc(test_r1$gana_fico , test_r1$p_fico) , metricas(test_r1$gana_fico , test_r1$p_fico , mean(train_r1$gana_fico)) , mean(test_r1$p_fico)) ,
                   pesos_x5 = c(auc(test_r1$gana_fico , p_pesos) , metricas(test_r1$gana_fico , p_pesos , 0.5) , mean(p_pesos)) ,
                   sobremuestreo = c(auc(test_r1$gana_fico , p_sobre) , metricas(test_r1$gana_fico , p_sobre , 0.5) , mean(p_sobre)) ,
                   submuestreo = c(auc(test_r1$gana_fico , p_sub) , metricas(test_r1$gana_fico , p_sub , 0.5) , mean(p_sub)))
colnames(tabla_des) <- c("auc" , "exactitud" , "sensibilidad" , "precision" , "p_media")
round(tabla_des , 3)
mean(test_r1$gana_fico)

## 3.8. la roc con los dos umbrales marcados: el mismo modelo, dos puntos
taus <- seq(0 , 1 , 0.005)
roc <- t(sapply(taus , function(t){
       pred <- as.integer(test_r1$p_fico >= t)
       c(fpr = sum(pred==1 & test_r1$gana_fico==0)/sum(test_r1$gana_fico==0) ,
         tpr = sum(pred==1 & test_r1$gana_fico==1)/sum(test_r1$gana_fico==1))
}))
plot(roc[,"fpr"] , roc[,"tpr"] , type="l" , col="blue" , lwd=2 , xlab="tasa de falsos positivos" , ylab="sensibilidad")
abline(a=0 , b=1 , lty=2 , col="gray")
m05 <- metricas(test_r1$gana_fico , test_r1$p_fico , 0.5)
mpr <- metricas(test_r1$gana_fico , test_r1$p_fico , mean(train_r1$gana_fico))
points(sum(test_r1$p_fico>=0.5 & test_r1$gana_fico==0)/sum(test_r1$gana_fico==0) , m05["sensibilidad"] , pch=16 , col="red" , cex=1.5)
points(sum(test_r1$p_fico>=mean(train_r1$gana_fico) & test_r1$gana_fico==0)/sum(test_r1$gana_fico==0) , mpr["sensibilidad"] , pch=16 , col="orange" , cex=1.5)

## 3.9. la matriz de confusion con los dos umbrales: las cuentas del recap (vp, fn, fp, vn)
table(real = test_r1$gana_fico , predicho = as.integer(test_r1$p_fico >= 0.5))
table(real = test_r1$gana_fico , predicho = as.integer(test_r1$p_fico >= mean(train_r1$gana_fico)))

## la especificidad con cada umbral (1 - especificidad = tasa de falsos positivos, el eje de la roc)
c(especificidad_05 = mean(test_r1$p_fico[test_r1$gana_fico==0] < 0.5) ,
  especificidad_prev = mean(test_r1$p_fico[test_r1$gana_fico==0] < mean(train_r1$gana_fico)))

## 3.10. la p de la logistica por clase: lo que queda a la derecha de cada umbral, se senala
par(mfrow=c(2,1))
hist(test_r1$p_fico[test_r1$gana_fico==1] , breaks=seq(0,1,0.05) , col="darkred" , main="fico gana" , xlab="p de la logistica")
abline(v=c(mean(train_r1$gana_fico) , 0.5) , lty=2 , lwd=2 , col=c("orange","red"))
hist(test_r1$p_fico[test_r1$gana_fico==0] , breaks=seq(0,1,0.05) , col="gray" , main="fico no gana" , xlab="p de la logistica")
abline(v=c(mean(train_r1$gana_fico) , 0.5) , lty=2 , lwd=2 , col=c("orange","red"))
par(mfrow=c(1,1))

##==: 4. la red binaria: la perilla K con validacion cruzada :==##

## 4.1. la grilla del deck: K = 1, 4, 16 y decay 0,01 y 0,1, juzgados con los 10 pliegues (tarda unos 5 minutos)
grid_red <- expand.grid(K = c(1,4,16) , decay = c(0.01,0.1))
tareas <- expand.grid(fila = 1:nrow(grid_red) , pliegue = 1:10)
oof <- foreach(t = 1:nrow(tareas) , .packages="nnet") %dopar% {
       g <- grid_red[tareas$fila[t],]
       dentro <- pliegue!=tareas$pliegue[t]
       centro_j <- colMeans(X[dentro,])
       escala_j <- apply(X[dentro,] , 2 , sd)
       a <- cbind(sweep(sweep(X[dentro,] , 2 , centro_j) , 2 , escala_j , "/") , D[dentro,])
       b <- cbind(sweep(sweep(X[!dentro,] , 2 , centro_j) , 2 , escala_j , "/") , D[!dentro,])
       set.seed(2026)
       m <- nnet(a , train$gana_petro[dentro] , size=g$K , decay=g$decay , maxit=1000 ,
                 MaxNWts=10000 , entropy=T , trace=F)
       list(fila = tareas$fila[t] , idx = which(!dentro) , p = as.vector(predict(m , b)))
}
grid_red$auc_cv <- NA
for (i in 1:nrow(grid_red)){
     p_cv <- rep(NA , nrow(train))
     for (o in oof) if (o$fila==i) p_cv[o$idx] <- o$p
     grid_red$auc_cv[i] <- auc(train$gana_petro , p_cv)
}
grid_red %>% mutate(auc_cv = round(auc_cv , 3))

## 4.2. la curva del deck: la U de la red (con 16 unidades la auc cae)
plot(1:3 , grid_red$auc_cv[grid_red$decay==0.01] , type="b" , pch=16 , col="red" , ylim=c(0.90,0.95) ,
     xaxt="n" , xlab="K: unidades en la capa oculta" , ylab="auc de validacion cruzada")
axis(1 , at=1:3 , labels=c(1,4,16))
lines(1:3 , grid_red$auc_cv[grid_red$decay==0.1] , type="b" , pch=16 , col="blue")
abline(h=0.916 , lty=2 , col="gray")
abline(h=0.941 , lty=3 , col="gray")
legend("bottomleft" , legend=c("decay 0,01","decay 0,1","logistica (S4)","boosting (S5)") ,
       col=c("red","blue","gray","gray") , lty=c(1,1,2,3) , lwd=2 , cex=0.8)

## 4.3. la elegida (K = 4, decay 0,1), entrenada con todo el entrenamiento, en la prueba
mejor <- grid_red[which.max(grid_red$auc_cv),]
mejor
set.seed(2026)
red <- nnet(x_red , train$gana_petro , size=mejor$K , decay=mejor$decay , maxit=1000 ,
            MaxNWts=10000 , entropy=T , trace=F)
length(red$wts)
test$p_red <- as.vector(predict(red , x_red_t))
pred <- as.integer(test$p_red >= 0.5)
c(auc = auc(test$gana_petro , test$p_red) , exactitud = mean(pred==test$gana_petro) ,
  brier = mean((test$p_red - test$gana_petro)^2))
table(real = test$gana_petro , predicho = pred)

## 4.4. un puesto por dentro de la red: los pesos de nnet, unidad por unidad (el intercepto
## y uno por entrada) y al final los de la salida (el intercepto y uno por unidad)
W1 <- matrix(red$wts[1:(mejor$K*(ncol(x_red)+1))] , nrow=mejor$K , byrow=T)
b2 <- red$wts[mejor$K*(ncol(x_red)+1) + 1:(mejor$K+1)]
round(b2 , 2)
Z_t <- cbind(1 , x_red_t) %*% t(W1)
A_t <- 1/(1 + exp(-Z_t))
t_t <- as.vector(cbind(1 , A_t) %*% b2)

## la cuenta a mano da lo mismo que predict
max(abs(1/(1 + exp(-t_t)) - test$p_red))

## los dos puestos del deck: lo que fabrica la capa (A), la suma de la salida (t) y la probabilidad
i <- c(which(test$nombre=="COLEGIO PEDRO GRAU AROLA") , which(test$nombre=="SEC ESC JULIA AGUDELO" & test$municipio=="MEDELLIN"))
data.frame(nombre = test$nombre[i] , z1 = round(Z_t[i,1] , 2) , A = round(A_t[i,] , 2) , t = round(t_t[i] , 2) ,
           p = round(test$p_red[i] , 2) , voto_petro = round(test$voto_petro[i] , 1))

## 4.5. a que se parece cada unidad: la variable del censo mas correlacionada con su activacion
## (una lectura, no una definicion) y en que % de la prueba esta encendida (A > 0,5)
cors <- cor(A_t , X_t)
for (k in 1:mejor$K){
     j <- which.max(abs(cors[k,]))
     print(c(unidad = k , variable = colnames(X_t)[j] , correlacion = round(cors[k,j] , 2) , encendida = round(100*mean(A_t[,k] > 0.5))))
}

##==: 5. parar a tiempo: la red sin decay, detenida en cada punto :==##

## 5.1. la red de 16 unidades sin penalizacion, de 25 a 1.600 iteraciones (tarda unos 3 minutos)
iter <- c(25,50,100,200,400,800,1600)
tray <- foreach(it = iter , .packages="nnet" , .combine=rbind) %dopar% {
        set.seed(2026)
        m <- nnet(x_red , train$gana_petro , size=16 , decay=0 , maxit=it , MaxNWts=10000 , entropy=T , trace=F)
        c(iteraciones = it ,
          auc_train = auc(train$gana_petro , as.vector(predict(m , x_red))) ,
          auc_test = auc(test$gana_petro , as.vector(predict(m , x_red_t))))
}
round(tray , 3)

## 5.2. dentro de muestra solo sube; en la prueba toca techo en 50 y baja
plot(log10(tray[,"iteraciones"]) , tray[,"auc_test"] , type="b" , pch=16 , col="red" , ylim=c(0.89,1) ,
     xaxt="n" , xlab="iteraciones (escala logaritmica)" , ylab="auc")
axis(1 , at=log10(iter) , labels=iter)
lines(log10(tray[,"iteraciones"]) , tray[,"auc_train"] , type="b" , pch=16 , col="gray50")
abline(v=log10(iter[which.max(tray[,"auc_test"])]) , lty=2)

##==: 6. la semilla tambien es un hiperparametro :==##

## 6.1. la elegida con cinco semillas (en paralelo: cerca de un minuto)
semillas <- c(2026,1,2,3,4)
p_semillas <- foreach(s = semillas , .packages="nnet" , .combine=cbind) %dopar% {
              set.seed(s)
              m <- nnet(x_red , train$gana_petro , size=mejor$K , decay=mejor$decay , maxit=1000 ,
                        MaxNWts=10000 , entropy=T , trace=F)
              as.vector(predict(m , x_red_t))
}
auc_semillas <- rep(NA , 5)
for (i in 1:5){
     auc_semillas[i] <- auc(test$gana_petro , p_semillas[,i])
}
round(auc_semillas , 3)

## 6.2. el promedio de las cinco: mejor que cualquiera; alcanza al bosque de la semana 5
test$p_prom <- rowMeans(p_semillas)
c(auc_promedio = auc(test$gana_petro , test$p_prom) , brier = mean((test$p_prom - test$gana_petro)^2))

##==: 7. mas de dos categorias: la primera vuelta con softmax :==##

## 7.1. las entradas de la primera vuelta, estandarizadas con su entrenamiento
X1 <- as.matrix(train_r1[,x_censo])
X1_t <- as.matrix(test_r1[,x_censo])
S1 <- scale(X1)
S1_t <- scale(X1_t , attr(S1 , "scaled:center") , attr(S1 , "scaled:scale"))
S1 <- matrix(S1 , nrow(S1))
S1_t <- matrix(S1_t , nrow(S1_t))
D1 <- sapply(dptos , function(d) as.numeric(train_r1$departamento==d))
D1_t <- sapply(dptos , function(d) as.numeric(test_r1$departamento==d))

## 7.2. una salida por clase y el softmax: cuatro probabilidades que suman uno
set.seed(2026)
red_mc <- nnet(cbind(S1 , D1) , class.ind(train_r1$gana_primera) , size=mejor$K , decay=mejor$decay ,
               maxit=1000 , MaxNWts=10000 , softmax=T , trace=F)
length(red_mc$wts)
p_mc <- predict(red_mc , cbind(S1_t , D1_t))
head(rowSums(p_mc))

## 7.3. la matriz 4x4, el recall por clase y la exactitud contra decir siempre petro
clase <- factor(colnames(p_mc)[max.col(p_mc)] , levels=levels(train_r1$gana_primera))
matriz <- table(real = test_r1$gana_primera , predicho = clase)
matriz
round(100*diag(matriz)/rowSums(matriz) , 1)
c(exactitud_red = mean(clase==test_r1$gana_primera) , siempre_petro = mean(test_r1$gana_primera=="petro"))

## 7.4. el puesto mas renido de la prueba segun la red: las cuatro probabilidades
entropia <- -rowSums(p_mc*log(pmax(p_mc , 1e-12)))
i <- which.max(entropia)
test_r1[i , c("nombre","municipio","gana_primera")]
round(100*p_mc[i,] , 1)

##==: 8. la carrera de clasificacion :==##

## 8.1. la logistica de la semana 4 para gana_petro (el arbol 0,896, el bosque 0,943 y el boosting 0,941 son de la semana 5)
modelo_log <- glm(gana_petro ~ . , data=train[,c("gana_petro" , x_censo , "dpto")] , family=binomial)
p_log <- predict(modelo_log , test , type="response")

## 8.2. tabla final: la auc de prueba
tabla <- data.frame(modelo = c("decir siempre gana petro" , "regresion logistica (semana 4)" ,
                               "red neuronal (K = 4)" , "promedio de cinco redes") ,
                    auc_test = c(0.5 , auc(test$gana_petro , p_log) ,
                                 auc(test$gana_petro , test$p_red) , auc(test$gana_petro , test$p_prom)))
tabla %>% mutate(auc_test = round(auc_test , 3))

## cerrar el cluster
stopCluster(cl)
