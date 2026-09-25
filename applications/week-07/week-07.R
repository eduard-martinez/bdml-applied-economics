## Big Data y Machine Learning para Economia Aplicada
## week-07: representar (comprimir con sentido)
## Last run: Sep 25, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , splines , cluster , nnet , xgboost , doParallel)

## cluster: las redes se ajustan en paralelo, cada tarea con su semilla (mismo resultado, menos tiempo)
cores <- parallel::detectCores()
cl <- makeCluster(max(1 , cores - 2))
registerDoParallel(cl)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-07/input/"
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

## 1.4. la ficha de 37: hoy no la miramos columna por columna, la resumimos
x_ficha <- c(setdiff(names(ficha) , "puesto") , "log_registrados" , "rural")
x_dpto <- c(x_ficha , "dpto")
length(x_ficha)

## 1.5. el pais: la particion fija del curso y los 10 pliegues de la semana 3 (semilla 2026)
train_p <- db %>% subset(muestra=="entrenamiento")
test_p <- db %>% subset(muestra=="prueba")
set.seed(2026)
pliegue_p <- sample(rep(1:10 , length.out=nrow(train_p)))
c(train = nrow(train_p) , test = nrow(test_p))

## 1.6. la ficha como matriz; el departamento en 33 columnas de ceros y unos (para la red)
X <- as.matrix(train_p[,x_ficha])
X_t <- as.matrix(test_p[,x_ficha])
dptos <- sort(unique(db$departamento))
D <- sapply(dptos , function(d) as.numeric(train_p$departamento==d))
D_t <- sapply(dptos , function(d) as.numeric(test_p$departamento==d))

## 1.7. la referencia de siempre: la media del entrenamiento
media_train <- mean(train_p$voto_petro)
sqrt(mean((test_p$voto_petro - media_train)^2))

##==: 2. sin y, en linea recta: las componentes principales :==##

## 2.1. dos columnas de la ficha, sin estandarizar: la direccion de maxima varianza
pc_2 <- prcomp(train_p[,c("educ_superior","serv_internet")] , center=T , scale.=F)
if (pc_2$rotation[1,1] < 0) pc_2$rotation[,1] <- -pc_2$rotation[,1]
if (pc_2$rotation[1,2] < 0) pc_2$rotation[,2] <- -pc_2$rotation[,2]
c(cor = cor(train_p$educ_superior , train_p$serv_internet) , pve_pc1 = pc_2$sdev[1]^2/sum(pc_2$sdev^2))

## las flechas de las dos componentes sobre la nube (800 puestos al azar, semilla 2026): el internet varia mas
set.seed(2026)
muestra_1 <- sample(nrow(train_p) , 800)
plot(train_p$educ_superior[muestra_1] , train_p$serv_internet[muestra_1] , pch=16 , cex=0.5 , col="gray50" ,
     xlim=c(0,80) , ylim=c(0,100) , xlab="educacion superior (%)" , ylab="internet (% de las viviendas)")
arrows(pc_2$center[1] , pc_2$center[2] , pc_2$center[1] + 1.5*pc_2$sdev[1]*pc_2$rotation[1,1] ,
       pc_2$center[2] + 1.5*pc_2$sdev[1]*pc_2$rotation[2,1] , col="blue" , lwd=3)
arrows(pc_2$center[1] , pc_2$center[2] , pc_2$center[1] + 2*pc_2$sdev[2]*pc_2$rotation[1,2] ,
       pc_2$center[2] + 2*pc_2$sdev[2]*pc_2$rotation[2,2] , col="red" , lwd=3)

## 2.2. el pca de las 37 columnas: prcomp estandariza con las medias y desviaciones del entrenamiento,
## y predict aplica esas mismas a la prueba (tambien sin y hay pipeline)
pc <- prcomp(X , center=T , scale.=T)
pve <- pc$sdev^2/sum(pc$sdev^2)
round(100*pve[1:10] , 1)
c(para_80 = which(cumsum(pve) >= 0.8)[1] , para_90 = which(cumsum(pve) >= 0.9)[1])

## la varianza de cada componente y la acumulada (el scree)
barplot(100*pve , names.arg=1:37 , col="blue" , ylim=c(0,100) , xlab="componente" , ylab="% de la varianza de la ficha")
lines(seq(0.7 , by=1.2 , length.out=37) , 100*cumsum(pve) , col="red" , lwd=2)

## 2.3. la ultima componente no tiene varianza: dos columnas son la misma informacion
tail(pve , 1)
round(pc$rotation[abs(pc$rotation[,37]) > 0.5 , 37] , 3)
range(train_p$edad_votar + train_p$edad_0_19)

## 2.4. el signo de una componente es arbitrario: la PC1 se orienta para que crezca con la ruralidad
## y la PC2 para que crezca con los ninos
if (pc$rotation["rural",1] < 0){
    pc$rotation[,1] <- -pc$rotation[,1]
    pc$x[,1] <- -pc$x[,1]
}
if (pc$rotation["edad_0_19",2] < 0){
    pc$rotation[,2] <- -pc$rotation[,2]
    pc$x[,2] <- -pc$x[,2]
}
z <- pc$x
z_t <- predict(pc , X_t)

## las cargas: la PC1 es un indice de carencias (servicios de un lado, ruralidad del otro); la PC2, la edad
round(head(pc$rotation[order(-abs(pc$rotation[,1])),1] , 10) , 2)
round(head(pc$rotation[order(-abs(pc$rotation[,2])),2] , 10) , 2)
par(mfrow=c(1,2) , mar=c(4,9,2,1))
cargas_1 <- head(pc$rotation[order(-abs(pc$rotation[,1])),1] , 10)
cargas_2 <- head(pc$rotation[order(-abs(pc$rotation[,2])),2] , 10)
barplot(rev(cargas_1) , horiz=T , las=1 , cex.names=0.7 , col=ifelse(rev(cargas_1) > 0 , "red" , "blue") , main="PC1")
barplot(rev(cargas_2) , horiz=T , las=1 , cex.names=0.7 , col=ifelse(rev(cargas_2) > 0 , "red" , "blue") , main="PC2")
par(mfrow=c(1,1) , mar=c(5,4,4,2))

## con que se correlacionan los puntajes
correlaciones <- data.frame(variable = c("rural","educ_superior","afro","indigena","edad_0_19","edad_60_74","voto_petro") ,
                            pc1 = NA , pc2 = NA , pc3 = NA)
for (i in 1:nrow(correlaciones)){
     for (k in 1:3){
          correlaciones[i,k+1] <- cor(z[,k] , train_p[[correlaciones$variable[i]]])
     }
}
correlaciones %>% mutate(across(-variable , ~ round(.x , 2)))

## 2.5. el indice sin y, auditado con y: la PC1 explica el 0,9 % del voto, y no en linea recta
r2_pc <- rep(NA , 37)
for (k in 1:37){
     r2_pc[k] <- cor(z[,k] , train_p$voto_petro)^2
}
round(r2_pc[1:3] , 4)
decil <- cut(z[,1] , quantile(z[,1] , 0:10/10) , include.lowest=T , labels=F)
por_decil <- train_p %>% mutate(decil = decil) %>% group_by(decil) %>%
             summarise(voto = mean(voto_petro) , rural = mean(rural) , educ = mean(educ_superior) , afro = mean(afro) , .groups="drop")
por_decil %>% mutate(across(-decil , ~ round(.x , 1)))
plot(por_decil$decil , por_decil$voto , type="b" , pch=16 , col="blue" , ylim=c(40,75) ,
     xlab="decil de la PC1 (1: urbano, con servicios; 10: rural, sin servicios)" , ylab="voto por Petro (%)")
abline(h=media_train , lty=2)

## 2.6. la PC1 como predictor: en linea recta y doblada con un spline natural de 5 grados
d1 <- data.frame(y = train_p$voto_petro , z1 = z[,1])
d1_t <- data.frame(z1 = z_t[,1])
c(pc1_recta = sqrt(mean((test_p$voto_petro - predict(lm(y ~ z1 , data=d1) , d1_t))^2)) ,
  pc1_doblada = sqrt(mean((test_p$voto_petro - predict(lm(y ~ ns(z1 , df=5) , data=d1) , d1_t))^2)))

##==: 3. comprimir para predecir sin mirar y: la regresion sobre componentes :==##

## 3.1. el juez de la semana 3 para M = 1, ..., 37: el pca se estima dentro de cada pliegue (cerca de medio minuto)
pcr <- data.frame(M = 1:37 , rmse_cv = NA , rmse_test = NA)
error_10 <- rep(NA , nrow(train_p))
for (M in 1:37){
     error_m <- rep(NA , nrow(train_p))
     for (j in 1:10){
          dentro <- pliegue_p!=j
          pc_j <- prcomp(X[dentro,] , center=T , scale.=T)
          A <- data.frame(y = train_p$voto_petro[dentro] , pc_j$x[,1:M,drop=F])
          B <- data.frame(predict(pc_j , X[!dentro,])[,1:M,drop=F])
          error_m[!dentro] <- train_p$voto_petro[!dentro] - predict(lm(y ~ . , data=A) , B)
     }
     if (M==10) error_10 <- error_m
     pcr$rmse_cv[M] <- sqrt(mean(error_m^2))
     A <- data.frame(y = train_p$voto_petro , z[,1:M,drop=F])
     B <- data.frame(z_t[,1:M,drop=F])
     pcr$rmse_test[M] <- sqrt(mean((test_p$voto_petro - predict(lm(y ~ . , data=A) , B))^2))
}

## con una componente, casi la media; con todas, OLS: comprimir sin y no le gana a usar todo
pcr %>% subset(M %in% c(1,2,5,10,20,36,37)) %>% mutate(across(-M , ~ round(.x , 2)))
pcr[which.min(pcr$rmse_cv),]
plot(pcr$M , pcr$rmse_cv , type="l" , col="blue" , lwd=2 , ylim=c(18,28) , xlab="M: componentes en la regresion" , ylab="rmse")
lines(pcr$M , pcr$rmse_test , col="red" , lwd=2 , lty=2)
abline(h=sqrt(mean((test_p$voto_petro - media_train)^2)) , lty=3)

## 3.2. el salto del juez con 9 a 11 componentes tiene nombre: una componente es la columna palenquero sola
peor <- which.max(abs(error_10))
data.frame(nombre = train_p$nombre[peor] , municipio = train_p$municipio[peor] ,
           voto_petro = train_p$voto_petro[peor] , voto_pred = train_p$voto_petro[peor] - error_10[peor])
componente_rara <- c(raizal = unname(which.max(abs(pc$rotation["raizal",]))) ,
                     palenquero = unname(which.max(abs(pc$rotation["palenquero",]))) ,
                     rom = unname(which.max(abs(pc$rotation["rom",]))))
componente_rara
round(pc$rotation["palenquero" , componente_rara["palenquero"]] , 3)
c(raizal = mean(train_p$raizal==0) , palenquero = mean(train_p$palenquero==0) , rom = mean(train_p$rom==0))

##==: 4. sin y, en grupos: k-medias :==##

## 4.1. la ficha estandarizada con el entrenamiento (sin los atributos que deja scale)
S <- scale(X)
centro_s <- attr(S , "scaled:center")
escala_s <- attr(S , "scaled:scale")
S <- matrix(S , nrow(S) , dimnames=list(NULL , colnames(X)))
S_t <- scale(X_t , centro_s , escala_s)
S_t <- matrix(S_t , nrow(S_t) , dimnames=list(NULL , colnames(X)))

## 4.2. el codo: la variacion dentro de los grupos sobre la total, K = 1, ..., 10 (25 arranques, semilla 2026)
grid_k <- data.frame(K = 1:10 , dentro_total = NA)
set.seed(2026)
for (i in 1:nrow(grid_k)){
     km <- kmeans(S , grid_k$K[i] , nstart=25 , iter.max=50)
     grid_k$dentro_total[i] <- km$tot.withinss/km$totss
}
grid_k %>% mutate(dentro_total = round(dentro_total , 3))

## 4.3. la silueta en 2.000 puestos al azar (semilla 2026): la distancia al propio grupo contra la del vecino
set.seed(2026)
muestra_s <- sample(nrow(S) , 2000)
distancias <- dist(S[muestra_s,])
grid_s <- data.frame(K = 2:8 , silueta = NA)
for (i in 1:nrow(grid_s)){
     set.seed(2026)
     km <- kmeans(S , grid_s$K[i] , nstart=25 , iter.max=50)
     grid_s$silueta[i] <- mean(silhouette(km$cluster[muestra_s] , distancias)[,3])
}
grid_s %>% mutate(silueta = round(silueta , 3))
par(mfrow=c(1,2))
plot(grid_k$K , grid_k$dentro_total , type="b" , pch=16 , col="blue" , xlab="K" , ylab="variacion dentro / total")
plot(grid_s$K , grid_s$silueta , type="b" , pch=16 , col="red" , xlab="K" , ylab="silueta promedio")
par(mfrow=c(1,1))

## 4.4. cuatro grupos, y su nombre por su perfil (el voto, solo para auditar)
set.seed(2026)
km4 <- kmeans(S , 4 , nstart=25 , iter.max=50)
perfil <- train_p %>% mutate(grupo = km4$cluster) %>% group_by(grupo) %>%
          summarise(n = n() , rural = 100*mean(rural) , educ = mean(educ_superior) , afro = mean(afro) , indigena = mean(indigena) ,
                    alcantarillado = mean(serv_alcantarillado) , voto = mean(voto_petro) , .groups="drop")
perfil$nombre <- NA
rurales <- perfil$grupo[perfil$rural > 50]
urbanos <- perfil$grupo[perfil$rural <= 50]
etnica <- rurales[which.max((perfil$afro + perfil$indigena)[match(rurales , perfil$grupo)])]
dotada <- urbanos[which.max(perfil$educ[match(urbanos , perfil$grupo)])]
perfil$nombre[perfil$grupo==etnica] <- "rural etnica"
perfil$nombre[perfil$grupo==setdiff(rurales , etnica)] <- "rural andina"
perfil$nombre[perfil$grupo==dotada] <- "urbana dotada"
perfil$nombre[perfil$grupo==setdiff(urbanos , dotada)] <- "urbana popular"
perfil <- perfil[match(c("rural etnica","rural andina","urbana dotada","urbana popular") , perfil$nombre),]
perfil %>% mutate(across(c(rural , educ , afro , indigena , alcantarillado , voto) , ~ round(.x , 1)))

## el mapa: k-medias no vio las coordenadas y los grupos dibujan el pais
colores <- c("orange","darkgreen","blue","red")
plot(train_p$lon , train_p$lat , pch=16 , cex=0.3 , col=colores[match(km4$cluster , perfil$grupo)] ,
     xlab="longitud" , ylab="latitud" , xlim=c(-79.5,-66.5) , ylim=c(-4.5,13))
legend("bottomright" , legend=perfil$nombre , col=colores , pch=16 , cex=0.8)

## 4.5. predicen los grupos? cada puesto de prueba recibe la media del grupo de su centroide mas cercano
test_p$grupo <- NA
for (i in 1:nrow(test_p)){
     test_p$grupo[i] <- which.min(colSums((t(km4$centers) - S_t[i,])^2))
}
media_grupo <- tapply(train_p$voto_petro , km4$cluster , mean)
media_dpto <- tapply(train_p$voto_petro , train_p$departamento , mean)
test_p$voto_pred <- media_grupo[test_p$grupo]
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

## el grupo solo, el departamento solo y el grupo dentro de cada departamento: describir no es predecir
media_gd <- tapply(train_p$voto_petro , paste(km4$cluster , train_p$departamento) , mean)
pred_gd <- media_gd[paste(test_p$grupo , test_p$departamento)]
pred_gd[is.na(pred_gd)] <- media_dpto[test_p$departamento][is.na(pred_gd)]
c(grupo = sqrt(mean((test_p$voto_petro - test_p$voto_pred)^2)) ,
  dpto = sqrt(mean((test_p$voto_petro - media_dpto[test_p$departamento])^2)) ,
  grupo_dpto = sqrt(mean((test_p$voto_petro - pred_gd)^2)))

## 4.6. la estabilidad: con otras cinco semillas, los mismos grupos?
acuerdo <- rep(NA , 5)
for (s in 1:5){
     set.seed(s)
     km_s <- kmeans(S , 4 , nstart=25 , iter.max=50)
     acuerdo[s] <- sum(apply(table(km4$cluster , km_s$cluster) , 1 , max))/nrow(S)
}
acuerdo

## 4.7. con K = 5 la ruralidad etnica se parte: aparece un nucleo que vota mas por Petro
set.seed(2026)
km5 <- kmeans(S , 5 , nstart=25 , iter.max=50)
perfil5 <- train_p %>% mutate(grupo = km5$cluster) %>% group_by(grupo) %>%
           summarise(n = n() , rural = 100*mean(rural) , afro = mean(afro) , indigena = mean(indigena) , voto = mean(voto_petro) , .groups="drop") %>%
           arrange(desc(voto))
perfil5 %>% mutate(across(-c(grupo , n) , ~ round(.x , 1)))

##==: 5. con y: una red neuronal de una sola unidad :==##

## 5.1. las entradas de la red: la ficha estandarizada con el entrenamiento, el departamento en ceros y unos,
## y las coordenadas estandarizadas igual
coord <- as.matrix(train_p[,c("lon","lat")])
coord_t <- as.matrix(test_p[,c("lon","lat")])
C <- sweep(sweep(coord , 2 , colMeans(coord)) , 2 , apply(coord , 2 , sd) , "/")
C_t <- sweep(sweep(coord_t , 2 , colMeans(coord)) , 2 , apply(coord , 2 , sd) , "/")
entradas <- list(ficha = S , dpto = cbind(S , D) , todo = cbind(S , D , C))
entradas_t <- list(ficha = S_t , dpto = cbind(S_t , D_t) , todo = cbind(S_t , D_t , C_t))

## 5.2. una unidad sobre la ficha (nnet: la unidad y la salida son sigmoides; el voto va en [0,1]; decay es Ridge)
set.seed(2026)
red_1 <- nnet(entradas$ficha , train_p$voto_petro/100 , size=1 , decay=0.01 , maxit=1000 , MaxNWts=10000 , trace=F)

## target predicho vs target original: la red de una unidad le gana a la PC1 recta y doblada
test_p$voto_pred <- 100*as.vector(predict(red_1 , entradas_t$ficha))
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
sqrt(mean((test_p$voto_petro - test_p$voto_pred)^2))

## 5.3. la direccion que eligio la y: los pesos de la unidad (el primero es la constante)
pesos <- red_1$wts
w <- setNames(pesos[2:38] , colnames(entradas$ficha))
round(head(w[order(-abs(w))] , 8) , 2)

## sus dos pesos mas grandes son casi opuestos: log poblacion menos log viviendas = las personas por vivienda
h <- pesos[1] + entradas$ficha %*% w
h_t <- pesos[1] + entradas_t$ficha %*% w
c(cor_h_pc1 = cor(h , z[,1]) , cor_ols_pc1 = cor(fitted(lm(train_p$voto_petro ~ S)) , z[,1]))

## el voto contra la pc1 y contra la activacion de la unidad (800 puestos de prueba al azar, semilla 2026)
set.seed(2026)
muestra_8 <- sample(nrow(test_p) , 800)
par(mfrow=c(1,2))
plot(z_t[muestra_8,1] , test_p$voto_petro[muestra_8] , pch=16 , cex=0.4 , col="gray50" , xlab="puntaje de la PC1" , ylab="voto por Petro (%)" , main="sin y: la PC1")
abline(lm(y ~ z1 , data=d1) , col="blue" , lwd=2)
plot(plogis(h_t[muestra_8]) , test_p$voto_petro[muestra_8] , pch=16 , cex=0.4 , col="gray50" , xlab="activacion de la unidad" , ylab="voto por Petro (%)" , main="con y: una unidad")
curve(100*plogis(pesos[39] + pesos[40]*x) , from=0 , to=1 , add=T , col="red" , lwd=2)
par(mfrow=c(1,1))

##==: 6. mas unidades, mas capacidad: el juez de las redes :==##

## 6.1. la grilla: con la ficha y el departamento, K = 1, ..., 16 y tres penalizaciones; con la ficha sola y con
## las coordenadas, una grilla mas corta. las entradas se estandarizan dentro de cada pliegue
grid_red <- rbind(expand.grid(ingredientes="dpto" , K=c(1,2,4,8,16) , decay=c(0.01,0.1,1) , stringsAsFactors=F) ,
                  expand.grid(ingredientes=c("ficha","todo") , K=c(4,8,16) , decay=c(0.01,0.1) , stringsAsFactors=F))
grid_red$rmse_cv <- NA
tareas <- expand.grid(fila = 1:nrow(grid_red) , pliegue = 1:10)
nrow(tareas)

## 6.2. las 270 redes (27 configuraciones x 10 pliegues) en paralelo: unos 10 minutos con 6 nucleos
sse <- foreach(t = 1:nrow(tareas) , .packages="nnet" , .combine=c) %dopar% {
       g <- grid_red[tareas$fila[t],]
       dentro <- pliegue_p!=tareas$pliegue[t]
       centro_j <- colMeans(X[dentro,])
       escala_j <- apply(X[dentro,] , 2 , sd)
       a <- sweep(sweep(X[dentro,] , 2 , centro_j) , 2 , escala_j , "/")
       b <- sweep(sweep(X[!dentro,] , 2 , centro_j) , 2 , escala_j , "/")
       if (g$ingredientes!="ficha"){
           a <- cbind(a , D[dentro,])
           b <- cbind(b , D[!dentro,])
       }
       if (g$ingredientes=="todo"){
           centro_cj <- colMeans(coord[dentro,])
           escala_cj <- apply(coord[dentro,] , 2 , sd)
           a <- cbind(a , sweep(sweep(coord[dentro,] , 2 , centro_cj) , 2 , escala_cj , "/"))
           b <- cbind(b , sweep(sweep(coord[!dentro,] , 2 , centro_cj) , 2 , escala_cj , "/"))
       }
       set.seed(2026)
       m <- nnet(a , train_p$voto_petro[dentro]/100 , size=g$K , decay=g$decay , maxit=1000 , MaxNWts=10000 , trace=F)
       sum((train_p$voto_petro[!dentro] - 100*predict(m , b))^2)
}
for (i in 1:nrow(grid_red)){
     grid_red$rmse_cv[i] <- sqrt(sum(sse[tareas$fila==i])/nrow(train_p))
}

## con 16 unidades y poca penalizacion la red sobreajusta; el weight decay es el lambda de Ridge
grid_red %>% subset(ingredientes=="dpto") %>% mutate(rmse_cv = round(rmse_cv , 2))
plot(1:5 , grid_red$rmse_cv[grid_red$ingredientes=="dpto" & grid_red$decay==0.01] , type="b" , pch=16 , col="red" , ylim=c(13,16) ,
     xaxt="n" , xlab="K: unidades en la capa oculta" , ylab="rmse de cv")
axis(1 , at=1:5 , labels=c(1,2,4,8,16))
lines(1:5 , grid_red$rmse_cv[grid_red$ingredientes=="dpto" & grid_red$decay==0.1] , type="b" , pch=16 , col="blue")
lines(1:5 , grid_red$rmse_cv[grid_red$ingredientes=="dpto" & grid_red$decay==1] , type="b" , pch=16 , col="orange")
legend("topright" , legend=c("decay 0,01","decay 0,1","decay 1") , col=c("red","blue","orange") , lwd=2)

## 6.3. la red que elige el juez con cada juego de ingredientes
elegida <- grid_red %>% group_by(ingredientes) %>% arrange(rmse_cv) %>% mutate(ranking = row_number()) %>% subset(ranking==1)
elegida

##==: 7. las tres redes elegidas, entrenadas con todo el entrenamiento, en la prueba :==##

## 7.1. un ajuste por juego de ingredientes (en paralelo: cerca de un minuto)
finales <- foreach(i = 1:nrow(elegida) , .packages="nnet") %dopar% {
           set.seed(2026)
           m <- nnet(entradas[[elegida$ingredientes[i]]] , train_p$voto_petro/100 , size=elegida$K[i] , decay=elegida$decay[i] ,
                     maxit=1000 , MaxNWts=10000 , trace=F)
           list(test = 100*as.vector(predict(m , entradas_t[[elegida$ingredientes[i]]])) , pesos = length(m$wts))
}
elegida$rmse_test <- NA
elegida$pesos <- NA
for (i in 1:nrow(elegida)){
     elegida$rmse_test[i] <- sqrt(mean((test_p$voto_petro - finales[[i]]$test)^2))
     elegida$pesos[i] <- finales[[i]]$pesos
}
elegida

## target predicho vs target original: la red con la ficha y el departamento
test_p$voto_pred <- finales[[which(elegida$ingredientes=="dpto")]]$test
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

##==: 8. parar a tiempo: la trayectoria de una red sin penalizacion :==##

## 8.1. la misma red de 16 unidades, detenida despues de 10, 25, ..., 1.600 iteraciones (en paralelo: unos 3 minutos)
trayectoria <- data.frame(iteraciones = c(10,25,50,100,200,400,800,1600) , rmse_train = NA , rmse_test = NA)
curvas <- foreach(it = trayectoria$iteraciones , .packages="nnet") %dopar% {
          set.seed(2026)
          m <- nnet(entradas$dpto , train_p$voto_petro/100 , size=16 , decay=0 , maxit=it , MaxNWts=10000 , trace=F)
          c(train = sqrt(mean((train_p$voto_petro - 100*predict(m , entradas$dpto))^2)) ,
            test = sqrt(mean((test_p$voto_petro - 100*predict(m , entradas_t$dpto))^2)))
}
for (i in 1:nrow(trayectoria)){
     trayectoria$rmse_train[i] <- curvas[[i]]["train"]
     trayectoria$rmse_test[i] <- curvas[[i]]["test"]
}

## dentro de muestra el error solo baja; en la prueba toca fondo y vuelve a subir
trayectoria %>% mutate(across(-iteraciones , ~ round(.x , 2)))
plot(log10(trayectoria$iteraciones) , trayectoria$rmse_test , type="b" , pch=16 , col="red" , ylim=c(9,19) ,
     xaxt="n" , xlab="iteraciones (escala logaritmica)" , ylab="rmse")
axis(1 , at=log10(trayectoria$iteraciones) , labels=trayectoria$iteraciones)
lines(log10(trayectoria$iteraciones) , trayectoria$rmse_train , type="b" , pch=16 , col="gray50")

##==: 9. la semilla tambien es un hiperparametro :==##

## 9.1. la red elegida con la ficha y el departamento, con cinco semillas (en paralelo: cerca de un minuto)
semillas <- c(2026 , 1 , 2 , 3 , 4)
por_semilla <- foreach(s = semillas , .packages="nnet") %dopar% {
               set.seed(s)
               m <- nnet(entradas$dpto , train_p$voto_petro/100 , size=elegida$K[elegida$ingredientes=="dpto"] ,
                         decay=elegida$decay[elegida$ingredientes=="dpto"] , maxit=1000 , MaxNWts=10000 , trace=F)
               100*as.vector(predict(m , entradas_t$dpto))
}
pred_semillas <- do.call(cbind , por_semilla)

## casi el mismo error, pero no la misma funcion: promediar las cinco baja el error
rmse_semillas <- rep(NA , 5)
for (i in 1:5){
     rmse_semillas[i] <- sqrt(mean((test_p$voto_petro - pred_semillas[,i])^2))
}
round(rmse_semillas , 3)
test_p$voto_pred <- rowMeans(pred_semillas)
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
sqrt(mean((test_p$voto_petro - test_p$voto_pred)^2))

##==: 10. la carrera: suma la red al promedio de la semana 5? :==##

## 10.1. el boosting de la semana 5 (ficha + departamento, d = 6, tasa 0,05) con las 738 rondas que eligio su juez
x_dpto_m <- model.matrix(~ . - 1 , data=train_p[,x_dpto])
x_dpto_t <- model.matrix(~ . - 1 , data=test_p[,x_dpto])
parametros <- xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 , subsample=0.8 , colsample_bytree=0.8 ,
                         nthread=1 , eval_metric="rmse")
set.seed(2026)
boosting <- xgb.train(params=parametros , data=xgb.DMatrix(x_dpto_m , label=train_p$voto_petro) , nrounds=738 , verbose=0)
pred_boosting <- predict(boosting , xgb.DMatrix(x_dpto_t))

## 10.2. el k-nn del mapa de la semana 5 (k = 7): el promedio del voto de los siete puestos mas cercanos
pred_knn <- rep(NA , nrow(test_p))
for (i in 1:nrow(test_p)){
     distancia <- (coord[,1] - coord_t[i,1])^2 + (coord[,2] - coord_t[i,2])^2
     pred_knn[i] <- mean(train_p$voto_petro[order(distancia)[1:7]])
}

## 10.3. el promedio de la semana 5, y con las cinco redes: la red no agrega porque se equivoca como el boosting
pred_redes <- rowMeans(pred_semillas)
c(boosting = sqrt(mean((test_p$voto_petro - pred_boosting)^2)) ,
  cinco_redes = sqrt(mean((test_p$voto_petro - pred_redes)^2)) ,
  boosting_knn = sqrt(mean((test_p$voto_petro - (pred_boosting + pred_knn)/2)^2)) ,
  boosting_knn_redes = sqrt(mean((test_p$voto_petro - (pred_boosting + pred_knn + pred_redes)/3)^2)))
cor(test_p$voto_petro - pred_redes , test_p$voto_petro - pred_boosting)

##==: 11. cuando no hay columnas: el voto como una imagen :==##

## 11.1. una imagen es una matriz de numeros: celdas de un cuarto de grado con el voto promedio de sus puestos
## (el territorio continental: San Andres y Providencia quedan fuera del recuadro)
continental <- db %>% subset(lon > -80 & lon < -66 & lat > -5 & lat < 13.5)
continental <- continental %>% mutate(fila = floor((lat + 4.5)/0.25) + 1 , columna = floor((lon + 79.25)/0.25) + 1)
por_celda <- continental %>% group_by(fila , columna) %>% summarise(voto = mean(voto_petro) , n = n() , .groups="drop")
imagen <- matrix(NA , max(continental$fila) , max(continental$columna))
for (i in 1:nrow(por_celda)){
     imagen[por_celda$fila[i] , por_celda$columna[i]] <- por_celda$voto[i]
}
c(filas = nrow(imagen) , columnas = ncol(imagen) , celdas_con_puestos = nrow(por_celda))

## 11.2. el filtro de bordes (Sobel): una ventana de 3 x 3 que recorre la imagen, horizontal y vertical
sobel <- matrix(c(-1,0,1,-2,0,2,-1,0,1) , 3 , byrow=T)
sobel
borde <- matrix(NA , nrow(imagen) , ncol(imagen))
for (r in 2:(nrow(imagen) - 1)){
     for (s in 2:(ncol(imagen) - 1)){
          ventana <- imagen[(r-1):(r+1),(s-1):(s+1)]
          if (!any(is.na(ventana))) borde[r,s] <- sqrt(sum(ventana*sobel)^2 + sum(ventana*t(sobel))^2)
     }
}

## el voto (mas oscuro, mas Petro) y sus bordes: el filtro no sabe que es un departamento, encuentra fronteras
par(mfrow=c(1,2))
image(t(imagen) , col=gray(seq(1 , 0 , length.out=100)) , axes=F , main="el voto")
image(t(borde) , col=gray(seq(1 , 0 , length.out=100)) , axes=F , main="los bordes (Sobel)")
par(mfrow=c(1,1))

##==: 12. tabla final :==##

## el voto: la carrera de la semana, con las referencias de las semanas 3 y 5 (misma prueba)
carrera <- data.frame(modelo = c("media del train" , "pca: la PC1" , "red, una unidad" , "ols, la ficha (pcr con todas)" ,
                                 "red, la ficha" , "red, ficha + departamento" , "cinco redes promediadas" ,
                                 "red, ficha + departamento + coordenadas" , "boosting, ficha + departamento (semana 5)" ,
                                 "promedio boosting + k-nn (semana 5)" , "promedio boosting + k-nn + cinco redes") ,
                      rmse_test = c(sqrt(mean((test_p$voto_petro - media_train)^2)) , pcr$rmse_test[1] ,
                                    sqrt(mean((test_p$voto_petro - 100*predict(red_1 , entradas_t$ficha))^2)) , pcr$rmse_test[37] ,
                                    elegida$rmse_test[elegida$ingredientes=="ficha"] , elegida$rmse_test[elegida$ingredientes=="dpto"] ,
                                    sqrt(mean((test_p$voto_petro - pred_redes)^2)) , elegida$rmse_test[elegida$ingredientes=="todo"] ,
                                    sqrt(mean((test_p$voto_petro - pred_boosting)^2)) ,
                                    sqrt(mean((test_p$voto_petro - (pred_boosting + pred_knn)/2)^2)) ,
                                    sqrt(mean((test_p$voto_petro - (pred_boosting + pred_knn + pred_redes)/3)^2))))
carrera %>% mutate(rmse_test = round(rmse_test , 1)) %>% arrange(rmse_test)

## las cinco redes promediadas: target predicho vs target original, y la diagonal
plot(test_p$voto_petro , pred_redes , pch=16 , col="blue" , cex=0.5 , xlim=c(0,100) , ylim=c(0,100) ,
     xlab="target original (%)" , ylab="target predicho (%)")
abline(a=0 , b=1 , lty=2)

## stop cluster
stopCluster(cl)
