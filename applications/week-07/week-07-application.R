## Big Data y Machine Learning para Economia Aplicada
## week-07-application: una red neuronal a pasitos (pintar una chiquita, combinar dos y ponerla a competir)

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , nnet , NeuralNetTools , randomForest)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-07/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)
primera <- import("input/puestos_primera_2022.rds" , trust=T)

## 1.2. la base: el censo, el departamento y quien gano cada puesto en la primera vuelta
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , gana_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      inner_join(primera , by="puesto") %>%
      mutate(dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
nrow(db)

## 1.3. split data (la particion fija del curso)
train <- db %>% subset(muestra=="entrenamiento")
test <- db %>% subset(muestra=="prueba")
c(train = nrow(train) , test = nrow(test))

## 1.4. mirar la y antes de cualquier modelo: la primera vuelta tiene cuatro categorias
table(train$gana_primera)

## 1.5. los dos targets de hoy: gana fico y gana petro (primera vuelta)
train$gana_fico <- as.integer(train$gana_primera=="gutierrez")
test$gana_fico <- as.integer(test$gana_primera=="gutierrez")
train$gana_petro1 <- as.integer(train$gana_primera=="petro")
test$gana_petro1 <- as.integer(test$gana_primera=="petro")

##==: 2. una red chiquita para fico: verla por dentro :==##

## 2.1. cuatro variables, estandarizadas con el entrenamiento (los pesos quedan comparables)
v4 <- c("educ_superior" , "afro" , "serv_internet" , "log_registrados")
S4 <- scale(train[,v4])
train_chica <- data.frame(gana_fico = train$gana_fico , S4)
test_chica <- data.frame(scale(test[,v4] , attr(S4,"scaled:center") , attr(S4,"scaled:scale")))

## 2.2. la red: 4 entradas, 4 neuronas, 1 salida (trace=T: mire bajar la log-loss en vivo)
set.seed(2026)
red_chica <- nnet(gana_fico ~ . , data=train_chica , size=4 , maxit=200 , entropy=T)

## cuantos pesos tiene
length(red_chica$wts)

## 2.3. pintar los pesos, uno a uno: b->h1 es el intercepto de la neurona 1, i1->h1 el peso de educ_superior
summary(red_chica)

## 2.4. pintar la red: grosor = tamano del peso; negro positivo, gris negativo
plotnet(red_chica , pad_x=0.7)

## 2.5. pintar la funcion de activacion: la sigmoide que dobla el indice z de cada neurona
curve(1/(1 + exp(-x)) , from=-6 , to=6 , lwd=2 , col="blue" , xlab="z" , ylab="A = g(z)")
abline(h=c(0,1) , lty=2 , col="gray")

## 2.6. la calibracion a pasitos: la misma red detenida en 1, 2, 5, ... iteraciones
## (para jugar con esto en el navegador: playground.tensorflow.org)
pasos <- c(1 , 2 , 5 , 10 , 25 , 50 , 100 , 200)
perdida <- rep(NA , length(pasos))
for (i in 1:length(pasos)){
     set.seed(2026)
     m <- nnet(gana_fico ~ . , data=train_chica , size=4 , maxit=pasos[i] , entropy=T , trace=F)
     perdida[i] <- m$value
}
data.frame(iteraciones = pasos , log_loss = perdida)

## plot calibracion
plot(pasos , perdida , type="b" , pch=16 , log="x" , xlab="iteraciones (escala log)" , ylab="log-loss en el entrenamiento")

## 2.7. la probabilidad de que gane fico en cada puesto de prueba
test$p_fico <- as.vector(predict(red_chica , test_chica))
summary(test$p_fico)

##==: 3. la misma red chiquita para petro :==##

## 3.1. mismas variables, mismo tamano; solo cambia la y
train_chica$gana_fico <- NULL
train_chica$gana_petro1 <- train$gana_petro1
set.seed(2026)
red_petro <- nnet(gana_petro1 ~ . , data=train_chica , size=4 , maxit=200 , entropy=T , trace=F)
test$p_petro <- as.vector(predict(red_petro , test_chica))

## 3.2. dos probabilidades por puesto, una por categoria
test %>% select(nombre , municipio , gana_primera , p_fico , p_petro) %>% head(5)

##==: 4. de las probabilidades a la prediccion :==##

## 4.1. la regla: la categoria con la probabilidad mas alta, siempre que pase del umbral
tau <- 0.5
test$pred <- "ninguna pasa"
test$pred[test$p_petro >= test$p_fico & test$p_petro >= tau] <- "petro"
test$pred[test$p_fico > test$p_petro & test$p_fico >= tau] <- "gutierrez"

## 4.2. la tabla: fico casi nunca pasa el umbral (la clase rara de la parte 1) y a hernandez nadie lo modelo
table(real = test$gana_primera , predicho = test$pred)

## la exactitud de la regla, y la exactitud cuando si se pronuncia
mean(test$pred==test$gana_primera)
mean(test$pred[test$pred!="ninguna pasa"]==test$gana_primera[test$pred!="ninguna pasa"])

## 4.3. target predicho vs target original
test %>% select(nombre , municipio , gana_primera , p_fico , p_petro , pred) %>% head(5)

##==: 5. una red grande: ya no se puede leer :==##

## 5.1. todas las variables: las 37 del censo estandarizadas + el departamento
v37 <- setdiff(names(censo37) , "puesto")
S37 <- scale(train[,v37])
train_grande <- data.frame(gana_petro = train$gana_petro , S37 , dpto = train$dpto)
test_grande <- data.frame(scale(test[,v37] , attr(S37,"scaled:center") , attr(S37,"scaled:scale")) , dpto = test$dpto)

## 5.2. la pregunta de siempre: gana petro la segunda vuelta; 16 neuronas (tarda cerca de un minuto)
set.seed(2026)
red_grande <- nnet(gana_petro ~ . , data=train_grande , size=16 , decay=0.1 , maxit=1000 , MaxNWts=10000 , entropy=T , trace=F)

## cuantos pesos tiene ahora
length(red_grande$wts)

## 5.3. pintarla: la misma figura de la chiquita vuelta una marana (ese es el punto)
plotnet(red_grande , circle_cex=2 , cex_val=0.4)

## 5.4. su auc en la prueba (la probabilidad de ordenar bien un par, semana 4)
test$p_red <- as.vector(predict(red_grande , test_grande))
r <- rank(test$p_red)
n1 <- sum(test$gana_petro==1)
n0 <- sum(test$gana_petro==0)
auc_red <- (sum(r[test$gana_petro==1]) - n1*(n1+1)/2)/(n1*n0)
auc_red

##==: 6. el competidor: un random forest, y ya :==##

## 6.1. el bosque con las mismas variables (los arboles no necesitan estandarizar)
set.seed(2026)
bosque <- randomForest(x = data.frame(train[,v37] , dpto=train$dpto) , y = factor(train$gana_petro) , ntree=300)

## 6.2. su auc en la prueba
test$p_bosque <- predict(bosque , data.frame(test[,v37] , dpto=test$dpto) , type="prob")[,"1"]
r <- rank(test$p_bosque)
auc_bosque <- (sum(r[test$gana_petro==1]) - n1*(n1+1)/2)/(n1*n0)
auc_bosque

##==: 7. la carrera final :==##

## 7.1. las dos cajas negras, lado a lado
## mira: el bosque gana, 0,943 contra 0,913; la red no tenia que ganar: hoy el punto era verla por dentro
data.frame(modelo = c("red grande (16 neuronas)" , "random forest (300 arboles)") ,
           auc_test = c(auc_red , auc_bosque))
