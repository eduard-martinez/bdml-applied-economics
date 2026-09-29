## Big Data y Machine Learning para Economia Aplicada
## week-03-b: zoom al juez y la perilla (cv, ridge y lasso con las 37 del censo)
## Last run: Sep 29, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , glmnet)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-03/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)

## 1.2. la misma base de week-03.R: cali y su area metropolitana con las 37 del censo
db <- puestos %>%
      select(puesto , nombre , municipio , departamento , voto_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      subset(!is.na(educ_superior) & !is.na(afro))
df <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
x_censo <- setdiff(names(censo37) , "puesto")
c(puestos = nrow(df) , variables = length(x_censo))

## 1.3. split data (la particion de la semana 2: semilla 2026, 25% a la prueba)
set.seed(2026)

## sample
sample <- sample(x = nrow(df) , round(nrow(df)*0.25))

## subset test
test <- df[sample,]

## subset train
train <- df[-sample,]

## validar
c(train = nrow(train) , test = nrow(test))

## 1.4. las matrices de glmnet (glmnet estandariza cada columna por dentro)
x_train <- as.matrix(train[,x_censo])
y_train <- train$voto_petro

##==: 2. el juez por dentro: 10 pliegues, 10 errores, un promedio y su ee :==##

## 2.1. los pliegues de la clase (semilla 2026, los mismos de week-03.R)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))
table(pliegue)

## 2.2. el modelo a juzgar: ols con las 37 (el ajuste dentro es un espejismo)
f_censo <- as.formula(paste0("voto_petro ~ " , paste(x_censo , collapse=" + ")))
modelo_ols <- lm(f_censo , data=train)
sqrt(mean(residuals(modelo_ols)^2))

## 2.3. el cv a mano: cada pliegue valida una vez y deja su propio rmse
cv_ols <- data.frame(pliegue = 1:10 , n = NA , mse = NA , rmse = NA)
error2 <- rep(NA , nrow(train))
for (j in 1:10){
     modelo_j <- lm(f_censo , data=train[pliegue!=j,])
     pred_j <- predict(modelo_j , train[pliegue==j,])
     error2[pliegue==j] <- (train$voto_petro[pliegue==j] - pred_j)^2
     cv_ols$n[j] <- sum(pliegue==j)
     cv_ols$mse[j] <- mean(error2[pliegue==j])
     cv_ols$rmse[j] <- sqrt(cv_ols$mse[j])
}
cv_ols %>% mutate(mse = round(mse,1) , rmse = round(rmse,2))

## 2.4. el estimador y su ee: el rmse de cv es un promedio, y un promedio trae error estandar
## (la formula del deck: ee = d.e.(mse_1 , ... , mse_J)/raiz de J)
ee_ols <- sd(cv_ols$mse)/sqrt(10)
c(rmse_cv = sqrt(mean(error2)) , ee_del_mse = ee_ols)

## 2.5. pintar el juez: los 10 mse, su promedio (linea) y promedio +/- ee (punteadas)
## (la dispersion entre pliegues es exactamente lo que mide el ee)
plot(cv_ols$pliegue , cv_ols$mse , pch=16 , col="blue" ,
     xlab="pliegue" , ylab="mse de validacion")
abline(h = mean(cv_ols$mse) , col="blue" , lwd=2)
abline(h = mean(cv_ols$mse) + c(-1,1)*ee_ols , lty=2)

##==: 3. ridge por dentro: encoger sin apagar :==##

## 3.1. el juez de glmnet con los mismos pliegues (la grilla se alarga para ver el minimo)
ridge <- cv.glmnet(x=x_train , y=y_train , alpha=0 , foldid=pliegue , lambda.min.ratio=1e-6 , nlambda=150)

## 3.2. pintar los lambda: el camino de los 37 coeficientes y la curva del juez
## (cada barra de la curva es el ee de ese lambda, el mismo de la seccion 2)
plot(ridge$glmnet.fit , xvar="lambda")
plot(ridge)

## 3.3. los dos lambda del juez y su rmse de cv
i_min_r <- which.min(abs(ridge$lambda - ridge$lambda.min))
i_1se_r <- which.min(abs(ridge$lambda - ridge$lambda.1se))
c(lambda_min = ridge$lambda.min , rmse_cv_min = sqrt(ridge$cvm[i_min_r]) ,
  lambda_1se = ridge$lambda.1se , rmse_cv_1se = sqrt(ridge$cvm[i_1se_r]))

## 3.4. zoom a las variables: tres paradas del camino (lambda casi 0, el elegido y el de 1se)
## una fila por variable, ordenadas por el tamano del coeficiente sin penalizar
betas_r <- as.matrix(ridge$glmnet.fit$beta)
zoom_r <- data.frame(lambda_cero = betas_r[,ncol(betas_r)] ,
                     lambda_min = betas_r[,i_min_r] ,
                     lambda_1se = betas_r[,i_1se_r])
zoom_r %>% arrange(desc(abs(lambda_cero))) %>% round(3) %>% head(8)

## cuantas variables quedan en cero exacto con el lambda elegido: ninguna
sum(zoom_r$lambda_min==0)

## 3.5. pintar el encogimiento: cada punto es una variable; todos van hacia cero, ninguno llega
plot(zoom_r$lambda_cero , zoom_r$lambda_min , pch=16 , col="blue" ,
     xlab="coeficiente con lambda casi 0 (ols)" , ylab="coeficiente con el lambda del juez")
abline(a=0 , b=1 , lty=2)
abline(h=0 , v=0 , col="gray70")

##==: 4. lasso por dentro: encoger y apagar :==##

## 4.1. el juez de glmnet con los mismos pliegues
lasso <- cv.glmnet(x=x_train , y=y_train , alpha=1 , foldid=pliegue)

## 4.2. pintar los lambda: el camino (las curvas mueren en cero exacto) y la curva del juez
plot(lasso$glmnet.fit , xvar="lambda")
plot(lasso)

## 4.3. los dos lambda del juez y su rmse de cv
i_min_l <- which.min(abs(lasso$lambda - lasso$lambda.min))
i_1se_l <- which.min(abs(lasso$lambda - lasso$lambda.1se))
c(lambda_min = lasso$lambda.min , rmse_cv_min = sqrt(lasso$cvm[i_min_l]) ,
  lambda_1se = lasso$lambda.1se , rmse_cv_1se = sqrt(lasso$cvm[i_1se_l]))

## 4.4. de donde sale lambda.1se: el ee del minimo define el empate
## (el mayor lambda cuyo mse de cv queda debajo de minimo + ee: mas simple, sin perder de verdad)
techo <- lasso$cvm[i_min_l] + lasso$cvsd[i_min_l]
c(a_mano = max(lasso$lambda[lasso$cvm <= techo]) , de_glmnet = lasso$lambda.1se)

## 4.5. cuantas variables sobreviven a cada precio (el eje de arriba de las figuras del deck)
plot(log(lasso$lambda) , lasso$nzero , type="s" , col="blue" , lwd=2 ,
     xlab="log lambda: mas penalizacion ->" , ylab="variables con coeficiente distinto de cero")
abline(v = log(c(lasso$lambda.min , lasso$lambda.1se)) , lty=2)

## el orden de entrada al camino: que compra primero el lasso cuando el precio baja
betas_l <- as.matrix(lasso$glmnet.fit$beta)
entrada <- apply(betas_l!=0 , 1 , function(x) which(x)[1])
names(sort(entrada))[1:6]

## 4.6. lo que queda con cada lambda: cuantas con el minimo, y cuales con la regla 1se
sum(betas_l[,i_min_l]!=0)
coef_1se <- coef(lasso , s="lambda.1se")
round(coef_1se[coef_1se[,1]!=0 , , drop=F] , 3)

## pintar las sobrevivientes de 1se (sin el intercepto)
b_1se <- coef_1se[coef_1se[,1]!=0 , ][-1]
par(mar=c(4,10,2,1))
barplot(sort(b_1se) , horiz=T , las=1 , col="lightblue" ,
        xlab="coeficiente" , main="lo que sobrevive con lambda 1se")
par(mar=c(5,4,4,2) + 0.1)

##==: 5. tabla final :==##

## rmse dentro y de cv de los cuatro modelos (el juez decide; la prueba sigue bajo llave)
tabla <- data.frame(modelo = c("ols con las 37 del censo","ridge (lambda min)","lasso (lambda min)","lasso (lambda 1se)"),
                    coeficientes = c(sum(!is.na(coef(modelo_ols))) - 1 , unname(ridge$nzero[i_min_r]) ,
                                     unname(lasso$nzero[i_min_l]) , unname(lasso$nzero[i_1se_l])),
                    rmse_train = c(sqrt(mean(residuals(modelo_ols)^2)) ,
                                   sqrt(mean((y_train - predict(ridge , x_train , s="lambda.min"))^2)) ,
                                   sqrt(mean((y_train - predict(lasso , x_train , s="lambda.min"))^2)) ,
                                   sqrt(mean((y_train - predict(lasso , x_train , s="lambda.1se"))^2))),
                    rmse_cv = c(sqrt(mean(error2)) , sqrt(ridge$cvm[i_min_r]) ,
                                sqrt(lasso$cvm[i_min_l]) , sqrt(lasso$cvm[i_1se_l])))
tabla %>% mutate(rmse_train = round(rmse_train,2) , rmse_cv = round(rmse_cv,2))
