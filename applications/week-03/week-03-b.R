## Big Data y Machine Learning para Economia Aplicada
## week-03-b: zoom al juez y la perilla en el pais (cv, ridge y lasso con las 37 del censo)
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

## 1.2. la misma base de week-03.R: los 12.000 puestos del pais con las 37 del censo
db <- puestos %>%
      select(puesto , nombre , municipio , departamento , voto_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")
c(puestos = nrow(db) , variables = length(x_censo))

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 bajo llave)

## subset test
test <- db %>% subset(muestra=="prueba")

## subset train
train <- db %>% subset(muestra=="entrenamiento")

## validar
c(train = nrow(train) , test = nrow(test))

## 1.4. las matrices de glmnet (glmnet estandariza cada columna por dentro)
x_train <- as.matrix(train[,x_censo])
y_train <- train$voto_petro

##==: 2. el juez por dentro: 10 pliegues, 10 errores, un promedio y su ee :==##

## 2.1. los 10 pliegues del pais (semilla 2026, los mismos de la parte C de week-03.R)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))
table(pliegue)

## 2.2. el modelo a juzgar: ols con las 37 del censo, y su rmse dentro de la muestra
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
## con 9.602 puestos el veredicto es firme: el ee queda chico frente al mse
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
## con 9.602 puestos y 37 columnas el juez elige el extremo: lambda_min es el lambda mas
## chico de la grilla y sus coeficientes son los de ols (no hay varianza que comprar);
## el encogimiento se ve en la columna de 1se, un precio mas alto
betas_r <- as.matrix(ridge$glmnet.fit$beta)
zoom_r <- data.frame(lambda_cero = betas_r[,ncol(betas_r)] ,
                     lambda_min = betas_r[,i_min_r] ,
                     lambda_1se = betas_r[,i_1se_r])
zoom_r %>% arrange(desc(abs(lambda_cero))) %>% round(3) %>% head(8)

## cuantas quedan en cero exacto aun con el precio de 1se: ninguna (ridge nunca apaga)
sum(zoom_r$lambda_1se==0)

## 3.5. pintar el encogimiento (con el precio de 1se: con el del juez seria la diagonal):
## cada punto es una variable; todos van hacia cero y ninguno llega
plot(zoom_r$lambda_cero , zoom_r$lambda_1se , pch=16 , col="blue" ,
     xlab="coeficiente con lambda casi 0 (ols)" , ylab="coeficiente con el lambda de 1se")
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

## 4.6. lo que queda con cada lambda: 36 con el minimo y 30 con la regla 1se
## (con n grande la penalizacion apenas muerde: casi todas las 37 aportan)
sum(betas_l[,i_min_l]!=0)
coef_1se <- coef(lasso , s="lambda.1se")
round(coef_1se[coef_1se[,1]!=0 , , drop=F] , 3)

## pintar las sobrevivientes de 1se (sin el intercepto)
b_1se <- coef_1se[coef_1se[,1]!=0 , ][-1]
par(mar=c(4,10,2,1))
barplot(sort(b_1se) , horiz=T , las=1 , col="lightblue" , cex.names=0.6 ,
        xlab="coeficiente" , main="lo que sobrevive con lambda 1se")
par(mar=c(5,4,4,2) + 0.1)

##==: 5. la escala: estandarizar antes de penalizar :==##

## 5.1. las 37 no viven en la misma escala (la promesa del comentario de 1.4)
## el peaje cobra por el tamano del coeficiente: una variable de unidades chicas
## necesita un coeficiente grande (cara) y una de unidades grandes, uno chico (barata)
sds <- apply(x_train , 2 , sd)
round(head(sort(sds) , 3) , 2)
round(tail(sort(sds) , 3) , 2)

## 5.2. el mismo lasso sin estandarizar (standardize=F): otro juez y otra lista
lasso_crudo <- cv.glmnet(x=x_train , y=y_train , alpha=1 , foldid=pliegue , standardize=F)
c(rmse_cv_min = sqrt(min(lasso_crudo$cvm)) ,
  coeficientes = unname(lasso_crudo$nzero[which.min(lasso_crudo$cvm)]))

## el orden de entrada cambia: sin estandarizar mandan las variables baratas (sd grande)
betas_c <- as.matrix(lasso_crudo$glmnet.fit$beta)
entrada_c <- apply(betas_c!=0 , 1 , function(x) which(x)[1])
names(sort(entrada_c))[1:6]

## 5.3. estandarizar a mano y apagar standardize: vuelven las cifras de la seccion 4
## (glmnet ya lo hacia por dentro; por eso standardize=T es el default y los
## coeficientes se reportan en las unidades originales; el tercer decimal difiere
## porque glmnet divide la sd por n y scale() por n - 1)
lasso_z <- cv.glmnet(x=scale(x_train) , y=y_train , alpha=1 , foldid=pliegue , standardize=F)
c(default = sqrt(min(lasso$cvm)) , a_mano = sqrt(min(lasso_z$cvm)) , crudo = sqrt(min(lasso_crudo$cvm)))

##==: 6. tabla final :==##

## rmse dentro y de cv de los cuatro modelos (el juez decide; la prueba sigue bajo llave)
## las cuatro filas casi empatan: con 9.602 puestos las 37 no necesitan peaje;
## la perilla paga cuando las columnas crecen (las 1.291 de la parte C de week-03.R)
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
