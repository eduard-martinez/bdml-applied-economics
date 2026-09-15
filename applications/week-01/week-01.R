#' ---
#' title: "Semana 1: Predecir, no explicar"
#' subtitle: "Big Data y Machine Learning para Economía Aplicada · Universidad Icesi · 2026-2"
#' author: "Eduard F. Martínez González"
#' date: "Aplicación en R de la sesión 1"
#' output:
#'   html_document:
#'     theme: flatly
#'     highlight: haddock
#'     toc: true
#'     toc_depth: 2
#'     toc_float: true
#' ---
#'
#+ setup, include=FALSE
knitr::opts_chunk$set(message = FALSE, warning = FALSE, fig.width = 7.5, fig.height = 4.3, fig.align = "center")
#'
#' **La idea de hoy:** qué tan bien un modelo ajusta los datos que ya vio no dice nada sobre qué tan bien
#' predice un dato que no ha visto. Lo vemos tres veces: con avisos reales de apartamentos en Bogotá,
#' agregando un dato nuevo al entrenamiento, y con una simulación en la que conocemos la verdad.
#'
#' *Cómo usar este archivo:* `week-01.R` es un script de R normal; se corre línea por línea en RStudio.
#' Los comentarios que empiezan con `#'` son el texto de esta página, que se genera con
#' `rmarkdown::render("week-01.R")`.
#'
#' # 0. Configuración
paquetes <- c("dplyr", "ggplot2", "tidyr")
for (p in paquetes) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(dplyr); library(ggplot2); library(tidyr)
theme_set(theme_minimal(base_size = 12))
set.seed(10993)   # el NRC del curso; cámbienla al final y vean qué cambia

#' # 1. Los datos: avisos de venta de apartamentos en Bogotá
#' Avisos publicados en Properati entre abril de 2019 y mayo de 2020 (`input/apartamentos_bogota.rds`),
#' depurados: apartamentos en venta, área construida entre 25 y 400 m² y precio entre 80 y 3.000 millones de pesos.
db <- readRDS("input/apartamentos_bogota.rds") %>%
  mutate(precio_mill = precio / 1e6, log_precio = log(precio_mill), log_area = log(area))
glimpse(db)
summary(db$precio_mill)
summary(db$area)
#' En logaritmos, la relación entre área y precio es casi una recta. Casi.
ggplot(slice_sample(db, n = 5000), aes(area, precio_mill)) +
  geom_point(alpha = 0.15, size = 0.8) + scale_x_log10() + scale_y_log10() +
  labs(x = "área construida (m², escala log)", y = "precio (millones de COP, escala log)",
       title = "5.000 avisos al azar: ¿qué forma tiene f?")

#' # 2. La partición va antes de cualquier modelo
#' Regla de oro #2: el conjunto de prueba se aparta primero y se usa al final, una sola vez.
#' Para que el sobreajuste se vea a simple vista, entrenamos con **100 avisos** y probamos con 5.000.
idx   <- sample(nrow(db), 100)
train <- db[idx, ]
test  <- db[sample(setdiff(seq_len(nrow(db)), idx), 5000), ]
c(entrenamiento = nrow(train), prueba = nrow(test))

#' # 3. Tres modelos sobre los mismos 100 puntos
#' Polinomios del logaritmo del área de grado 1, 3 y 10. Dos funciones que usaremos todo el semestre:
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

ajustar <- function(datos, d) lm(log_precio ~ poly(log_area, d), data = datos)
malla   <- function(...) data.frame(log_area = seq(min(...), max(...), length.out = 300))
curva   <- function(m, etiqueta, g) data.frame(modelo = etiqueta, log_area = g$log_area, log_precio = predict(m, g))

grados3 <- c(1, 3, 10)
modelos <- setNames(lapply(grados3, ajustar, datos = train), paste("grado", grados3))
g_train <- malla(train$log_area)
curvas  <- bind_rows(Map(curva, modelos, names(modelos), MoreArgs = list(g = g_train))) %>%
  mutate(modelo = factor(modelo, names(modelos)))

ggplot(train, aes(log_area, log_precio)) + geom_point(alpha = 0.4) +
  geom_line(data = curvas, aes(colour = modelo), linewidth = 1) +
  coord_cartesian(ylim = range(train$log_precio)) +
  labs(x = "log(área)", y = "log(precio)", colour = NULL,
       title = "Grado 1, 3 y 10 ajustados a los mismos 100 avisos")

#' ¿Cuál ajusta mejor? Dentro de la muestra, siempre el más flexible. Fuera de ella, no:
bondad <- function(m, etiqueta) data.frame(
  modelo             = etiqueta,
  R2_entrenamiento   = r2(train$log_precio, fitted(m)),
  R2_prueba          = r2(test$log_precio, predict(m, test)),
  RMSE_entrenamiento = rmse(train$log_precio, fitted(m)),
  RMSE_prueba        = rmse(test$log_precio, predict(m, test)))
bind_rows(unname(Map(bondad, modelos, names(modelos)))) %>% mutate(across(-modelo, ~ round(.x, 3)))
#' El grado 10 ajusta mejor los 100 avisos que vio y da disparates en la prueba: para áreas fuera del rango
#' de esos 100 avisos, el polinomio extrapola sin control.
#'
#' **Pregunta para la sala:** si les pidieran un modelo de valoración automática, ¿cuál de los tres
#' entregarían y con qué número lo defenderían?

#' # 4. La curva U: grado 1 a 12
res <- bind_rows(lapply(1:12, function(d) bondad(ajustar(train, d), d))) %>% rename(grado = modelo)
res %>% mutate(across(-grado, ~ round(.x, 3)))
res %>%
  pivot_longer(c(RMSE_entrenamiento, RMSE_prueba), names_to = "muestra", values_to = "RMSE") %>%
  ggplot(aes(grado, RMSE, colour = muestra)) + geom_line(linewidth = 1) + geom_point() +
  scale_colour_manual(values = c(RMSE_entrenamiento = "blue", RMSE_prueba = "red"),
                      labels = c("entrenamiento", "prueba (nunca vista)")) +
  coord_cartesian(ylim = c(0.25, 0.6)) + scale_x_continuous(breaks = 1:12) +
  labs(x = "grado del polinomio (complejidad)", y = "RMSE de log(precio)", colour = NULL,
       title = "El error de entrenamiento siempre baja; el de prueba tiene forma de U",
       subtitle = "Los grados altos se salen del gráfico: extrapolan disparates en los extremos del rango de áreas")
#' El error de entrenamiento cae monótonamente: más flexibilidad siempre ajusta mejor la muestra.
#' El de prueba mejora hasta el grado 3 y después el sobreajuste lo destruye. Esta figura es el curso entero.

#' # 5. ¿Qué pasa cuando llega un dato nuevo?
#' Dos preguntas distintas. **Primera:** llega un aviso que el modelo no ha visto, ¿qué precio le pone cada uno?
#' **Segunda:** si agregamos ese aviso al entrenamiento y reajustamos, ¿cuánto cambia cada modelo?
#' La segunda es la varianza del método, vista en vivo.
prediccion <- function(mods, dato) bind_rows(unname(Map(function(m, e)
  data.frame(modelo = e, predicho_mill = round(exp(predict(m, dato)), 0)), mods, names(mods)))) %>%
  as_tibble() %>%
  mutate(observado_mill = dato$precio_mill, error_pct = round(100 * (predicho_mill / observado_mill - 1), 1))

reajustar <- function(nuevos, titulo) {
  train2   <- bind_rows(train, nuevos)
  modelos2 <- setNames(lapply(grados3, ajustar, datos = train2), names(modelos))
  g        <- malla(train2$log_area)
  antes    <- bind_rows(Map(curva, modelos,  names(modelos),  MoreArgs = list(g = g))) %>% mutate(ajuste = "antes")
  despues  <- bind_rows(Map(curva, modelos2, names(modelos2), MoreArgs = list(g = g))) %>% mutate(ajuste = "despues")
  ambos    <- bind_rows(antes, despues) %>% mutate(modelo = factor(modelo, names(modelos)))
  fig <- ggplot(train, aes(log_area, log_precio)) + geom_point(alpha = 0.2) +
    geom_point(data = nuevos, colour = "red", size = 3) +
    geom_line(data = filter(ambos, ajuste == "despues"), aes(colour = modelo), linewidth = 1.1) +
    geom_line(data = filter(ambos, ajuste == "antes"), colour = "black", linetype = 2, linewidth = 0.7) +
    facet_wrap(~ modelo) + coord_cartesian(ylim = range(train2$log_precio)) +
    labs(x = "log(área)", y = "log(precio)", title = titulo,
         subtitle = "negro punteado: ajuste con los 100 avisos; color: después de agregar los puntos rojos") +
    theme(legend.position = "none")
  print(fig)
  ambos %>% pivot_wider(names_from = ajuste, values_from = log_precio) %>%
    mutate(area = exp(log_area), cambio_pct = 100 * (exp(despues - antes) - 1)) %>%
    group_by(modelo) %>%
    summarise(cambio_max_pct   = round(max(abs(cambio_pct)), 1),
              cambio_en_80m2   = round(cambio_pct[which.min(abs(area - 80))], 1),
              cambio_en_200m2  = round(cambio_pct[which.min(abs(area - 200))], 1))
}

#' ## 5.1 Un aviso nuevo típico
nuevo <- test %>% filter(between(area, 70, 90)) %>% slice_sample(n = 1)
nuevo %>% select(area, habitaciones, banos, zona, precio_mill)
prediccion(modelos, nuevo)
reajustar(nuevo, "Se agrega un aviso típico (entre 70 y 90 m²)")

#' ## 5.2 Un aviso nuevo poco común: un apartamento grande
grande <- test %>% filter(area >= 250) %>% slice_sample(n = 1)
grande %>% select(area, habitaciones, banos, zona, precio_mill)
prediccion(modelos, grande)
reajustar(grande, "Se agrega un aviso poco común (250 m² o más)")

#' ## 5.3 Diez avisos nuevos al azar, veinte veces
#' Un solo dato nuevo mueve poco; para ver la varianza hay que repetir el experimento. Veinte veces
#' agregamos diez avisos nuevos al azar a los mismos 100 y reajustamos los tres modelos.
repeticiones <- bind_rows(lapply(1:20, function(r) {
  train2   <- bind_rows(train, slice_sample(test, n = 10))
  modelos2 <- setNames(lapply(grados3, ajustar, datos = train2), names(modelos))
  bind_rows(unname(Map(curva, modelos2, names(modelos2), MoreArgs = list(g = g_train)))) %>% mutate(rep = r)
})) %>% mutate(modelo = factor(modelo, names(modelos)))
ggplot(train, aes(log_area, log_precio)) + geom_point(alpha = 0.2) +
  geom_line(data = repeticiones, aes(colour = modelo, group = rep), alpha = 0.35, linewidth = 0.7) +
  geom_line(data = curvas, colour = "black", linetype = 2, linewidth = 0.7) +
  facet_wrap(~ modelo) + coord_cartesian(ylim = range(train$log_precio)) + theme(legend.position = "none") +
  labs(x = "log(área)", y = "log(precio)", title = "Veinte reajustes, cada uno con diez avisos nuevos distintos",
       subtitle = "negro punteado: el ajuste original con los 100 avisos")
#' ¿Cuánto cambia la predicción de un apartamento de 80 m² y de uno de 200 m² de un reajuste a otro?
repeticiones %>% mutate(area = exp(log_area)) %>%
  filter(area %in% c(area[which.min(abs(area - 80))], area[which.min(abs(area - 200))])) %>%
  mutate(area = round(area)) %>% group_by(modelo, area) %>%
  summarise(desv_pct = round(100 * sd(log_precio), 1), rango_pct = round(100 * diff(range(log_precio)), 1), .groups = "drop")

#' **Lo que acabamos de ver es la varianza.** El modelo rígido casi no se mueve cuando cambian los datos:
#' poca varianza, pero se equivoca siempre en la misma dirección (sesgo). El flexible cambia de forma con
#' cada puñado de datos nuevos: está ajustando el ruido de la muestra, no la señal. Ninguno de los números
#' de la tabla de la sección 3 lo habría contado; hicieron falta datos nuevos.

#' # 6. Simulación: sesgo y varianza cuando conocemos la verdad
#' Con datos reales no conocemos f, así que no podemos separar sesgo de varianza. En una simulación sí.
#' La naturaleza es $f(x) = \sin(2\pi x)$ con ruido $\sigma = 0{,}3$; fijamos $x_0 = 0{,}9$ y repetimos el
#' muestreo 200 veces con $n = 60$ para grado 1, 4 y 12.
f <- function(x) sin(2 * pi * x)
sigma <- 0.3; n <- 60; R <- 200; x0 <- 0.9
sim_pred <- function(d) replicate(R, {
  x <- runif(n); y <- f(x) + rnorm(n, sd = sigma)
  predict(lm(y ~ poly(x, d, raw = TRUE)), data.frame(x = x0))
})
dianas <- bind_rows(lapply(c(1, 4, 12), function(d) data.frame(grado = paste("grado", d), pred = sim_pred(d)))) %>%
  mutate(grado = factor(grado, levels = paste("grado", c(1, 4, 12))))
dianas %>% group_by(grado) %>%
  summarise(sesgo2 = (mean(pred) - f(x0))^2, varianza = var(pred), irreducible = sigma^2,
            ECM_total = sesgo2 + varianza + irreducible, .groups = "drop") %>%
  mutate(across(-grado, ~ round(.x, 4)))
#' Grado 1: sesgo alto y varianza baja. Grado 12: sesgo casi nulo y varianza alta. Grado 4: el mínimo del
#' error total, que es la suma de las tres piezas.
ggplot(dianas, aes(pred, fill = grado)) +
  geom_histogram(bins = 40, alpha = 0.7) + geom_vline(xintercept = f(x0), linetype = 2) +
  facet_wrap(~ grado, ncol = 1) + theme(legend.position = "none") +
  labs(x = expression(hat(f)(x[0]) ~ "en 200 muestras de entrenamiento distintas"), y = NULL,
       title = "La línea punteada es f(x0): ¿dónde está el centro de la nube y qué tan dispersa es?")

#' # 7. Para llevar
#' 1. El error de entrenamiento es optimista y baja siempre con la complejidad (regla de oro #1).
#' 2. El conjunto de prueba se aparta antes de tocar un modelo y se usa una sola vez (regla de oro #2).
#' 3. Un dato nuevo revela lo que la bondad de ajuste esconde: el modelo flexible cambia con cada muestra
#'    (varianza); el rígido se equivoca siempre igual (sesgo). El óptimo está en el medio: la curva U.
#'
#' *Para la casa:* cambien la semilla y el tamaño del entrenamiento (100, 500, 2.000). ¿Se mueve el mínimo de
#' la curva U? ¿En qué dirección y por qué?
