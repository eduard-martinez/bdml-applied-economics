##============================================================================##
# 00_generar_datos.R  —  Avisos simulados de apartamentos
#------------------------------------------------------------------------------#
# Produce input/vivienda.rds (grano: aviso). El precio sale de una f verdadera
# del area (cubica en logaritmos) mas ruido, asi que en clase conocemos f.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)
set.seed(10993)

## path
out <- "input"

##============================================================================##
##=== 1. Simular avisos  (grano: aviso)                                    ===##
##============================================================================##

## area en m2 (lognormal, entre 30 y 400)
n    <- 5000
area <- exp(rnorm(n, mean = 4.5, sd = 0.45))
area <- pmin(pmax(round(area), 30), 400)

## f verdadera: log(precio en millones) como funcion del log(area)
la <- log(area)
f  <- 6.2 + 0.9 * (la - 4.5) - 0.30 * (la - 4.5)^2 + 0.15 * (la - 4.5)^3

## precio observado = f + ruido (sd = 0.25)
log_precio <- f + rnorm(n, mean = 0, sd = 0.25)
precio     <- round(exp(log_precio))

vivienda <- data.frame(id = 1:n, area = area, precio = precio)
head(vivienda)

## export data
export(vivienda, file.path(out, "vivienda.rds"))
