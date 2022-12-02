################Modelos generalizados aditivos########

library(ISLR2)
attach(Wage)
library(splines)
help(Wage)

help(ns) #(spline natural) para ajustar splines c�bicas naturales.
# los grados de libertad son iguales a df son proporcionales al n�mero de nodos
### a) Estimar un modelo GAM con una spline natural de a�o y de edad. Con lm.
gam1 <- lm(wage ~ ns(year, 4) + ns(age, 5) + education,
           data = Wage)
summary(gam1)

### Usar el paquete GAM de la librer�a gam es m�s recomendable.

library(gam)
help(s) #(spline suavizada) diferencia entre df (grados de libertad objetivo) y spar (suavizamiento con n nodos)
help(smooth.spline) #hace la regresion simple por splines (solo dos variables)
help(gam)

gam.m3 <- gam(wage ~ s(year, 4) + s(age, 5) + education,
              data = Wage)
### b) Resultados
par(mfrow = c(1, 3))
plot(gam.m3, se = TRUE, col = "blue")
### Resultados de lm
plot.Gam(gam1, se = TRUE, col = "red")
### c) Comparar 3 modelos: uno sin a�o, otro con a�o lineal y otro con una spline de a�o
gam.m1 <- gam(wage ~ s(age, 5) + education, data = Wage)
gam.m2 <- gam(wage ~ year + s(age, 5) + education,
              data = Wage)
anova(gam.m1, gam.m2, gam.m3, test = "F")
### d) Resultados (mire los valores param�tricos (significancia relaciones lineales) versus no param�tricos (lineal vs no lineal))
summary(gam.m3)
summary(gam.m2)
### e) El modelo se puede usar para predecir
x11()
preds <- predict(gam.m2, newdata = Wage)
plot(Wage$wage,preds)

### f) Usando una regresion local en vez de splines
help(lo)
gam.lo <- gam(wage ~ s(year , df = 4) + lo(age , span = 0.7) + education ,data = Wage)
plot.Gam(gam.lo, se = TRUE , col = "green")
summary(gam.lo)

###Introduciendo una interacci�n con lo()
gam.lo.i <- gam(wage ~ s(year , df = 4) + lo(age , span = 0.7) +lo(year , age , span = 0.5) + education ,
                data = Wage)
summary(gam.lo.i)
plot.Gam(gam.lo.i, se = TRUE , col = "orange")



