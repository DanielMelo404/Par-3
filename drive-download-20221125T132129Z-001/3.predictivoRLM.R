####selección de modelos lineales por criterios predictivos

################## Selección del mejor subconjunto de variables #########
#########################################################################

###1. Lectura y limpieza de datos
library(ISLR2)
help(Hitters)
View(Hitters)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
###
Hitters <- na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

###2. Encontrar los mejores modelos para cada cantidad de variables usando SCE
library(leaps)
regfit.full <- regsubsets(Salary ~ ., Hitters) #Todos los criterios de información coinciden en la selección para 
#un mismo tamaño (AIC, BIC, R^2 ajustado)
summary(regfit.full)

###
regfit.full <- regsubsets(Salary ~ ., data = Hitters,
                          nvmax = 19)
reg.summary <- summary(regfit.full)
###
names(reg.summary)
###
reg.summary$rsq
###
par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Número de variables",
     ylab = "SCE", type = "l")
plot(reg.summary$adjr2, xlab = "Número de variables",
     ylab = " R2 Ajustado", type = "l")
###
which.max(reg.summary$adjr2)
points(11, reg.summary$adjr2[11], col = "red", cex = 2, 
       pch = 20)
###
plot(reg.summary$cp, xlab = "Número de variables",
     ylab = "Cp", type = "l")
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], col = "red", cex = 2,
       pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic, xlab = "Número de variables",
     ylab = "BIC", type = "l")
points(6, reg.summary$bic[6], col = "red", cex = 2,
       pch = 20)

###3. Visualización de las variables para cada criterio
x11()
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
### Coeficientes del mejor modelo con 6 variables
coef(regfit.full, 6)

################## REGRESION FORWARD Y BACKWARD #########
#########################################################################

### 1. Forward y backward regression
regfit.fwd <- regsubsets(Salary ~ ., data = Hitters,
                         nvmax = 19, method = "forward")
summary(regfit.fwd)
#Funciona incluso en modelos con p>n. Use data=Hitters[1:10,]

regfit.bwd <- regsubsets(Salary ~ ., data = Hitters,
                         nvmax = 19, method = "backward")
summary(regfit.bwd)

library(glmtoolbox)
help("stepCriterion.glm") #Ventaja. Tiene métodos híbridos y tiene función scope.
#Desventaja: toca escribirle explícitamente las variables. No sirve si p>n
mod<-lm(Salary ~ AtBat + Hits + HmRun + Runs + RBI + Walks, data = Hitters)
summary(mod)
stepCriterion(mod,direction = "forward",criterion = "aic")
scope=list(lower=~Hits,upper=~AtBat*Hits*HmRun*Runs*RBI*Walks)
stepCriterion(mod,direction = "forward",criterion = "aic",scope=scope)

### 2. Comparación del mejor modelo con 7 variables por los tres métodos
coef(regfit.full, 7)
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)

################## Elegir modelos con base en el enfoque predictivo #########
#########################################################################

#1. Escoger un porcentaje de datos para entrenamiento y otro para testeo.

set.seed(1)
train1<-sample(1:nrow(Hitters),size=0.7*nrow(Hitters),replace=FALSE)
train<-rep(FALSE,nrow(Hitters))
train[train1] <- TRUE
test <- (!train)

### 2.Entrenar y encontrar los mejores modelos usando solo los datos de ENTRENAMIENTO
regfit.best <- regsubsets(Salary ~ .,
                          data = Hitters[train, ], nvmax = 19)

### 3. Crear una matriz con las columnas de X para el testeo
test.mat <- model.matrix(Salary ~ ., data = Hitters[test, ])

### 4. Calcular el error de predicción de la parte del testeo para los modelos
val.errors <- rep(NA, 19)
for (i in 1:19) {
  coefi <- coef(regfit.best, id = i)
  pred <- test.mat[, names(coefi)] %*% coefi
  val.errors[i] <- mean((Hitters$Salary[test] - pred)^2)
}

### 5. Buscar el mejor modelo
val.errors
which.min(val.errors)
coef(regfit.best, 12)

### 6. Comparar ahora con los coeficientes estimados sobre todo el set de datos
regfit.best <- regsubsets(Salary ~ ., data = Hitters,
                          nvmax = 19)
coef(regfit.best, 12)

###Función útil
predict.regsubsets <- function(object, newdata, id, ...) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

############## Enfoque por k-fold cross-validation

### Creación de los folds
k <- 10
n <- nrow(Hitters)
set.seed(1)
folds <- sample(rep(1:k, length = n))
table(folds)
cv.errors <- matrix(NA, k, 19,
                    dimnames = list(NULL, paste(1:19)))

### Entrenar los modelos
for (j in 1:k) {
  best.fit <- regsubsets(Salary ~ .,
                         data = Hitters[folds != j, ],
                         nvmax = 19)
  for (i in 1:19) {
    pred <- predict(best.fit, Hitters[folds == j, ], id = i)
    cv.errors[j, i] <-
      mean((Hitters$Salary[folds == j] - pred)^2)
  }
}
### Resumir los errores por modelo
mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors
par(mfrow = c(1, 1))
plot(mean.cv.errors, type = "b")
###
reg.best <- regsubsets(Salary ~ ., data = Hitters,
                       nvmax = 19)
coef(reg.best, 10)

################## Regresión ridge y LASSO #########
#########################################################################
library(glmnet)

###
x <- model.matrix(Salary ~ ., Hitters)[, -1] #transforma las variables cualitativas 
#en dummy porque ridge y LASSO solo aceptan var. cuantitativas
y <- Hitters$Salary

### Ridge Regression

### Secuencia de búsqueda para lambda
grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid) #alpha=0 significa ridge y =1 LASSO
#glmnet estandariza por defecto. Si se desea remover ese efecto, standardize=FALSE

dim(coef(ridge.mod)) #100 filas (una por lambda) con cada estimación
ridge.mod$lambda[50]
coef(ridge.mod)[, 50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))

predict(ridge.mod, s = 50, type = "coefficients")[1:20, ] #se puede usar para 
#obtener los coeficientes de un nuevo valor de lambda, que se ingresa en s

###Enfoque de validación
y.test <- y[test]
#Calibración del modelo (alpha=0: ridge, alpha=1:LASSO)
ridge.mod <- glmnet(x[train, ], y[train], alpha = 0,
                    lambda = grid, thresh = 1e-12) #thresh es un parámetro de convergencia numérica

###¿Cómo le va a un modelo con lambda=4?
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test, ])
mean((ridge.pred - y.test)^2)

###Modelo de solo media (modelo de referencia)
mean((mean(y[train]) - y.test)^2)

###¿Cómo le va a un modelo con lambda=10^10? mismo result. anterior
ridge.pred <- predict(ridge.mod, s = 1e10, newx = x[test, ])
mean((ridge.pred - y.test)^2)

###¿Cómo le va a un modelo con lambda=0? mismo OLS
ridge.pred <- predict(ridge.mod, s = 0, newx = x[test, ],
                      exact = T, x = x[train, ], y = y[train])
mean((ridge.pred - y.test)^2)
lm(y ~ x, subset = train)
predict(ridge.mod, s = 0, exact = T, type = "coefficients",
        x = x[train, ], y = y[train])[1:20, ]
#exact=T sirve para que R calcule de manera exacta los coeficientes

### Determinación lambda por valid. Cruzada
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0) #10-fold crossval.
x11()
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

### MSE de predicción
ridge.pred <- predict(ridge.mod, s = bestlam,
                      newx = x[test, ])
mean((ridge.pred - y.test)^2)

### Coeficientes estimados con el set completo
out <- glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlam)[1:20, ]

### The Lasso
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1,
                    lambda = grid)
x11()
plot(lasso.mod)
###
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s = bestlam,
                      newx = x[test, ])
mean((lasso.pred - y.test)^2)
###
out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients",
                      s = bestlam)[1:20, ]
lasso.coef
lasso.coef[lasso.coef != 0]
