
library(glmtoolbox)
library(aplore3)

# a) Conjunto de datos relacionado con muertes por quemaduras

burn1000 <- aplore3::burn1000
burn1000 <- within(burn1000, death2 <- ifelse(death=="Dead",1,0)) #exito es muerte
str(burn1000)
head(burn1000)
summary(burn1000)

# b-d) ajustar un primer modelo
help(glm)
fit1 <- glm(death2 ~ age + gender + race + tbsa + inh_inj + flame, family=binomial("logit"), data=burn1000)
help(family)
summary(fit1)
adj_pseudoR2<-with(fit1,1-deviance*df.null/(null.deviance*df.residual)) #pseudo R^2

fit1$fitted.values

fitted<-exp(model.matrix(fit1)%*%coef(fit1))/(1+exp(model.matrix(fit1)%*%coef(fit1)))

all.equal(as.numeric(fit1$fitted.values),as.numeric(fitted))

plot(as.numeric(fit1$fitted.values),burn1000$death2,pch=20, col="lightblue", xlab="valores predichos de probabilidad de muerte", ylab="Muerto(1)/Vivo(0)")

#e)  de los parámetros
exp(coef(fit1)[-1]) #interpretación de cuántas veces es más grande/pequeño la razón de chances

(exp(coef(fit1)[-1])-1)*100

exp(confint2(fit1)[-1,])

(exp(confint2(fit1)[-1,])-1)*100

#f) Elección de una función de enlace en términos de su bondad de ajuste

fit2 <- glm(death2 ~ age + gender + race + tbsa + inh_inj + flame, family=binomial("probit"), data=burn1000)
fit3 <- glm(death2 ~ age + gender + race + tbsa + inh_inj + flame, family=binomial("cloglog"), data=burn1000)
fit4 <- glm(death2 ~ age + gender + race + tbsa + inh_inj + flame, family=binomial("cauchit"), data=burn1000)

fit2<-update(fit1,family=binomial("probit"))
fit3<-update(fit1,family=binomial("cloglog"))
fit4<-update(fit1,family=binomial("cauchit"))

cbind(AIC(fit1,fit2,fit3,fit4),BIC(fit1,fit2,fit3,fit4))

#g-h) Selección automática del modelo
fit5 <- glm(death2 ~ age + gender + race + tbsa + inh_inj + flame + age*inh_inj + tbsa*inh_inj, family=binomial("logit"), data=burn1000)
a<-stepCriterion(fit5, direction="backward", criterion="bic")

fit6 <- update(fit5,formula. = a$final)
summary(fit6)

fit7<-update(fit6,family=binomial("probit"))
fit8<-update(fit6,family=binomial("cloglog"))
fit9<-update(fit6,family=binomial("cauchit"))

cbind(AIC(fit1,fit2,fit3,fit4,fit5,fit6,fit7,fit9),BIC(fit1,fit2,fit3,fit4,fit5,fit6,fit7,fit9))

#i) Validación de los supuestos
envelope(fit7)
plot(fit7)

#########################EJERCICIO 2#########################

###a)
library(ISLR2)
names(Smarket)
dim(Smarket)
summary(Smarket)
pairs(Smarket)
help(Smarket)

###b) Ajustar el modelo
glm.fits <- glm(
  Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
  data = Smarket, family = binomial
)
summary(glm.fits)
plot(glm.fits)

###c) Revisar la matriz de confusión cuando el modelo es estimado con todos los datos

glm.probs <- predict(glm.fits, type = "response")
glm.probs[1:10]

glm.pred <- rep("Down", 1250)
glm.pred[glm.probs > .5] = "Up"

table(glm.pred, Smarket$Direction)
(507 + 145) / 1250
mean(glm.pred == Smarket$Direction)


###d) Entrenar el modelo con los datos hasta 2004 y luego testearlo con los de 2005 en adelante
train <- (Smarket$Year < 2005)
Smarket.2005 <- Smarket[!train, ]
dim(Smarket.2005)
Direction.2005 <- Smarket$Direction[!train]
Dir <- ifelse(Direction.2005=="Up",1,0)
###
glm.fits <- glm(
  Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
  data = Smarket, family = binomial, subset = train
)
summary(glm.fits)
glm.probs <- predict(glm.fits, Smarket.2005,
                     type = "response")
###
glm.pred <- rep("Down", 252)
glm.pred[glm.probs > .5] <- "Up"
table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005)
mean(glm.pred != Direction.2005)

#e) 
glm.fits2 <- glm(
  Direction ~ Lag1*Lag2*Lag3*Lag4*Lag5*Volume,
  data = Smarket, family = binomial, subset = train
)

#b<-stepCriterion(glm.fits2, direction="backward", criterion="bic")

summary(glm.fits2)
glm.probs2 <- predict(glm.fits2, Smarket.2005,
                     type = "response")



#f) Desempeño general del modelo
ROCc(cbind(Dir,glm.probs))
ROCc(cbind(Dir,glm.probs2))

#Elección de un tau óptimo por habilidad predictiva
summary(glm.probs) #importante revisar el rango de probabilidades que tiene la respuesta
tau<-seq(0.4,0.6,by=0.02)

AER<-NULL
recall<-NULL
precision<-NULL
F1<-NULL
exito<-"Up"
frac<-"Down"

for (i in 1:length(tau)){
  glm.pred <- rep("Down", 252)
  glm.pred[glm.probs > tau[i]] <- "Up"
  tab<-table(glm.pred, Direction.2005)
  if (!frac %in% rownames(tab)){
    tab<-rbind(tab,c(0,0))
    rownames(tab)[2]<-frac
  } 
  if (!exito %in% rownames(tab)){
    tab<-rbind(tab,c(0,0))
    rownames(tab)[2]<-exito
  }
  AER[i]<-(tab[exito,frac]+tab[frac,exito])/sum(tab)
  precision[i]<-(tab[exito,exito])/sum(tab[exito,])
  recall[i]<-(tab[exito,exito])/sum(tab[,exito])
  F1[i]<-2*precision[i]*recall[i]/(precision[i]+recall[i])
}

cbind(tau,AER,precision,recall,F1)

################# EJERCICIO 3################
#a)
library(dplyr)

stratified <- burn1000 %>%
  group_by(death) %>%
  slice_sample(prop=0.7)

summary(stratified)

test<-burn1000[setdiff(burn1000$id,stratified$id),]

#b) Entrenamiento del modelo
fit5b<-update(fit5,data=stratified)

a<-stepCriterion(fit5b, direction="backward", criterion="bic")

fit6b <- update(fit5b,formula. = a$final)
summary(fit6b)

fit7b<-update(fit6b,family=binomial("probit"))
fit8b<-update(fit6b,family=binomial("cloglog"))
fit9b<-update(fit6b,family=binomial("cauchit"))

#c) Mejor modelo en términos globales

pr6b <- predict(fit6b, newdata=test, type="response")
pr7b <- predict(fit7b, newdata=test, type="response")
pr8b <- predict(fit8b, newdata=test, type="response")
pr9b <- predict(fit9b, newdata=test, type="response")

testres<-test$death2
ROCc(cbind(testres,pr6b))
ROCc(cbind(testres,pr7b))
ROCc(cbind(testres,pr8b))
ROCc(cbind(testres,pr9b))


summary(pr7b) #importante revisar el rango de probabilidades que tiene la respuesta
tau<-seq(0.1,0.9,by=0.05)

AER<-NULL
recall<-NULL
precision<-NULL
F1<-NULL
exito<-"1"
frac<-"0"

for (i in 1:length(tau)){
  glm.pred <- rep("0", length(testres))
  glm.pred[pr7b > tau[i]] <- "1"
  tab<-table(glm.pred, testres)
  if (!frac %in% rownames(tab)){
    tab<-rbind(tab,c(0,0))
    rownames(tab)[2]<-frac
  } 
  if (!exito %in% rownames(tab)){
    tab<-rbind(tab,c(0,0))
    rownames(tab)[2]<-exito
  }
  AER[i]<-(tab[exito,frac]+tab[frac,exito])/sum(tab)
  precision[i]<-(tab[exito,exito])/sum(tab[exito,])
  recall[i]<-(tab[exito,exito])/sum(tab[,exito])
  F1[i]<-2*precision[i]*recall[i]/(precision[i]+recall[i])
}

cbind(tau,AER,precision,recall,F1)

