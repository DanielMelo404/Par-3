###############################Inversa generalizada
###################################################################

A<-matrix(c(0,2,0,1,9,3,4,8,5),nrow=3,byrow=T)
solve(A)

y<-c(2,8,4)

solve(A,y)

solve(A)%*%y

B<-matrix(c(0,2,0,1,9,3),nrow=2,byrow=T)
solve(B)

library(MASS)
invB<-ginv(B)

all.equal(B%*%invB%*%B,B) #Cond.1
all.equal(invB%*%B%*%invB,invB) #Cond. 2
all.equal(B%*%invB,t(B%*%invB)) #Simetría de cond. 3
all.equal(B%*%invB,(B%*%invB)%*%(B%*%invB)) #idempotencia de cond. 3
all.equal(invB%*%B,t(invB%*%B)) #Simetría de cond. 4
all.equal(invB%*%B,(invB%*%B)%*%(invB%*%B)) #idempotencia de cond. 4

y2<-c(2,8)
invB%*%y2

C<-matrix(c(1,1,-1,1,1,-1,-1,-1),nrow=4,byrow=T)

y3<-c(1,1,1,1)

ginv(C)%*%y3

