x<-c(1,2,2.5,3,5,7,10)
y<-rep(0,length(x))

hist(x,prob=TRUE,ylim=c(0,0.25))
points(x,y,col="red",pch=12)
lines(density(x,kernel="rectangular",bw=0.7),lwd=2,col="blue") #bw es igual a b/2 

hist(x,prob=TRUE)
points(x,y,col="red",pch=12)
lines(density(x,kernel="rectangular",bw=3),lwd=2,col="blue") #bw es igual a b/2 

hist(x,prob=TRUE)
points(x,y,col="red",pch=12)
lines(density(x,kernel="rectangular"),lwd=2,col="blue") #bw es igual a b/2 


hist(x,prob=TRUE,ylim=c(0,0.25))
points(x,y,col="red",pch=12)
lines(density(x,bw=1.2),lwd=2,col="blue") #bw es igual a b/2

hist(x,prob=TRUE)
points(x,y,col="red",pch=12)
lines(density(x),lwd=2,col="blue") #bw es igual a b/2 



