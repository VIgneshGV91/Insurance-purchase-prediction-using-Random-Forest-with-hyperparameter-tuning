library(kernlab)
library(e1071)
library(caret)
library(randomForest)

data(ticdata)
str(ticdata)
dim(ticdata)
summary(ticdata)
#Converting variables to factos
ticdata[,65:85] <- lapply(ticdata[,65:85], factor)
tic0 = ticdata[1:5822, ]
tic1 = ticdata[-(1:5822), ]
attach(tic0)

set.seed(4)
RF_ticdata = randomForest(CARAVAN ~ . , data = tic0, ntree = 500, mtry = sqrt(ncol(tic0)), proximity =
                            TRUE, replace = TRUE, sampsize = nrow(tic0), importance = TRUE )
print(RF_ticdata)
#Variable Importance plot
varImpPlot(RF_ticdata)

plot(RF_ticdata)
tic_pred = predict(RF_ticdata, newdata = tic1)
tic_pred = as.data.frame(tic_pred)
tic_test = as.data.frame(tic1[,86])
colnames(tic_pred)= c("Predicted")
colnames(tic_test)= c("Original")
accuracy_tic<-confusionMatrix(tic_test$Original, tic_pred$Predicted)
accuracy_tic
#Confusion Matrix and Statistics

#Overall Accuracy of the above model is 94%.

#1000 individuals who, based on the model, seem most likely to buy policies.
top_insurer= tic0[order(RF_ticdata$votes[1:1000,2]),]
rest_insurer= tic0[order(RF_ticdata$votes[-(1:1000),2]),]
View(top_insurer)
#For top insurers
table(top_insurer$CARAVAN)
#For rest
table(rest_insurer$CARAVAN)
#Success rate is 6.27% for top insurer while 6.19% for the rest. So, there is not much difference in
#both the sample in terms of insurers.
bestsize <- function(n0=696, mtry=9, nselect=1000, form=CARAVAN~., data=tic0[,-1])
{
  ticm.rf <- randomForest(form, sampsize=c(n0,348), mtry=mtry, data=data)
  nr <- (1:nrow(tic0))[order(ticm.rf$votes[,2], decreasing=T)[1:nselect]]
  sum(tic0[nr, 86] == "insurance")
}
#Keeping mtry at sqrt(ncol(tic0)) i.e. 9 and nselect =1000 and varying n0 value.
bestsize(n0 = 1000, mtry = 9, nselect = 1000) #152
bestsize(n0 = 900, mtry = 9, nselect = 1000) #155

bestsize(n0 = 800, mtry = 9, nselect = 1000) #157
bestsize(n0 = 700, mtry = 9, nselect = 1000) #153
bestsize(n0 = 600, mtry = 9, nselect = 1000) #160
bestsize(n0 = 500, mtry = 9, nselect = 1000) #157
bestsize(n0 = 450, mtry = 9, nselect = 1000) #155
bestsize(n0 = 400, mtry = 9, nselect = 1000) #160
bestsize(n0 = 350, mtry = 9, nselect = 1000) #156
bestsize(n0 = 300, mtry = 9, nselect = 1000) #164
#From above iterations, the best response that we received is for n0 = 300, which returned 164.
#Now keeping n0= 300, nselect =100 and varying mtry for optimization.
bestsize(n0 = 300, mtry = 9, nselect = 1000) #162
bestsize(n0 = 300, mtry = 6, nselect = 1000) #158
bestsize(n0 = 300, mtry = 12, nselect = 1000) #156
bestsize(n0 = 300, mtry = 15, nselect = 1000) #163
bestsize(n0 = 300, mtry = 13, nselect = 1000) #160
bestsize(n0 = 300, mtry = 11, nselect = 1000) #167
bestsize(n0 = 300, mtry = 8, nselect = 1000) #154
bestsize(n0 = 300, mtry = 10, nselect = 1000) #161
#Best value of 167 is attained at mtry= 11.

result<-0
for (i in 6:20){
  result[i-5] = bestsize(n0 = 300, mtry = i, nselect = 1000)
}
result
#[1] 156 160 157 165 161 159 161 166 154 160 159 160 156 154 158

#Best result is achieved on mtry = 13 and value is 166.

RF_ticdata_tuned <- tuneRF(tic0[,-86], CARAVAN, stepFactor=1.5, improve=1e-5, ntree=1000, plot = TRUE, samsize=1000)
#mtry = 9 OOB error = 6.05%
#Searching left ...
#mtry = 6 OOB error = 6.05%
#0 1e-05
#Searching right ...
#mtry = 13 OOB error = 6.01%
#0.005681818 1e-05
#mtry = 19 OOB error = 5.98%
#0.005714286 1e-05
#mtry = 28 OOB error = 5.94%
#0.005747126 1e-05
#mtry = 42 OOB error = 5.94%
#0 1e-05
RF_ticdata_tuned <- tuneRF(tic0[,-86], CARAVAN, stepFactor=1.5, improve=0.005, ntree=1000, plot = TRUE, samsize=2000)
#mtry = 9 OOB error = 6.03%
#Searching left ...
#mtry = 6 OOB error = 6.05%
#-0.002849003 0.005
#Searching right ...
#mtry = 13 OOB error = 6.01%

#0.002849003 0.005
RF_ticdata_tuned <- tuneRF(tic0[,-86], CARAVAN, stepFactor=1.5, improve=0.005, ntree=1000, plot = TRUE, samsize=3000)
#mtry = 9 OOB error = 6.05%
#Searching left ...
#mtry = 6 OOB error = 6.08%
#-0.005681818 0.005
#Searching right ...
#mtry = 13 OOB error = 5.98%
#0.01136364 0.005
#mtry = 19 OOB error = 5.99%
#-0.002873563 0.005
RF_ticdata_tuned <- tuneRF(tic0[,-86], CARAVAN, stepFactor=1.5, improve=0.005, ntree=1000, plot = TRUE, samsize=800)
#mtry = 9 OOB error = 6.03%
#Searching left ...
#mtry = 6 OOB error = 6.06%
#-0.005698006 0.005
#Searching right ...
#mtry = 13 OOB error = 5.98%
#0.008547009 0.005
#mtry = 19 OOB error = 5.98%
#0 0.005

RF_ticdata_tuned <- tuneRF(tic0[,-86], CARAVAN, stepFactor=1.2, improve=0.005, ntree=1000, plot = TRUE, samsize=500)
#mtry = 9 OOB error = 6.05%
#Searching left ...
#mtry = 8 OOB error = 6.03%
#0.002840909 0.005
#Searching right ...
#mtry = 10 OOB error = 6.01%
#0.005681818 0.005
#mtry = 12 OOB error = 5.98%
#0.005714286 0.005
#mtry = 14 OOB error = 5.99%
#-0.002873563 0.005
#For both optimal value of mtry = 13 which gave us the least OOB error. Reducing sample size tends to
#increase the accuracy upto a certain extend but it stops after a point and OOB starts increasing,
#which was 1000 in our case. For both the case greater accuracy was achieved to the right side
#i.e increasing the mtry value from 9.
levels(ticdata$STYPE)
levels(tic0$STYPE)
# There are seems to be 39 levels of variable STYPE. But Random Forrest seems to handle only 32 variables
# level at best, so we have reduced the levels by finding proximity, i.e. similar levels should be merged
# to reduce the total level count to less to or equal to 6.

subrow = sample(1:nrow(tic0), 1000)
tic9 = tic0[subrow, ]
tab9 = table(tic9[,86])
ssize = as.vector(rep(tab9[2], 2))
ticm9.rf = randomForest(tic9$CARAVAN ~ ., sampsize=ssize, mtry=13, data=tic9[,-1], proximity=TRUE)
pts = cmdscale(1-ticm9.rf$proximity)
#Now plot the points, identifying the levels of STYPE:
prox<-xyplot(pts[,2] ~ pts[,1], panel=lattice.getOption("panel.xyplot"))
prox

#Converting pts in data frame
pts<-as.data.frame(pts)
#Adding a new column to pts and assigning PTYPE according to indexes
pts$STYPE<-tic0$STYPE[match(rownames(pts),rownames(tic0))]
#Groupiing the variable STYPE in 6 groups according to V1 and V2 value approximately from
#xy plot
pts$STYPEG[(pts$V1<=-0.3) && (pts$V2<=0)]<- "A"
pts$STYPEG[(pts$V1>=0.2) & (pts$V2<=0)]<- "E"
pts$STYPEG[(pts$V1<=-0.3) & (pts$V2>0)]<- "B"
pts$STYPEG[(pts$V1>=0.2) & (pts$V2>0)]<- "F"
pts$STYPEG[(pts$V1<0.2 & pts$V1>-0.3) & (pts$V2<=0)]<- "D"
pts$STYPEG[(pts$V1<0.2 & pts$V1>-0.3) & (pts$V2>0)]<- "C"
#Checking distribution and overlap of both variable
table(pts$STYPE,pts$STYPEG)
table(pts$STYPEG)
#Creating a new dataframe test for backup and merging tic9 and pts data frame.
test<-data.frame(tic9, pts)
#Checking the consistancy of data frame.
test$STYPE==test$STYPE1
is.na(test)
str(test)
View(tic9)
#adding a new variable, STYPEG to tic0, tic9 and tic1 and assigning the group according to xy plot
#and overlap

tic9$STYPEG<-test$STYPEG
tic0$STYPEG<-as.factor(tic0$STYPEG)
tic0$STYPEG[tic0$STYPE=="Affluent senior apartments"]<-"C"
tic0$STYPEG[tic0$STYPE=="Affluent young families"]<-"C"
tic0$STYPEG[tic0$STYPE=="Dinki's (double income no kids)"]<-"C"
tic0$STYPEG[tic0$STYPE=="High Income, expensive child"]<-"C"
tic0$STYPEG[tic0$STYPE=="High status seniors"]<-"C"
tic0$STYPEG[tic0$STYPE=="Young seniors in the city"]<-"C"
tic0$STYPEG[tic0$STYPE=="Career and childcare"]<-"D"
tic0$STYPEG[tic0$STYPE=="Couples with teens 'Married with children'"]<-"D"
tic0$STYPEG[tic0$STYPE=="Family starters"]<-"D"
tic0$STYPEG[tic0$STYPE=="Low income catholics"]<-"D"
tic0$STYPEG[tic0$STYPE=="Middle class families"]<-"D"
tic0$STYPEG[tic0$STYPE=="Modern, complete families"]<-"D"
tic0$STYPEG[tic0$STYPE=="Very Important Provincials"]<-"D"
tic0$STYPEG[tic0$STYPE=="Young all american family"]<-"D"
tic0$STYPEG[tic0$STYPE=="Young and rising"]<-"D"
tic0$STYPEG[tic0$STYPE=="Village families"]<-"E"
tic0$STYPEG[tic0$STYPE=="Traditional families"]<-"E"
tic0$STYPEG[tic0$STYPE=="Mixed small town dwellers"]<-"E"
tic0$STYPEG[tic0$STYPE=="Mixed seniors"]<-"E"
tic0$STYPEG[tic0$STYPE=="Mixed rurals"]<-"E"
tic0$STYPEG[tic0$STYPE=="Mixed apartment dwellers"]<-"E"
tic0$STYPEG[tic0$STYPE=="Large religous families"]<-"E"
tic0$STYPEG[tic0$STYPE=="Large family, employed child"]<-"E"
tic0$STYPEG[tic0$STYPE=="Large family farms"]<-"E"

tic0$STYPEG[tic0$STYPE=="Stable family"]<-"E"
tic0$STYPEG[tic0$STYPE=="Etnically diverse"]<-"F"
tic0$STYPEG[tic0$STYPE=="Fresh masters in the city"]<-"F"
tic0$STYPEG[tic0$STYPE=="Young urban have-nots"]<-"F"
tic0$STYPEG[tic0$STYPE=="Suburban youth"]<-"F"
tic0$STYPEG[tic0$STYPE=="Single youth"]<-"F"
tic0$STYPEG[tic0$STYPE=="Seniors in apartments"]<-"F"
tic0$STYPEG[tic0$STYPE=="Senior cosmopolitans"]<-"F"
tic0$STYPEG[tic0$STYPE=="Residential elderly"]<-"F"
tic0$STYPEG[tic0$STYPE=="Religious elderly singles"]<-"F"
tic0$STYPEG[tic0$STYPE=="Porchless seniors: no front yard"]<-"F"
tic0$STYPEG[tic0$STYPE=="Own home elderly"]<-"F"
tic0$STYPEG[tic0$STYPE=="Lower class large families"]<-"B"
tic0$STYPEG[tic0$STYPE=="Students in apartments"]<-"B"
tic0$STYPEG[tic0$STYPE=="Young, low educated"]<-"B"
tic1$STYPEG<-"NULL"
tic1$STYPEG<-as.factor(tic1$STYPEG)
tic1$STYPEG[tic1$STYPE=="Affluent senior apartments"]<-"C"
tic1$STYPEG[tic1$STYPE=="Affluent young families"]<-"C"
tic1$STYPEG[tic1$STYPE=="Dinki's (double income no kids)"]<-"C"
tic1$STYPEG[tic1$STYPE=="High Income, expensive child"]<-"C"
tic1$STYPEG[tic1$STYPE=="High status seniors"]<-"C"
tic1$STYPEG[tic1$STYPE=="Young seniors in the city"]<-"C"
tic1$STYPEG[tic1$STYPE=="Career and childcare"]<-"D"
tic1$STYPEG[tic1$STYPE=="Couples with teens 'Married with children'"]<-"D"
tic1$STYPEG[tic1$STYPE=="Family starters"]<-"D"
tic1$STYPEG[tic1$STYPE=="Low income catholics"]<-"D"
tic1$STYPEG[tic1$STYPE=="Middle class families"]<-"D"

tic1$STYPEG[tic1$STYPE=="Modern, complete families"]<-"D"
tic1$STYPEG[tic1$STYPE=="Very Important Provincials"]<-"D"
tic1$STYPEG[tic1$STYPE=="Young all american family"]<-"D"
tic1$STYPEG[tic1$STYPE=="Young and rising"]<-"D"
tic1$STYPEG[tic1$STYPE=="Village families"]<-"E"
tic1$STYPEG[tic1$STYPE=="Traditional families"]<-"E"
tic1$STYPEG[tic1$STYPE=="Mixed small town dwellers"]<-"E"
tic1$STYPEG[tic1$STYPE=="Mixed seniors"]<-"E"
tic1$STYPEG[tic1$STYPE=="Mixed rurals"]<-"E"
tic1$STYPEG[tic1$STYPE=="Mixed apartment dwellers"]<-"E"
tic1$STYPEG[tic1$STYPE=="Large religous families"]<-"E"
tic1$STYPEG[tic1$STYPE=="Large family, employed child"]<-"E"
tic1$STYPEG[tic1$STYPE=="Large family farms"]<-"E"
tic1$STYPEG[tic1$STYPE=="Stable family"]<-"E"
tic1$STYPEG[tic1$STYPE=="Etnically diverse"]<-"F"
tic1$STYPEG[tic1$STYPE=="Fresh masters in the city"]<-"F"
tic1$STYPEG[tic1$STYPE=="Young urban have-nots"]<-"F"
tic1$STYPEG[tic1$STYPE=="Suburban youth"]<-"F"
tic1$STYPEG[tic1$STYPE=="Single youth"]<-"F"
tic1$STYPEG[tic1$STYPE=="Seniors in apartments"]<-"F"
tic1$STYPEG[tic1$STYPE=="Senior cosmopolitans"]<-"F"
tic1$STYPEG[tic1$STYPE=="Residential elderly"]<-"F"
tic1$STYPEG[tic1$STYPE=="Religious elderly singles"]<-"F"
tic1$STYPEG[tic1$STYPE=="Porchless seniors: no front yard"]<-"F"
tic1$STYPEG[tic1$STYPE=="Own home elderly"]<-"F"
tic1$STYPEG[tic1$STYPE=="Lower class large families"]<-"B"
tic1$STYPEG[tic1$STYPE=="Students in apartments"]<-"B"
tic1$STYPEG[tic1$STYPE=="Young, low educated"]<-"B"

#Making a new RF forest model on new grouped varibale and leaving out variable STYPE
set.seed(123)
subrow_opt = sample(1:nrow(tic0), 2000)
tic13 = tic0[subrow_opt, ]
tab13 = table(tic13[,86])
ssize_opt = as.vector(rep(tab13[2], 2))
RF_ticdata_opt = randomForest(tic0$CARAVAN ~ . , data = tic0[,-1], ntree = 1000, mtry = 13, proximity = TRUE, replace = TRUE, sampsize = ssize_opt, importance = TRUE )
print(RF_ticdata_opt)
#OOB estimate of error rate: 6.18%
#Confusion matrix:
#noinsurance insurance class.error
#noinsurance 5457 17 0.00310559
#insurance 343 5 0.98563218
#Accuracy remained approximately the same. There was no improvement in the model.
#The reaon can be that the response variable is higly skewed and there are very few instances
#with response variable "insurance"

#Testing the performance of optimized RF with the test data tic1
tic_pred_opt = predict(RF_ticdata_opt, newdata = tic1[,-1])
tic_pred_opt = as.data.frame(tic_pred_opt)
tic_test_opt = as.data.frame(tic1[,86])
colnames(tic_pred_opt)= c("Predicted")
colnames(tic_test_opt)= c("Original")

accuracy_tic_opt<-confusionMatrix(tic_test_opt$Original, tic_pred_opt$Predicted)
accuracy_tic_opt
#Result
#Confusion Matrix and Statistics
#Reference
#Prediction noinsurance insurance
#noinsurance 3756 6
#insurance 233 5
#Accuracy : 0.9402
#95% CI : (0.9325, 0.9474)
#No Information Rate : 0.9972
#1000 observations that are from individuals who are predicted as most likely to be insurance holders on
#optimized RF model
top_insurer_opt= tic0[order(RF_ticdata_opt$votes[1:1000,2]),]
rest_insurer_opt= tic0[order(RF_ticdata_opt$votes[-(1:1000),2]),]
#59 of those are holding isurance out of 1000 observations.
table(top_insurer_opt$CARAVAN)
#noinsurance insurance
#941 59
table(rest_insurer_opt$CARAVAN)
#noinsurance insurance
#4541 281
subrow = sample(1:nrow(tic0), 1000)

tic9 = tic0[subrow, ]
tab9 = table(tic9[,86])
ssize = as.vector(rep(tab9[2], 2))
ticm9.rf = randomForest(CARAVAN ~ ., sampsize=ssize, mtry=13, data=tic9[,-1], proximity=TRUE)
pts = cmdscale(1-ticm9.rf$proximity)
#Now plot the points, identifying the levels of STYPE:
xyplot(pts[,2] ~ pts[,1], panel=panel.text(pts[,2],pts[,1],labels=tic0$STYPE[subscripts]))
prox<-xyplot(pts[,2] ~ pts[,1], panel=lattice.getOption("panel.xyplot"))
prox
pts<-as.data.frame(pts)
pts$STYPE<-tic0$STYPE[match(rownames(pts),rownames(tic0))]
View(pts)
pts$STYPEG[(pts$V1<=-0.3) && (pts$V2<=0)]<- "A"
pts$STYPEG[(pts$V1>=0.2) & (pts$V2<=0)]<- "E"
pts$STYPEG[(pts$V1<=-0.3) & (pts$V2>0)]<- "B"
pts$STYPEG[(pts$V1>=0.2) & (pts$V2>0)]<- "F"
pts$STYPEG[(pts$V1<0.2 & pts$V1>-0.3) & (pts$V2<=0)]<- "D"
pts$STYPEG[(pts$V1<0.2 & pts$V1>-0.3) & (pts$V2>0)]<- "C"