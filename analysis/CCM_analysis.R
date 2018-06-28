#install.packages("qualtRics")
#library(qualtRics)
library(foreign)
# skewness and kurtosis
library(e1071)
# outlier detection
library(mvoutlier)
library(ggpubr)
library(dplyr)
library(lsmeans)
library(multcompView)

# read in a file
rawData <- read.csv(file='/home/lisette/Radboud/CCM Language and Web Interaction/sad/results/cleaned_results.csv', header=TRUE, sep=",")
#rawData <- read.spss(file='CCM_raw.sav')
head(rawData)

# plot some stuff
plot(rawData$Gender, xlab='Gender', ylab='Amount of participants', main='Gender distribution of participants')
hist(rawData$Age, xlab='Age', ylab='Amount of participants', main='Age distribution of participants')
hist(rawData$Duration..in.seconds., xlab='Duration', ylab='Amount of participants', main='Survey duration distribution of participants')
# re-order the interest variable for plotting
rawData$v3<-factor(rawData$Interest_politics, c("Very low", "Low", "Medium", "High", "Very High"))
plot(rawData$v3, xlab='Interest', ylab='Amount of participants', main="Distribution of participants' interest in American politics")

# plot the scores
hist(rawData$SC0, breaks=seq(0,42,by=2), xlab='Score', main='Score distribution of participants')

# correlation between duration and score
cor.test(rawData$Duration..in.seconds., rawData$SC0, method = c("pearson"))

# correlation between age and score
cor.test(rawData$Age, rawData$SC0, method = c("pearson"))

groups <- list('t','c')
methods <-list('real', 'lstm','markov')
#mapping <- setNames(methods,list(1,2,2))

for(i in groups){
  for(j in methods){
    cond_name <- paste(i,j,sep="")
    print(cond_name)
    # use a regex for matching the condition
    cols <- grep(cond_name, names(rawData), value=T)
    
    # force to integer
    # rawData[cols] <- lapply(rawData[cols], as.integer) 
    
    # get the corresponding columns
    allInd <- rawData[, cols]
    # because R's maps are apparently clumsy, hard-code stuff
    if(j == 'real'){
      answer = 1
    }
    else{
      answer = 2
    }
    # compute the attribution rate
    cond_name_att <- paste(cond_name,'_att',sep="")
    rawData[,cond_name_att] <- (rowSums(allInd == 1) / length(cols))
    
    # compute the error rate for this combination
    cond_name_err <- paste(cond_name,'_err',sep="")
    rawData[,cond_name_err] <- 1 - (rowSums(allInd == answer) / length(cols))
    #print(allInd)
  }
}
response_columns_err = c('treal_err','creal_err', 'tlstm_err', 'clstm_err', 'tmarkov_err','cmarkov_err')
response_columns_att = c('treal_att','creal_att', 'tlstm_att', 'clstm_att', 'tmarkov_att','cmarkov_att')

# detect outliers in response patterns
#outliers <- aq.plot(rawData[response_columns_err])
# print
#outliers

#means <- sapply(rawData,mean)

# Stack the data
rawData_s <- stack(rawData, select=response_columns_att)
names(rawData_s)[names(rawData_s) == 'values'] <- 'attributionRate'


# determines the set type based on the cond_name
account_type <- function(x){
  first <- substr(x,0,1)
  if(first =='c'){
    return('CNN')
  }
  else{
    return('Trump')
  }
}

method_type <- function(x){
  if(grepl('markov',x)){
    return('Markov')
  }
  else if(grepl('lstm',x)){
    return('LSTM')
  }
  else{
    return('Real')
  }
}
# extract DV Method and Dataset from ind column
rawData_s$account <- as.factor(sapply(rawData_s$ind,FUN = function(x) account_type(x)))
rawData_s$method <- as.factor(sapply(rawData_s$ind,FUN = function(x) method_type(x)))

# Box plot with multiple groups
# +++++++++++++++++++++
# Plot accuracy by group ("method")
# Color box plot by a second group: "grpoup"
ggboxplot(rawData_s, x = "method", y = "attributionRate", color = "account",
          palette = c("#00AFBB", "#E7B800"), xlab="Method used")

# line plot to visualize possible interaction effects
ggline(rawData_s, x = "method", y = "attributionRate", color = "account",
       add = c("mean_se", "dotplot"),
       palette = c("#00AFBB", "#E7B800"), xlab="Method used")


allTrump <- subset(rawData_s, rawData_s$account == 'Trump')

allCNN <- subset(rawData_s, rawData_s$account == 'CNN')



anova_res <- aov(attributionRate ~ method + account
                 + method:account, data = rawData_s)

summary(anova_res)

# look at the data by group
model.tables(anova_res, type="means", se = TRUE)

# Tukey to identify which differences between methods are significant
TukeyHSD(anova_res, which = "method")

# analyze interaction effects
marginal = lsmeans(anova_res, pairwise ~ account:method, adjust="tukey")

# decide how to plot this stuff (look at a critical difference plot)

cld(marginal, alpha=0.05, Letters=letters, adjust="tukey")

## verify test assumptions

# 1. Homogeneity of variances
plot(anova_res, 1)

# 2. Normality
plot(anova_res, 2)



# Q-Q Plot for variable scores
attach(mtcars)
qqnorm(rawData$treal)
qqline(rawData$treal)

#rawData[, c("AF3","F7","P8","O1","O2","MARKER")]


