#Data is from 90s
#Adding tidyverse library 
library(tidyverse)
library(reshape2)

#adding housing data as DataFrame
housing = read.csv('CA_housing.csv')

head(housing)
summary(housing)


#Houses in data are in the groups that are in the "communities" 
#NA's in total_bedrooms need to be addressed. These must be given a value
#We will split the ocean_proximity into binary columns.
#Most machine learning algorithms in R can handle categoricals in a single column, but we will cater to the lowest common denominator and do the splitting.
#Make the total_bedrooms and total_rooms into a mean_number_bedrooms and mean_number_rooms columns as there are likely more accurate depections of the houses in a given group.

par(mfrow=c(2,5))

colnames(housing)

#Generating some initial visualization for the community 
ggplot(data = melt(housing), mapping = aes(x = value)) + 
  geom_histogram(bins = 30) + facet_wrap(~variable, scales = 'free_x')

# to make it more fun we could x10 the housing values, but I would leave it as is for now. 
#Substitute median values with missing values
housing$total_bedrooms[is.na(housing$total_bedrooms)] = median(housing$total_bedrooms , na.rm = TRUE)


#Visualize on a map
install.packages('plotly')
library(plotly)
df <- housing
# geo styling
g <- list(
  scope = 'usa',
  projection = list(type = 'albers usa'),
  showland = TRUE,
  landcolor = toRGB("gray95"),
  subunitcolor = toRGB("gray85"),
  countrycolor = toRGB("gray85"),
  countrywidth = 0.5,
  subunitwidth = 0.5
)

fig <- plot_geo(df, lat = ~latitude, lon = ~longitude)
fig <- fig %>% add_markers(
  text = ~paste(total_bedrooms, population, housing_median_age, sep = "<br />"),
  color = ~median_house_value, symbol = I("square"), size = I(8), hoverinfo = "text"
)
fig <- fig %>% colorbar(title = "housing")
fig <- fig %>% layout(
  title = 'Ca housing)', geo = g
)

fig

#Fix the total columns - make them means
housing$mean_bedrooms = housing$total_bedrooms/housing$households
housing$mean_rooms = housing$total_rooms/housing$households

drops = c('total_bedrooms', 'total_rooms')

housing = housing[ , !(names(housing) %in% drops)]

#Check the results
head(housing)
summary(housing)

#Cleanup the booleans 
#Turn categoricals into booleans
#Below I do the following:
  
#Get a list of all the categories in the 'ocean_proximity' column
#Make a new empty dataframe of all 0s, where each category is its own colum
#Use a for loop to populate the appropriate columns of the dataframe
#Drop the original column from the dataframe.

categories = unique(housing$ocean_proximity)
#split the categories off
cat_housing = data.frame(ocean_proximity = housing$ocean_proximity)

for(cat in categories){
  cat_housing[,cat] = rep(0, times= nrow(cat_housing))
}
head(cat_housing) #see the new columns on the right

for(i in 1:length(cat_housing$ocean_proximity)){
  cat = as.character(cat_housing$ocean_proximity[i])
  cat_housing[,cat][i] = 1
}

head(cat_housing)

cat_columns = names(cat_housing)
keep_columns = cat_columns[cat_columns != 'ocean_proximity']
cat_housing = select(cat_housing,one_of(keep_columns))

tail(cat_housing)

#Lattice experiment___________________________________________________
#Training and creating the model option 1 
# Multiple variables Linear regression
#Adding libraries
library(lattice)
library(ggplot2)
library(caret)
#get data
colnames(housing)
dat = housing[,c("longitude","latitude","housing_median_age","median_income","population","median_house_value")]
# train a regression model
model = train(median_house_value ~., dat, method ="lm" )
View(model)
model$finalModel
summary(model)

#Measuring the errors of estimation
#Step3 predict mpg | plug in and measure an accuracy. 
predictive_median_house_value = predict(model, dat)
as.numeric(predictive_median_house_value[1:5])
dat$median_house_value[1:5]

#step4: measuring an accuracy
#MAE
mean(abs(predictive_median_house_value-dat$median_house_value))
#RMSE
RMSE(predictive_median_house_value,dat$median_house_value)
#R^2
R2(predictive_median_house_value, dat$median_house_value)

#SVM , RF tidyverse experiment__________________________________________________________________________________
#Scale the numerical variables
#Note here I scale every one of the numericals except for
#'median_house_value' as this is what we will be working to predict. 
#The x values are scaled so that coefficients in things like support vector machines are given equal weight, 
#but the y value scale doesn't affect the learning algorithms in the same way (and we would just need to re-scale the predictions at the end which is another hassle).

colnames(housing)
drops = c('ocean_proximity','median_house_value')
housing_num =  housing[ , !(names(housing) %in% drops)]
head(housing_num)
scaled_housing_num = scale(housing_num)
head(scaled_housing_num)
#Merge the altered numerical and categorical dataframes
cleaned_housing = cbind(cat_housing, scaled_housing_num, median_house_value=housing$median_house_value)
head(cleaned_housing)
#We pull this subsection from the main dataframe and put it to the side to not be 
#looked at prior to testing our models. Don't look at it, as snooping the test data introduces a bias to your work!
#This is the data we use to validate our model, when we train a machine learning algorithm the goal is usually to 
#make an algorithm that predicts well on data it hasn't seen before. To assess this feature, we pull a set of 
#data to validate the models as accurate/inaccurate once we have completed the training process.

set.seed(1738) # Set a random seed so that same sample can be reproduced in future runs
sample = sample.int(n = nrow(cleaned_housing), size = floor(.8*nrow(cleaned_housing)), replace = F)
train = cleaned_housing[sample, ] #just the samples
test  = cleaned_housing[-sample, ] #everything but the samples

head(train)
nrow(train) + nrow(test) == nrow(cleaned_housing)

#Adding the ML training library
library('boot')

#glm  generalized linear models 
?cv.glm # note the K option for K fold cross validation
glm_house = glm(median_house_value~median_income+mean_rooms+population, data=cleaned_housing)
k_fold_cv_error = cv.glm(cleaned_housing , glm_house, K=5)
k_fold_cv_error$delta #This is the meaningless shit

glm_cv_rmse = sqrt(k_fold_cv_error$delta)[1]
glm_cv_rmse #off by about $83,000... it is a start
summary(glm_house)
glm_house$importance

#Trying Random Forest model
install.packages('randomForest')
library('randomForest')
?randomForest #Explains what the hell is RF 
names(train)
set.seed(1738)

train_y = train[,'median_house_value']
train_x = train[, names(train) !='median_house_value']

head(train_y)
head(train_x)

#some people like weird r format like this... I find it causes headaches
#rf_model = randomForest(median_house_value~. , data = train, ntree =500, importance = TRUE)
rf_model = randomForest(train_x, y = train_y , ntree = 500, importance = TRUE)
names(rf_model) #these are all the different things you can call from the model.
rf_model$importance
