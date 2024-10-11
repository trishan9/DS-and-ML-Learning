dataset = read.csv("Salary_Data.csv")

library(caTools)
split = sample.split(dataset$Salary, SplitRatio = 0.3)
training_set = subset(dataset, split==FALSE)
test_set = subset(dataset, split==TRUE)

regressor = lm(formula = Salary ~ YearsExperience, data=training_set)
regressor
# summary(regressor)

y_hat = predict(regressor, newdata = test_set)
y_hat

# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary), colour="blue") +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)), colour="red") +
  ggtitle('Years of Experience v/s Salary (Training Set)') + 
  xlab('Years of Experience') + 
  ylab('Salary')

library(ggplot2)
ggplot() + 
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary), colour="blue") +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)), colour="red") +
  ggtitle('Years of Experience v/s Salary (Test Set)') + 
  xlab('Years of Experience') + 
  ylab('Salary')