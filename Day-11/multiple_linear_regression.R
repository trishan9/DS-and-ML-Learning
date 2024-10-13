dataset = read.csv('50_Startups.csv')

dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = Profit ~ .,
               data = training_set)

y_hat = predict(regressor, newdata = test_set)


library(ggplot2)
ggplot() + 
  geom_density(aes(x = test_set$Profit), color = "red") +
  geom_density(aes(x = y_hat), color = "blue") +
  ggtitle("Actual (Red) v/s Fitted (Blue) Values (Test Set)") + 
  xlab("Value") + 
  ylab("Density")