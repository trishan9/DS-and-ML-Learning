dataset = read.csv("Position_Salaries.csv")

linear_regressor = lm(formula = Salary ~ Level, data = dataset)
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="darkgreen") +
  geom_line(aes(x=dataset$Level, y=predict(linear_regressor, newdata = dataset)), color="red") +
  ggtitle("Linear Regression Model for Level v/s Salary") + 
  xlab("Level") +
  ylab("Salary")
y_hat_linear = predict(linear_regressor, newdata=data.frame(Level = 7.5))


polynomial_regressor = lm(formula = Salary ~ poly(Level, 3), data = dataset)
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color="blue") +
  geom_line(aes(x=dataset$Level, y=predict(polynomial_regressor, newdata = dataset)), color="red") +
  ggtitle("Polynomial Regression Model for Level v/s Salary") + 
  xlab("Level") +
  ylab("Salary")
y_hat_poly = predict(polynomial_regressor, newdata=data.frame(Level = 7.5))