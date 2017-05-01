# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting linear regression
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fitting polynomial regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
poly_reg = lm(formula = Salary ~.,
             data = dataset)

# Visualising results
# install.packages('ggplot2')
library(ggplot2)

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or bluff (linear regression)') +
  xlab('Level') +
  ylab('Salary')