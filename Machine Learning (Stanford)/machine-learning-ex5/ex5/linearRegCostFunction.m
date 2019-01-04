function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate hypothesis
h = X * theta;

% Compute the squared errors for each elements.
% h is hypothesis vector, y is training set result values vector, do matrix substraction and then element wise square
squaredErrors = (h - y) .^ 2;

% Theta 0 (first element) is not regularized, matlab uses indexes starting at one, so put this together into thetaNoZero vector.
thetaNoZero = [ 0; theta(2:end) ];

% Put the whole cost function together.
J = (1 / (2 * m)) * sum(squaredErrors) + (lambda / (2 * m)) * sum(thetaNoZero .^ 2);

% =========================================================================

grad = grad(:);

end
