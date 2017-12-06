function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


constant = 1/m * (-1);
thetaTransX = theta' * X';
X_sigmoid = sigmoid(thetaTransX);
X_sigmoid = X_sigmoid';

J = y' * log(X_sigmoid) + (1-y)' * log(1-X_sigmoid);
J = constant * J;
J = J + constant * (-1/2) * lambda * sum(theta(2:size(theta),:) .^ 2);


grad = ((X_sigmoid - y)' * X)';
grad = (-1) * constant * grad;
grad = grad + (lambda *(-1) * constant) * [0;theta(2:size(theta),:)];


% =============================================================

end
