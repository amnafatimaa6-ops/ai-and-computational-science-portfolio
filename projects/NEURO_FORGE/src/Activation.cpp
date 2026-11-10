#include "Activation.hpp"

#include <cmath>

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double relu(double x)
{
    return x > 0.0 ? x : 0.0;
}

double reluDerivative(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}