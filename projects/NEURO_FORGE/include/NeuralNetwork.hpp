#pragma once

#include "Matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

// Stores the actual calculations from a neural-network forward pass.
struct PredictionTrace
{
    // Nine input feature values.
    std::vector<double> inputs;

    // Hidden layer calculations: 8 neurons.
    std::vector<double> hiddenWeightedSums;
    std::vector<double> hiddenActivations;

    // Actual input-to-hidden weights: 8 rows x 9 columns.
    std::vector<std::vector<double>> inputHiddenWeights;

    // One bias for each hidden neuron.
    std::vector<double> hiddenBiases;

    // Eight weights connecting the hidden layer to the output.
    std::vector<double> hiddenOutputWeights;

    // Output neuron bias.
    double outputBias = 0.0;

    // Output layer calculation.
    double outputWeightedSum = 0.0;
    double prediction = 0.0;
};

class NeuralNetwork
{
private:
    Matrix weightsInputHidden_;
    Matrix weightsHiddenOutput_;

    Matrix biasHidden_;
    Matrix biasOutput_;

    double learningRate_;

public:
    NeuralNetwork(
        std::size_t inputSize,
        std::size_t hiddenSize,
        double learningRate = 0.01
    );

    // Standard prediction.
    double predict(const Matrix& input);

    // Prediction plus actual internal calculations and parameters.
    PredictionTrace predictWithTrace(const Matrix& input);

    // Train using one sample.
    void train(
        const Matrix& input,
        double target,
        double classWeight = 1.0
    );

    // Calculate squared-error loss.
    double loss(double prediction, double target);

    // Save trained network parameters.
    bool saveModel(const std::string& filename) const;

    // Load trained parameters into a matching network.
    bool loadModel(const std::string& filename);
};