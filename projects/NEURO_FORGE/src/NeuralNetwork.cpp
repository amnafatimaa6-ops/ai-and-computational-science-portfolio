#include "NeuralNetwork.hpp"
#include "Activation.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

NeuralNetwork::NeuralNetwork(
    std::size_t inputSize,
    std::size_t hiddenSize,
    double learningRate
)
    : weightsInputHidden_(hiddenSize, inputSize),
      weightsHiddenOutput_(1, hiddenSize),
      biasHidden_(hiddenSize, 1),
      biasOutput_(1, 1),
      learningRate_(learningRate)
{
    if (inputSize == 0 || hiddenSize == 0)
        throw std::invalid_argument("Network dimensions must be greater than zero.");

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (std::size_t i = 0; i < hiddenSize; ++i)
    {
        for (std::size_t j = 0; j < inputSize; ++j)
            weightsInputHidden_(i, j) = distribution(generator);

        weightsHiddenOutput_(0, i) = distribution(generator);
    }
}

PredictionTrace NeuralNetwork::predictWithTrace(const Matrix& input)
{
    if (input.cols() != 1 ||
        input.rows() != weightsInputHidden_.cols())
        throw std::invalid_argument("Input dimensions do not match the network.");

    PredictionTrace trace;

    // Record actual inputs.
    for (std::size_t i = 0; i < input.rows(); ++i)
        trace.inputs.push_back(input(i, 0));

    // Export actual input-to-hidden weights.
    trace.inputHiddenWeights.resize(weightsInputHidden_.rows());

    for (std::size_t i = 0; i < weightsInputHidden_.rows(); ++i)
    {
        for (std::size_t j = 0; j < weightsInputHidden_.cols(); ++j)
            trace.inputHiddenWeights[i].push_back(weightsInputHidden_(i, j));
    }

    // Export actual hidden biases.
    for (std::size_t i = 0; i < biasHidden_.rows(); ++i)
        trace.hiddenBiases.push_back(biasHidden_(i, 0));

    // Hidden layer: z = W*x + b.
    Matrix hiddenRaw = weightsInputHidden_.multiply(input);
    hiddenRaw = hiddenRaw.add(biasHidden_);

    Matrix hidden = hiddenRaw;

    for (std::size_t i = 0; i < hidden.rows(); ++i)
    {
        const double z = hiddenRaw(i, 0);
        const double activation = sigmoid(z);

        trace.hiddenWeightedSums.push_back(z);
        trace.hiddenActivations.push_back(activation);

        hidden(i, 0) = activation;
    }

    // Export actual hidden-to-output weights.
    for (std::size_t i = 0; i < weightsHiddenOutput_.cols(); ++i)
        trace.hiddenOutputWeights.push_back(weightsHiddenOutput_(0, i));

    trace.outputBias = biasOutput_(0, 0);

    // Output layer: z = W*a + b.
    Matrix outputRaw = weightsHiddenOutput_.multiply(hidden);
    outputRaw = outputRaw.add(biasOutput_);

    trace.outputWeightedSum = outputRaw(0, 0);
    trace.prediction = sigmoid(trace.outputWeightedSum);

    return trace;
}

double NeuralNetwork::predict(const Matrix& input)
{
    return predictWithTrace(input).prediction;
}

void NeuralNetwork::train(
    const Matrix& input,
    double target,
    double classWeight
)
{
    if (input.cols() != 1 ||
        input.rows() != weightsInputHidden_.cols())
        throw std::invalid_argument("Input dimensions do not match the network.");

    if (target < 0.0 || target > 1.0 ||
        classWeight < 0.0 || !std::isfinite(classWeight))
        throw std::invalid_argument("Invalid target or class weight.");

    // Forward pass.
    Matrix hiddenRaw = weightsInputHidden_.multiply(input);
    hiddenRaw = hiddenRaw.add(biasHidden_);

    Matrix hidden = hiddenRaw;

    for (std::size_t i = 0; i < hidden.rows(); ++i)
        hidden(i, 0) = sigmoid(hidden(i, 0));

    Matrix outputRaw = weightsHiddenOutput_.multiply(hidden);
    outputRaw = outputRaw.add(biasOutput_);

    const double output = sigmoid(outputRaw(0, 0));
    const double outputError = (output - target) * classWeight;
    const double outputGradient =
        outputError * sigmoidDerivative(outputRaw(0, 0));

    // Use pre-update output weights for hidden gradients.
    Matrix oldOutputWeights = weightsHiddenOutput_;

    for (std::size_t i = 0; i < hidden.rows(); ++i)
    {
        const double gradient = outputGradient * hidden(i, 0);
        weightsHiddenOutput_(0, i) -= learningRate_ * gradient;
    }

    biasOutput_(0, 0) -= learningRate_ * outputGradient;

    for (std::size_t i = 0; i < hidden.rows(); ++i)
    {
        const double hiddenError =
            oldOutputWeights(0, i) * outputGradient;

        const double hiddenGradient =
            hiddenError * sigmoidDerivative(hiddenRaw(i, 0));

        for (std::size_t j = 0; j < input.rows(); ++j)
        {
            weightsInputHidden_(i, j) -=
                learningRate_ * hiddenGradient * input(j, 0);
        }

        biasHidden_(i, 0) -= learningRate_ * hiddenGradient;
    }
}

double NeuralNetwork::loss(double prediction, double target)
{
    const double error = prediction - target;
    return 0.5 * error * error;
}

bool NeuralNetwork::saveModel(const std::string& filename) const
{
    std::ofstream file(filename);

    if (!file.is_open())
        return false;

    file << std::setprecision(17);
    file << "NEUROFORGE_MODEL_V1\n";
    file << weightsInputHidden_.rows() << ' '
         << weightsInputHidden_.cols() << ' '
         << learningRate_ << '\n';

    for (std::size_t i = 0; i < weightsInputHidden_.rows(); ++i)
    {
        for (std::size_t j = 0; j < weightsInputHidden_.cols(); ++j)
            file << weightsInputHidden_(i, j) << ' ';

        file << '\n';
    }

    for (std::size_t i = 0; i < biasHidden_.rows(); ++i)
        file << biasHidden_(i, 0) << ' ';

    file << '\n';

    for (std::size_t i = 0; i < weightsHiddenOutput_.cols(); ++i)
        file << weightsHiddenOutput_(0, i) << ' ';

    file << '\n' << biasOutput_(0, 0) << '\n';

    return file.good();
}

bool NeuralNetwork::loadModel(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
        return false;

    std::string version;
    std::size_t hiddenSize = 0;
    std::size_t inputSize = 0;
    double savedLearningRate = 0.0;

    if (!(file >> version) || version != "NEUROFORGE_MODEL_V1")
        return false;

    if (!(file >> hiddenSize >> inputSize >> savedLearningRate))
        return false;

    if (hiddenSize != weightsInputHidden_.rows() ||
        inputSize != weightsInputHidden_.cols() ||
        !std::isfinite(savedLearningRate) ||
        savedLearningRate <= 0.0)
        return false;

    Matrix inputHidden(hiddenSize, inputSize);
    Matrix hiddenBias(hiddenSize, 1);
    Matrix hiddenOutput(1, hiddenSize);
    Matrix outputBias(1, 1);

    for (std::size_t i = 0; i < hiddenSize; ++i)
    {
        for (std::size_t j = 0; j < inputSize; ++j)
        {
            if (!(file >> inputHidden(i, j)) ||
                !std::isfinite(inputHidden(i, j)))
                return false;
        }
    }

    for (std::size_t i = 0; i < hiddenSize; ++i)
    {
        if (!(file >> hiddenBias(i, 0)) ||
            !std::isfinite(hiddenBias(i, 0)))
            return false;
    }

    for (std::size_t i = 0; i < hiddenSize; ++i)
    {
        if (!(file >> hiddenOutput(0, i)) ||
            !std::isfinite(hiddenOutput(0, i)))
            return false;
    }

    if (!(file >> outputBias(0, 0)) ||
        !std::isfinite(outputBias(0, 0)))
        return false;

    // Apply parameters only after all values have been validated.
    weightsInputHidden_ = inputHidden;
    biasHidden_ = hiddenBias;
    weightsHiddenOutput_ = hiddenOutput;
    biasOutput_ = outputBias;
    learningRate_ = savedLearningRate;

    return true;
}