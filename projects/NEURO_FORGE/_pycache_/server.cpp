#include "NeuralNetwork.hpp"
#include "Dataset.hpp"
#include "Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr int INPUT_COUNT = 9;
    constexpr int HIDDEN_COUNT = 8;

    const std::string MODEL_PATH = "data/neuroforge.model";
    const std::string DATASET_PATH = "data/exoplanets_clean.csv";

    const std::vector<std::string> FEATURE_NAMES = {
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "pl_eqt",
        "pl_rade",
        "st_teff",
        "st_mass",
        "st_rad",
        "st_met"
    };

    struct ModelContext
    {
        Dataset referenceData;
        std::vector<double> minimum;
        std::vector<double> maximum;
        NeuralNetwork network;

        ModelContext()
            : network(INPUT_COUNT, HIDDEN_COUNT, 0.01)
        {
            if (!referenceData.loadCSV(
                    DATASET_PATH, INPUT_COUNT))
            {
                throw std::runtime_error(
                    "Could not load " + DATASET_PATH);
            }

            if (referenceData.size() == 0 ||
                referenceData.featureCount() != INPUT_COUNT)
            {
                throw std::runtime_error(
                    "Reference dataset must have 9 features.");
            }

            minimum.assign(
                INPUT_COUNT,
                std::numeric_limits<double>::max());

            maximum.assign(
                INPUT_COUNT,
                std::numeric_limits<double>::lowest());

            for (const auto& sample : referenceData.samples())
            {
                if (sample.features.size() != INPUT_COUNT)
                {
                    throw std::runtime_error(
                        "Invalid feature count in reference dataset.");
                }

                for (int j = 0; j < INPUT_COUNT; ++j)
                {
                    minimum[j] = std::min(
                        minimum[j], sample.features[j]);

                    maximum[j] = std::max(
                        maximum[j], sample.features[j]);
                }
            }

            if (!network.loadModel(MODEL_PATH))
            {
                throw std::runtime_error(
                    "Could not load " + MODEL_PATH);
            }
        }

        std::vector<double> normalize(
            const std::vector<double>& raw) const
        {
            if (raw.size() != INPUT_COUNT)
            {
                throw std::invalid_argument(
                    "Exactly 9 features are required.");
            }

            std::vector<double> result(INPUT_COUNT);

            for (int j = 0; j < INPUT_COUNT; ++j)
            {
                const double range =
                    maximum[j] - minimum[j];

                result[j] = range > 0.0
                    ? (raw[j] - minimum[j]) / range
                    : 0.0;
            }

            return result;
        }

        Matrix makeInput(
            const std::vector<double>& normalized) const
        {
            Matrix input(INPUT_COUNT, 1);

            for (int j = 0; j < INPUT_COUNT; ++j)
            {
                input(j, 0) = normalized[j];
            }

            return input;
        }
    };

    std::vector<double> parseValues(
        const std::string& line,
        char delimiter)
    {
        std::stringstream stream(line);
        std::vector<double> values;
        std::string cell;

        while (std::getline(stream, cell, delimiter))
        {
            std::stringstream parser(cell);
            double value;
            char extra;

            if (!(parser >> value) ||
                (parser >> extra) ||
                !std::isfinite(value))
            {
                throw std::invalid_argument(
                    "Invalid numeric feature: " + cell);
            }

            values.push_back(value);
        }

        if (values.size() != INPUT_COUNT)
        {
            throw std::invalid_argument(
                "Each row must contain exactly 9 features.");
        }

        return values;
    }

    void printNumber(double value)
    {
        std::cout << std::setprecision(15) << value;
    }

    void printArray(const std::vector<double>& values)
    {
        std::cout << "[";

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (i > 0)
                std::cout << ",";

            printNumber(values[i]);
        }

        std::cout << "]";
    }

    void printStringArray(
        const std::vector<std::string>& values)
    {
        std::cout << "[";

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (i > 0)
                std::cout << ",";

            std::cout << "\"" << values[i] << "\"";
        }

        std::cout << "]";
    }

    void runBatch(ModelContext& context)
    {
        std::string line;

        while (std::getline(std::cin, line))
        {
            if (line.empty())
                continue;

            const auto raw = parseValues(line, ',');
            const auto normalized = context.normalize(raw);
            const Matrix input = context.makeInput(normalized);

            const double score = context.network.predict(input);
            const int predictedClass = score >= 0.5 ? 1 : 0;

            std::cout << "{\"prediction_score\":";
            printNumber(score);

            std::cout << ",\"predicted_class\":"
                      << predictedClass << "}\n";
        }
    }

    void runSingle(
        ModelContext& context,
        int argc,
        char* argv[])
    {
        if (argc != INPUT_COUNT + 1)
        {
            throw std::invalid_argument(
                "Usage: neuroforge_bridge.exe "
                "<pl_orbper> <pl_orbsmax> <pl_orbeccen> "
                "<pl_eqt> <pl_rade> <st_teff> <st_mass> "
                "<st_rad> <st_met>\n"
                "Batch mode: neuroforge_bridge.exe --batch");
        }

        std::vector<double> raw(INPUT_COUNT);

        for (int i = 0; i < INPUT_COUNT; ++i)
        {
            std::stringstream parser(argv[i + 1]);
            char extra;

            if (!(parser >> raw[i]) ||
                (parser >> extra) ||
                !std::isfinite(raw[i]))
            {
                throw std::invalid_argument(
                    "Invalid feature argument: " +
                    std::string(argv[i + 1]));
            }
        }

        const auto normalized = context.normalize(raw);
        const Matrix input = context.makeInput(normalized);
        const auto trace = context.network.predictWithTrace(input);

        const double score = trace.prediction;
        const int predictedClass = score >= 0.5 ? 1 : 0;

        std::cout << std::setprecision(15) << "{";

        std::cout << "\"status\":\"success\",";
        std::cout << "\"feature_names\":";
        printStringArray(FEATURE_NAMES);

        std::cout << ",\"raw_inputs\":";
        printArray(raw);

        std::cout << ",\"normalized_inputs\":";
        printArray(normalized);

        std::cout << ",\"input_hidden_weights\":[";
        for (int h = 0; h < HIDDEN_COUNT; ++h)
        {
            if (h > 0)
                std::cout << ",";

            std::cout << "[";

            for (int j = 0; j < INPUT_COUNT; ++j)
            {
                if (j > 0)
                    std::cout << ",";

                printNumber(
                    trace.inputHiddenWeights[h][j]);
            }

            std::cout << "]";
        }
        std::cout << "]";

        std::cout << ",\"hidden_biases\":";
        printArray(trace.hiddenBiases);

        std::cout << ",\"hidden_weighted_sums\":";
        printArray(trace.hiddenWeightedSums);

        std::cout << ",\"hidden_activations\":";
        printArray(trace.hiddenActivations);

        std::cout << ",\"hidden_output_weights\":";
        printArray(trace.hiddenOutputWeights);

        std::cout << ",\"output_bias\":";
        printNumber(trace.outputBias);

        std::cout << ",\"output_weighted_sum\":";
        printNumber(trace.outputWeightedSum);

        std::cout << ",\"prediction_score\":";
        printNumber(score);

        std::cout << ",\"predicted_class\":"
                  << predictedClass;

        std::cout << ",\"predicted_label\":\""
                  << (predictedClass == 1
                      ? "Transit-method class"
                      : "Non-Transit-method class")
                  << "\"}\n";
    }
}

int main(int argc, char* argv[])
{
    try
    {
        ModelContext context;

        // A two-argument invocation is batch mode.
        // Read nine comma-separated values from each input line.
        if (argc == 2)
        {
            runBatch(context);
            return 0;
        }

        // Otherwise, preserve the existing single-prediction mode.
        runSingle(context, argc, argv);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "NeuroForge error: "
                  << error.what() << '\n';
        return 1;
    }
}
