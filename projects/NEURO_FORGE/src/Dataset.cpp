#include "Dataset.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

bool Dataset::loadCSV(
    const std::string& filename,
    int labelColumn
)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        return false;
    }

    samples_.clear();

    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream stream(line);

        std::vector<double> values;
        std::string cell;

        while (std::getline(stream, cell, ','))
        {
            try
            {
                values.push_back(
                    std::stod(cell)
                );
            }
            catch (...)
            {
                values.clear();
                break;
            }
        }

        if (values.empty())
        {
            continue;
        }

        if (
            labelColumn < 0 ||
            labelColumn >=
                static_cast<int>(values.size())
        )
        {
            return false;
        }

        DataSample sample;

        sample.label =
            values[labelColumn];

        for (std::size_t i = 0;
             i < values.size();
             ++i)
        {
            if (
                static_cast<int>(i)
                != labelColumn
            )
            {
                sample.features.push_back(
                    values[i]
                );
            }
        }

        samples_.push_back(sample);
    }

    return !samples_.empty();
}

void Dataset::normalize()
{
    if (samples_.empty())
    {
        return;
    }

    std::size_t featureCount =
        samples_[0].features.size();

    for (std::size_t j = 0;
         j < featureCount;
         ++j)
    {
        double minValue =
            samples_[0].features[j];

        double maxValue =
            samples_[0].features[j];

        for (const auto& sample : samples_)
        {
            minValue =
                std::min(
                    minValue,
                    sample.features[j]
                );

            maxValue =
                std::max(
                    maxValue,
                    sample.features[j]
                );
        }

        double range =
            maxValue - minValue;

        if (range == 0.0)
        {
            continue;
        }

        for (auto& sample : samples_)
        {
            sample.features[j] =
                (sample.features[j] - minValue)
                / range;
        }
    }
}

void Dataset::trainTestSplit(
    double trainRatio,
    Dataset& trainSet,
    Dataset& testSet
) const
{
    if (
        trainRatio <= 0.0 ||
        trainRatio >= 1.0
    )
    {
        throw std::invalid_argument(
            "Train ratio must be between 0 and 1."
        );
    }

    std::vector<DataSample> shuffled =
        samples_;

    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(
        shuffled.begin(),
        shuffled.end(),
        generator
    );

    std::size_t trainSize =
        static_cast<std::size_t>(
            shuffled.size() * trainRatio
        );

    trainSet.samples_.assign(
        shuffled.begin(),
        shuffled.begin() + trainSize
    );

    testSet.samples_.assign(
        shuffled.begin() + trainSize,
        shuffled.end()
    );
}

const std::vector<DataSample>&
Dataset::samples() const
{
    return samples_;
}

std::size_t Dataset::size() const
{
    return samples_.size();
}

std::size_t Dataset::featureCount() const
{
    if (samples_.empty())
    {
        return 0;
    }

    return samples_[0].features.size();
}