#pragma once

#include <string>
#include <vector>

struct DataSample
{
    std::vector<double> features;
    double label;
};

class Dataset
{
private:
    std::vector<DataSample> samples_;

public:
    bool loadCSV(
        const std::string& filename,
        int labelColumn
    );

    void normalize();

    void trainTestSplit(
        double trainRatio,
        Dataset& trainSet,
        Dataset& testSet
    ) const;

    const std::vector<DataSample>& samples() const;

    std::size_t size() const;

    std::size_t featureCount() const;
};