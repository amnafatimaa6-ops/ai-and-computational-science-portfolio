
#include "Preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    double missingValue()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    bool isMissing(double value)
    {
        return std::isnan(value);
    }

    std::vector<std::string> parseCSVLine(
        const std::string& line
    )
    {
        std::vector<std::string> fields;

        std::string field;
        bool insideQuotes = false;

        for (std::size_t i = 0; i < line.size(); ++i)
        {
            char c = line[i];

            if (c == '"')
            {
                if (
                    insideQuotes &&
                    i + 1 < line.size() &&
                    line[i + 1] == '"'
                )
                {
                    field += '"';
                    ++i;
                }
                else
                {
                    insideQuotes = !insideQuotes;
                }
            }
            else if (c == ',' && !insideQuotes)
            {
                fields.push_back(field);
                field.clear();
            }
            else
            {
                field += c;
            }
        }

        fields.push_back(field);

        return fields;
    }

    double parseNumber(
        const std::string& value
    )
    {
        if (value.empty())
        {
            return missingValue();
        }

        try
        {
            std::size_t position = 0;

            double number =
                std::stod(value, &position);

            if (position != value.size())
            {
                return missingValue();
            }

            return number;
        }
        catch (...)
        {
            return missingValue();
        }
    }

    double calculateMedian(
        std::vector<double> values
    )
    {
        values.erase(
            std::remove_if(
                values.begin(),
                values.end(),
                [](double value)
                {
                    return isMissing(value);
                }
            ),
            values.end()
        );

        if (values.empty())
        {
            return 0.0;
        }

        std::sort(
            values.begin(),
            values.end()
        );

        std::size_t middle =
            values.size() / 2;

        if (values.size() % 2 == 0)
        {
            return (
                values[middle - 1] +
                values[middle]
            ) / 2.0;
        }

        return values[middle];
    }
}

bool Preprocessor::createCleanDataset(
    const std::string& inputFile,
    const std::string& outputFile
)
{
    std::ifstream input(inputFile);

    if (!input.is_open())
    {
        std::cerr
            << "ERROR: Could not open NASA dataset: "
            << inputFile
            << '\n';

        return false;
    }

    std::string headerLine;

    if (!std::getline(input, headerLine))
    {
        std::cerr
            << "ERROR: NASA dataset is empty.\n";

        return false;
    }

    std::vector<std::string> headers =
        parseCSVLine(headerLine);

    std::unordered_map<std::string, std::size_t>
        columnIndex;

    for (std::size_t i = 0;
         i < headers.size();
         ++i)
    {
        columnIndex[headers[i]] = i;
    }

    // Scientific features used by NeuroForge.
    const std::vector<std::string> featureNames =
    {
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

    for (const auto& feature : featureNames)
    {
        if (columnIndex.find(feature)
            == columnIndex.end())
        {
            std::cerr
                << "ERROR: Required column missing: "
                << feature
                << '\n';

            return false;
        }
    }

    if (
        columnIndex.find("discoverymethod")
        == columnIndex.end()
    )
    {
        std::cerr
            << "ERROR: discoverymethod column missing.\n";

        return false;
    }

    const std::size_t discoveryColumn =
        columnIndex["discoverymethod"];

    std::vector<
        std::vector<double>
    > rows;

    std::vector<
        std::vector<double>
    > observedValues(
        featureNames.size()
    );

    std::string line;

    std::size_t totalRows = 0;
    std::size_t skippedRows = 0;

    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        ++totalRows;

        std::vector<std::string> fields =
            parseCSVLine(line);

        if (fields.size() <= discoveryColumn)
        {
            ++skippedRows;
            continue;
        }

        std::string discoveryMethod =
            fields[discoveryColumn];

        // Rows without a discovery method
        // cannot be assigned a meaningful class.
        if (discoveryMethod.empty())
        {
            ++skippedRows;
            continue;
        }

        std::vector<double> row;

        bool validRow = true;

        for (const auto& feature : featureNames)
        {
            std::size_t index =
                columnIndex[feature];

            double value = missingValue();

            if (index < fields.size())
            {
                value =
                    parseNumber(fields[index]);
            }

            row.push_back(value);

            if (!isMissing(value))
            {
                observedValues[
                    row.size() - 1
                ].push_back(value);
            }
        }

        if (!validRow)
        {
            ++skippedRows;
            continue;
        }

        // Binary target:
        // Transit = 1
        // Every other discovery method = 0
        double label =
            discoveryMethod == "Transit"
                ? 1.0
                : 0.0;

        row.push_back(label);

        rows.push_back(row);
    }

    input.close();

    if (rows.empty())
    {
        std::cerr
            << "ERROR: No usable rows found.\n";

        return false;
    }

    // Calculate feature medians using only
    // actually observed values.
    std::vector<double> medians;

    for (const auto& values : observedValues)
    {
        medians.push_back(
            calculateMedian(values)
        );
    }

    // Replace missing numerical values with
    // their corresponding feature median.
    for (auto& row : rows)
    {
        for (std::size_t j = 0;
             j < featureNames.size();
             ++j)
        {
            if (isMissing(row[j]))
            {
                row[j] = medians[j];
            }
        }
    }

    std::ofstream output(outputFile);

    if (!output.is_open())
    {
        std::cerr
            << "ERROR: Could not create clean dataset: "
            << outputFile
            << '\n';

        return false;
    }

    // Header
    for (std::size_t i = 0;
         i < featureNames.size();
         ++i)
    {
        output << featureNames[i] << ',';
    }

    output << "label\n";

    // Data
    for (const auto& row : rows)
    {
        for (std::size_t j = 0;
             j < row.size();
             ++j)
        {
            output << row[j];

            if (j + 1 < row.size())
            {
                output << ',';
            }
        }

        output << '\n';
    }

    output.close();

    std::size_t transitCount = 0;
    std::size_t otherCount = 0;

    for (const auto& row : rows)
    {
        if (row.back() == 1.0)
        {
            ++transitCount;
        }
        else
        {
            ++otherCount;
        }
    }

    std::cout
        << "NASA preprocessing complete.\n";

    std::cout
        << "Original rows read: "
        << totalRows
        << '\n';

    std::cout
        << "Usable rows: "
        << rows.size()
        << '\n';

    std::cout
        << "Skipped rows: "
        << skippedRows
        << '\n';

    std::cout
        << "Transit samples: "
        << transitCount
        << '\n';

    std::cout
        << "Non-Transit samples: "
        << otherCount
        << '\n';

    std::cout
        << "Clean dataset: "
        << outputFile
        << "\n\n";

    return true;
}

