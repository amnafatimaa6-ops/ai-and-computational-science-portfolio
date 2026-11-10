#pragma once

#include <vector>
#include <cstddef>

class Matrix
{
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

public:
    Matrix(std::size_t rows, std::size_t cols);

    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    std::size_t rows() const;
    std::size_t cols() const;

    Matrix multiply(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;

    void print() const;
};