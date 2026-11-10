#include "Matrix.hpp"

#include <iostream>
#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, 0.0)
{
}

double& Matrix::operator()(std::size_t row, std::size_t col)
{
    return data_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const
{
    return data_[row * cols_ + col];
}

std::size_t Matrix::rows() const
{
    return rows_;
}

std::size_t Matrix::cols() const
{
    return cols_;
}

Matrix Matrix::multiply(const Matrix& other) const
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    Matrix result(rows_, other.cols_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < other.cols_; ++j)
        {
            for (std::size_t k = 0; k < cols_; ++k)
            {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }

    return result;
}

Matrix Matrix::add(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }

    return result;
}

Matrix Matrix::subtract(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }

    return result;
}

void Matrix::print() const
{
    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            std::cout << (*this)(i, j) << " ";
        }

        std::cout << '\n';
    }
}