//
//  NumericaBridge.swift
//  SwiftMachina
//

import Foundation
import MLX
import SwiftNumerica

// MARK: - SwiftNumerica Bridge
//
// Small dense matrices (features × features) are converted to Double and
// handed to SwiftNumerica's LAPACK-backed linear algebra. MLX runs these
// ops on the CPU stream anyway, and Double precision is strictly better
// for covariance matrices.

func numericaMatrix(_ array: MLXArray) -> Matrix? {
    guard array.shape.count == 2 else { return nil }
    let values = array.asArray(Float.self).map(Double.init)
    return Matrix(values: values, rows: array.shape[0], columns: array.shape[1])
}

func mlxArray(_ matrix: Matrix) -> MLXArray {
    MLXArray(matrix.values.map(Float.init), [matrix.rowCount, matrix.columnCount])
}

// MARK: - Inverse

/// Inverts a square matrix in Double precision.
/// Returns nil for singular input so callers can fall back to MLX.
func numericaInverse(_ array: MLXArray) -> MLXArray? {
    guard let matrix = numericaMatrix(array),
          let inverse = matrix.inverse() else {
        return nil
    }
    return mlxArray(inverse)
}

// MARK: - Log-determinant

/// log(det(A)) for a symmetric positive-definite matrix via the sum of
/// log-eigenvalues, which stays stable where a direct determinant would
/// under- or overflow.
///
/// The matrix is symmetrized first: Float32 covariance products can be
/// asymmetric in the last bits, which SwiftNumerica's symmetric
/// eigensolver rejects.
func numericaLogDeterminant(symmetric array: MLXArray) -> Float? {
    guard let matrix = numericaMatrix(array), matrix.isSquare else {
        return nil
    }

    let n = matrix.rowCount
    var values = matrix.values
    for row in 0..<n {
        for column in (row + 1)..<n {
            let mean = (values[row * n + column] + values[column * n + row]) / 2
            values[row * n + column] = mean
            values[column * n + row] = mean
        }
    }

    guard let symmetrized = Matrix(values: values, rows: n, columns: n),
          let eigenvalues = symmetrized.eigenvalues() else {
        return nil
    }

    var logDet = 0.0
    for eigenvalue in eigenvalues {
        guard eigenvalue > 0 else { return nil }
        logDet += log(eigenvalue)
    }
    return Float(logDet)
}
