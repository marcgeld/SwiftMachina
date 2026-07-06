//
//  LogisticRegression.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX
import MLXNN
import MLXOptimizers

// MARK: - Logistic Regression (Binary)
public struct LogisticRegression: Estimator, Predictor {

    // MARK: - Properties
    private var linear: Linear
    public let inputSize: Int
    public let epochs: Int
    public let learningRate: Float

    // MARK: - Init
    public init(
        inputSize: Int,
        epochs: Int = 500,
        learningRate: Float = 0.01
    ) throws {
        try require(inputSize > 0, .invalidParameter("inputSize must be greater than zero"))
        try require(epochs >= 0, .invalidParameter("epochs must be non-negative"))
        try require(learningRate > 0, .invalidParameter("learningRate must be greater than zero"))
        self.inputSize = inputSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.linear = Linear(inputDimensions: inputSize, outputDimensions: 1)
    }

    // MARK: - Forward (logits, ej sigmoid!)
    private func forward(_ X: MLXArray) -> MLXArray {
        linear(X)
    }

    // MARK: - Fit (protocol conformance)
    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try fit(X: X, y: y, verbose: false)
    }

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray, verbose: Bool) throws {

        try require(X.shape.count == 2 && X.shape[1] == inputSize,
                    .invalidShape("X must have shape [N, inputSize]"))
        try require(y.shape.count == 2 && y.shape[1] == 1,
                    .invalidShape("y must have shape [N, 1]"))
        try require(y.shape[0] == X.shape[0],
                    .invalidShape("X and y must have same number of rows"))

        let optimizer = SGD(learningRate: learningRate)

        let lg = valueAndGrad(model: linear) { model, x, y -> MLXArray in
            let logits = model(x)
            let loss =
                maximum(logits, 0)
                - logits * y
                + log(1 + exp(-abs(logits)))
            return loss.mean()
        }

        for epoch in 0..<epochs {
            let (loss, grads) = lg(linear, X, y)
            optimizer.update(model: linear, gradients: grads)
            MLX.eval(linear)

            if verbose && epoch % 100 == 0 {
                print("Epoch \(epoch), loss: \(loss.item(Float.self))")
            }
        }
    }

    // MARK: - Predict probabilities
    public func predictProba(X: MLXArray) throws -> MLXArray {
        try require(X.shape.count == 2 && X.shape[1] == inputSize,
                    .invalidShape("X must have shape [N, inputSize]"))
        return sigmoid(forward(X))
    }

    // MARK: - Predict classes (0/1)
    public func predict(X: MLXArray) throws -> MLXArray {
        let probs = try predictProba(X: X)
        return probs .> 0.5
    }

    // MARK: - Weights access (debug / analys)
    public func weights() -> (W: MLXArray, b: MLXArray?) {
        (linear.weight, linear.bias)
    }
}
