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
    ) {
        precondition(inputSize > 0, "inputSize must be greater than zero")
        precondition(epochs >= 0, "epochs must be non-negative")
        precondition(learningRate > 0, "learningRate must be greater than zero")
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
    public mutating func fit(X: MLXArray, y: MLXArray) {
        fit(X: X, y: y, verbose: false)
    }

    // MARK: - Fit
    public mutating func fit(X: MLXArray, y: MLXArray, verbose: Bool) {

        precondition(X.shape.count == 2 && X.shape[1] == inputSize,
                     "X must have shape [N, inputSize]")
        precondition(y.shape.count == 2 && y.shape[1] == 1,
                     "y must have shape [N, 1]")
        precondition(y.shape[0] == X.shape[0],
                     "X and y must have same number of rows")

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
    public func predictProba(X: MLXArray) -> MLXArray {
        precondition(X.shape.count == 2 && X.shape[1] == inputSize,
                     "X must have shape [N, inputSize]")
        return sigmoid(forward(X))
    }

    // MARK: - Predict classes (0/1)
    public func predict(X: MLXArray) -> MLXArray {
        let probs = predictProba(X: X)
        return probs .> 0.5
    }

    // MARK: - Weights access (debug / analys)
    public func weights() -> (W: MLXArray, b: MLXArray?) {
        (linear.weight, linear.bias)
    }
}
