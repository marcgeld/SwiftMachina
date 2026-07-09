//
//  SVM.swift
//  SwiftMachina
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX
import MLXNN
import MLXOptimizers

// MARK: - Linear SVM (Binary Classification)
// This implementation uses a linear model trained with hinge loss.
// Labels are expected to be in {-1, +1}, not {0, 1}.
//
// The model learns a decision boundary of the form:
//      f(x) = XW + b
//
// Prediction is based on the sign of f(x):
//      f(x) > 0  → class +1
//      f(x) <= 0 → class -1

public struct SVM: Estimator, Predictor {

    // MARK: - Model parameters
    private var linear: Linear
    private var isFitted: Bool = false

    // Training hyperparameters
    public let inputSize: Int
    public let epochs: Int
    public let learningRate: Float
    public let lambda: Float

    // MARK: - Initializer
    public init(
        inputSize: Int,
        epochs: Int = 500,
        learningRate: Float = 0.01,
        lambda: Float = 0.01
    ) throws {
        try require(inputSize > 0, .invalidParameter("inputSize must be greater than zero"))
        try require(epochs >= 0, .invalidParameter("epochs must be non-negative"))
        try require(learningRate > 0, .invalidParameter("learningRate must be greater than zero"))
        try require(lambda >= 0, .invalidParameter("lambda must be non-negative"))
        self.inputSize = inputSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.lambda = lambda
        self.linear = Linear(inputDimensions: inputSize, outputDimensions: 1)
    }

    // MARK: - Forward pass (returns logits)
    private func forward(_ X: MLXArray) -> MLXArray {
        linear(X)
    }

    // MARK: - Fit (protocol conformance)
    public mutating func fit(X: MLXArray, y: MLXArray) throws {
        try fit(X: X, y: y, verbose: false)
    }

    // MARK: - Training (fit)
    public mutating func fit(
        X: MLXArray,
        y: MLXArray,
        verbose: Bool
    ) throws {
        try require(
            X.shape.count == 2 && X.shape[1] == inputSize,
            .invalidShape("X must have shape [N, inputSize]")
        )
        try requireLabelVector(y, rows: X.shape[0])
        let yTrain = y.shape.count == 1 ? y.reshaped([X.shape[0], 1]) : y

        let optimizer = SGD(learningRate: learningRate)
        let lambda = self.lambda

        let lg = valueAndGrad(model: linear) { model, x, y -> MLXArray in
            let logits = model(x)
            let hinge = maximum(0, 1 - y * logits)
            let dataLoss = hinge.mean()
            let regLoss = lambda * (model.weight * model.weight).sum()
            return dataLoss + regLoss
        }

        for epoch in 0..<epochs {
            let (loss, grads) = lg(linear, X, yTrain)
            optimizer.update(model: linear, gradients: grads)
            MLX.eval(linear)

            if verbose && epoch % 100 == 0 {
                print("Epoch \(epoch), loss: \(loss.item(Float.self))")
            }
        }

        isFitted = true
    }

    // MARK: - Prediction
    public func predict(X: MLXArray) throws -> MLXArray {
        try require(isFitted, .notFitted("SVM must be fitted before prediction"))
        try require(
            X.shape.count == 2 && X.shape[1] == inputSize,
            .invalidShape("X must have shape [N, inputSize]")
        )
        let logits = forward(X)
        return `where`(logits .> 0,
                       MLXArray(1),
                       MLXArray(-1))
    }

    // MARK: - Decision function
    public func decisionFunction(X: MLXArray) throws -> MLXArray {
        try require(isFitted, .notFitted("SVM must be fitted before prediction"))
        try require(
            X.shape.count == 2 && X.shape[1] == inputSize,
            .invalidShape("X must have shape [N, inputSize]")
        )
        return forward(X)
    }

    // MARK: - Access learned parameters
    public func weights() -> (W: MLXArray, b: MLXArray?) {
        (linear.weight, linear.bias)
    }
}

extension SVM: FittedStatePersistable {
    public struct FittedState: Codable {
        public let schemaVersion: Int
        public let modelType: String
        public let inputSize: Int
        public let epochs: Int
        public let learningRate: Float
        public let lambda: Float
        public let weight: SwiftMachinaArray
        public let bias: SwiftMachinaArray?
    }

    public func fittedState() throws -> FittedState {
        try require(isFitted, .notFitted("SVM must be fitted before saving"))
        return FittedState(
            schemaVersion: fittedStateSchemaVersion,
            modelType: "SVM",
            inputSize: inputSize,
            epochs: epochs,
            learningRate: learningRate,
            lambda: lambda,
            weight: SwiftMachinaArray(linear.weight),
            bias: linear.bias.map(SwiftMachinaArray.init)
        )
    }

    public init(fittedState: FittedState) throws {
        try requireFittedState(
            schemaVersion: fittedState.schemaVersion,
            modelType: fittedState.modelType,
            expectedModelType: "SVM"
        )
        try require(fittedState.inputSize > 0, .invalidParameter("inputSize must be greater than zero"))
        try require(fittedState.epochs >= 0, .invalidParameter("epochs must be non-negative"))
        try require(fittedState.learningRate > 0, .invalidParameter("learningRate must be greater than zero"))
        try require(fittedState.lambda >= 0, .invalidParameter("lambda must be non-negative"))

        let weight = try fittedState.weight.mlxArray()
        try require(
            weight.shape == [1, fittedState.inputSize],
            .invalidShape("SVM weight must have shape [1, inputSize]")
        )
        let bias = try fittedState.bias?.mlxArray()
        if let bias {
            try require(bias.shape == [1], .invalidShape("SVM bias must have shape [1]"))
        }

        self.inputSize = fittedState.inputSize
        self.epochs = fittedState.epochs
        self.learningRate = fittedState.learningRate
        self.lambda = fittedState.lambda
        self.linear = Linear(weight: weight, bias: bias)
        self.isFitted = true
    }
}
