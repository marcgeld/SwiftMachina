//
//  ConfusionMatrix.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Confusion Matrix (Binary)
// Computes:
//  TP (True Positive)
//  TN (True Negative)
//  FP (False Positive)
//  FN (False Negative)
//
// Supports labels:
//  - {0, 1}
//  - {-1, +1}
//
// Assumes yTrue and yPred have same shape [N, 1] or [N]

public struct ConfusionMatrix {

    public init() {}

    // MARK: - Result type
    public struct Result {
        public let TP: Int
        public let TN: Int
        public let FP: Int
        public let FN: Int

        // Optional helpers
        public var accuracy: Float {
            let total = TP + TN + FP + FN
            return total > 0 ? Float(TP + TN) / Float(total) : 0
        }

        public var precision: Float {
            let denom = TP + FP
            return denom > 0 ? Float(TP) / Float(denom) : 0
        }

        public var recall: Float {
            let denom = TP + FN
            return denom > 0 ? Float(TP) / Float(denom) : 0
        }

        public var f1: Float {
            let p = precision
            let r = recall
            let denom = p + r
            return denom > 0 ? 2 * (p * r) / denom : 0
        }

        public var specificity: Float {
            let denom = TN + FP
            return denom > 0 ? Float(TN) / Float(denom) : 0
        }

        public var balancedAccuracy: Float {
            (recall + specificity) / 2.0
        }

        public var mcc: Float {
            let tp = Double(TP), tn = Double(TN)
            let fp = Double(FP), fn = Double(FN)
            let num = tp * tn - fp * fn
            let denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).squareRoot()
            return denom > 0 ? Float(num / denom) : 0
        }
    }

    // MARK: - Compute
    public func compute(_ yTrue: MLXArray, _ yPred: MLXArray) throws -> Result {

        try require(
            yTrue.shape == yPred.shape,
            .invalidShape("yTrue and yPred must have the same shape")
        )

        // Normalize labels to {0,1}
        let yT = normalize(yTrue)
        let yP = normalize(yPred)

        let tp = ((yP .== 1) * (yT .== 1)).sum().item(Int.self)
        let tn = ((yP .== 0) * (yT .== 0)).sum().item(Int.self)
        let fp = ((yP .== 1) * (yT .== 0)).sum().item(Int.self)
        let fn = ((yP .== 0) * (yT .== 1)).sum().item(Int.self)

        return Result(TP: tp, TN: tn, FP: fp, FN: fn)
    }

    // MARK: - Normalize labels
    // Converts:
    //  {-1, +1} → {0, 1}
    //  {0, 1}   → {0, 1}
    private func normalize(_ y: MLXArray) -> MLXArray {
        return `where`(y .> 0, MLXArray(1), MLXArray(0))
    }
}
