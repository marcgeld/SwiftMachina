//
//  BinaryCrossEntropy.swift
//  swiftmlx
//
//  Created by Marcus Gelderman on 2026-04-28.
//

import MLX

// MARK: - Binary Cross Entropy
// Provides two variants:
//
// 1. binaryCrossEntropyWithLogits (RECOMMENDED for training)
//    - Input: raw logits (no sigmoid applied)
//    - Numerically stable
//
// 2. binaryCrossEntropy
//    - Input: probabilities (after sigmoid)
//    - Less stable, mainly for analysis/debug

public struct BinaryCrossEntropy {

    public init() {}

    // MARK: - BCE with logits (stable, recommended)
    // Computes:
    //  max(x, 0) - x*y + log(1 + exp(-|x|))
    //
    // where:
    //  x = logits
    //  y = target (0 or 1)
    public func withLogits(
        logits: MLXArray,
        target: MLXArray
    ) -> MLXArray {

        precondition(
            logits.shape == target.shape,
            "logits and target must have same shape"
        )

        let loss =
            maximum(logits, 0)
            - logits * target
            + log(1 + exp(-abs(logits)))

        return loss.mean()
    }

    // MARK: - BCE with probabilities (less stable)
    // Computes:
    //  -[y log(p) + (1-y) log(1-p)]
    //
    // where:
    //  p = sigmoid(logits)
    public func withProbabilities(
        probs: MLXArray,
        target: MLXArray
    ) -> MLXArray {

        precondition(
            probs.shape == target.shape,
            "probs and target must have same shape"
        )

        let eps: Float = 1e-7

        let p = clip(probs, min: eps, max: 1 - eps)

        let loss =
            -(target * log(p) + (1 - target) * log(1 - p))

        return loss.mean()
    }
}
