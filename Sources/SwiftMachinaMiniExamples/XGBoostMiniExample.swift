import Foundation
import SwiftMachina

// XGBoost-style regularized second-order boosting fitted-state round trip;
// persists the boosted trees, base score, and hyperparameters.
enum XGBoostMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "XGBoostClassifier",
            model: try XGBoostClassifier(nEstimators: 8, learningRate: 0.3, maxDepth: 2, randomState: 7),
            outputDirectory: outputDirectory
        )
    }
}
