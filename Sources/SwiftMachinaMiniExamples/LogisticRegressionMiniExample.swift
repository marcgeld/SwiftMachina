import Foundation
import SwiftMachina

// Binary logistic regression (MLXNN + SGD) fitted-state round trip.
enum LogisticRegressionMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "LogisticRegression",
            model: try LogisticRegression(inputSize: 2, epochs: 120, learningRate: 0.1),
            outputDirectory: outputDirectory
        )
    }
}
