import Foundation
import SwiftMachina

// Linear SVM (hinge loss, {-1, +1} labels) fitted-state round trip.
enum SVMMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "SVM",
            model: try SVM(inputSize: 2, epochs: 100, learningRate: 0.1),
            signedLabels: true,
            outputDirectory: outputDirectory
        )
    }
}
