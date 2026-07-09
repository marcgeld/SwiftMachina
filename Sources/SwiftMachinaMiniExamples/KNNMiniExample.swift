import Foundation
import SwiftMachina

// K-nearest neighbors (k = 1) fitted-state round trip; persists the
// memorized training samples.
enum KNNMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "KNN",
            model: try KNN(k: 1),
            outputDirectory: outputDirectory
        )
    }
}
