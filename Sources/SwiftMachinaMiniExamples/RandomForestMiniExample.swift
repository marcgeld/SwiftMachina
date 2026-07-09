import Foundation
import SwiftMachina

// Random forest fitted-state round trip; persists every member tree.
enum RandomForestMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "RandomForest",
            model: try RandomForest(nTrees: 5, maxDepth: 2, randomState: 7),
            outputDirectory: outputDirectory
        )
    }
}
