import Foundation
import SwiftMachina

// Extremely randomized trees fitted-state round trip.
enum ExtraTreesMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "ExtraTrees",
            model: try ExtraTrees(nTrees: 5, maxDepth: 2, randomState: 7),
            outputDirectory: outputDirectory
        )
    }
}
