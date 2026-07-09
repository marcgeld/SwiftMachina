import Foundation
import SwiftMachina

// Binary log-loss gradient boosting fitted-state round trip; persists the
// boosted regression trees and the initial log-odds.
enum GradientBoostingMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "GradientBoosting",
            model: try GradientBoosting(nEstimators: 8, learningRate: 0.3, maxDepth: 2),
            outputDirectory: outputDirectory
        )
    }
}
