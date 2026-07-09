import Foundation
import SwiftMachina

// Gaussian naive Bayes fitted-state round trip; persists per-class means,
// variances, and priors.
enum GaussianNaiveBayesMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "GaussianNaiveBayes",
            model: GaussianNaiveBayes(),
            outputDirectory: outputDirectory
        )
    }
}
