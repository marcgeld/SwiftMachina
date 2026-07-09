import Foundation
import SwiftMachina

// Linear discriminant analysis fitted-state round trip; persists class
// means, priors, and the shared inverse covariance.
enum LDAMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "LDA",
            model: LDA(),
            outputDirectory: outputDirectory
        )
    }
}
