import Foundation
import SwiftMachina

// Quadratic discriminant analysis fitted-state round trip; persists
// per-class means and covariance matrices.
enum QDAMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "QDA",
            model: try QDA(regParam: 0.1),
            outputDirectory: outputDirectory
        )
    }
}
