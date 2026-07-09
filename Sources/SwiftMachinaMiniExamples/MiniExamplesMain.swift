import Foundation
import MLX
import SwiftMachina

// Runs every per-model mini example. Each <Model>MiniExample file trains a
// tiny classifier, saves its fitted state as JSON, reloads it, and verifies
// the loaded predictions. Adding an estimator means adding one file and one
// line below.

@main
struct SwiftMachinaMiniExamples {
    static func main() throws {
        try Device.withDefaultDevice(.cpu) {
            let outputDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("MiniModelBundles", isDirectory: true)

            try FileManager.default.createDirectory(
                at: outputDirectory,
                withIntermediateDirectories: true
            )

            try LogisticRegressionMiniExample.run(outputDirectory: outputDirectory)
            try SVMMiniExample.run(outputDirectory: outputDirectory)
            try KNNMiniExample.run(outputDirectory: outputDirectory)
            try GaussianNaiveBayesMiniExample.run(outputDirectory: outputDirectory)
            try LDAMiniExample.run(outputDirectory: outputDirectory)
            try QDAMiniExample.run(outputDirectory: outputDirectory)
            try DecisionTreeMiniExample.run(outputDirectory: outputDirectory)
            try RandomForestMiniExample.run(outputDirectory: outputDirectory)
            try ExtraTreesMiniExample.run(outputDirectory: outputDirectory)
            try GradientBoostingMiniExample.run(outputDirectory: outputDirectory)
            try XGBoostMiniExample.run(outputDirectory: outputDirectory)

            print("\nAll fitted-state model JSON files loaded and verified.")
        }
    }
}
