import Foundation
import SwiftMachina

// CART decision tree fitted-state round trip; persists the explicit tree
// structure as shared TreeNodeState JSON.
enum DecisionTreeMiniExample {
    static func run(outputDirectory: URL) throws {
        try runMiniExample(
            name: "DecisionTree",
            model: try DecisionTree(maxDepth: 2),
            outputDirectory: outputDirectory
        )
    }
}
