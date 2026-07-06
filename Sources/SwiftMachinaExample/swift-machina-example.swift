// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import SwiftMachina
import MLX
import MLXNN
import MLXOptimizers
import TabularData

// Verify build without Metal: export MLX_METAL_ENABLED=0

@main
struct swiftmlx {

    static func loadData() throws -> (MLXArray, MLXArray) {
        let url = Bundle.module.url(forResource: "breast_cancer", withExtension: "csv")!
        let df = try DataFrame(contentsOfCSVFile: url)

        print("Rows:", df.rows.count)
        print("Columns:", df.columns.count)

        // Labels: M → 1.0, B → 0.0
        let labels: [Float32] = df["diagnosis"].map {
            ($0 as! String == "M") ? 1.0 : 0.0
        }
        let y = MLXArray(labels)

        // Features (drop id + diagnosis)
        let featureColumns =
            df.columns.filter { $0.name != "id" && $0.name != "diagnosis" }

        var features: [Float32] = []
        features.reserveCapacity(df.rows.count * featureColumns.count)

        for row in df.rows {
            for col in featureColumns {
                features.append(Float32(row[col.name] as! Double))
            }
        }

        let X = MLXArray(features).reshaped([df.rows.count, featureColumns.count])

        precondition(X.shape == [569, 30])
        precondition(y.shape == [569])

        // Return raw data — StandardScaler in Pipeline handles normalization
        return (X, y)
    }

    struct ModelResult {
        let name: String
        let cm: ConfusionMatrix.Result
        let trainTimeSec: Double
        let inferenceTimeSec: Double
    }

    static func main() throws {

        try Device.withDefaultDevice(.cpu) {

            print("Device:", Device.defaultDevice)

            // MARK: - Load data
            let (X, y): (MLXArray, MLXArray) = try loadData()

            print("X shape:", X.shape)
            print("y shape:", y.shape)

            // Ensure y has shape [N, 1]
            let y2 = y.reshaped([y.shape[0], 1])

            // MARK: - Train/Test split (80/20, stratified, seeded)
            let splitResult = trainTestSplit(
                X: X, y: y2,
                testSize: 0.2,
                randomState: 42,
                stratify: true
            )
            let Xtrain = splitResult.Xtrain
            let ytrain = splitResult.ytrain
            let Xtest  = splitResult.Xtest
            let ytest  = splitResult.ytest

            print("Train: \(Xtrain.shape[0]), Test: \(Xtest.shape[0])")

            // MARK: - SVM requires {-1, +1}
            let ytrainSVM = 2 * ytrain - 1
            let ytestSVM  = 2 * ytest - 1

            // MARK: - Models
            // Names match Python benchtest for compare-swift
            let models: [(String, any Model, MLXArray, MLXArray)] = [

                ("Logistic Regression",
                 LogisticRegression(inputSize: X.shape[1], epochs: 500, learningRate: 0.01),
                 ytrain, ytest),

                ("SVM (Linear)",
                 SVM(inputSize: X.shape[1], epochs: 500, learningRate: 0.01),
                 ytrainSVM, ytestSVM),

                ("KNN",
                 KNN(k: 5),
                 ytrain, ytest),

                ("Naive Bayes",
                 GaussianNaiveBayes(),
                 ytrain, ytest),

                ("LDA",
                 LDA(),
                 ytrain, ytest),

                ("QDA",
                 QDA(),
                 ytrain, ytest),

                ("Decision Tree",
                 DecisionTree(maxDepth: 5),
                 ytrain, ytest),

                ("Random Forest",
                 RandomForest(nTrees: 20, maxDepth: 5, randomState: 42),
                 ytrain, ytest),

                ("Extra Trees",
                 ExtraTrees(nTrees: 20, maxDepth: 5, randomState: 42),
                 ytrain, ytest),

                ("Gradient Boosting",
                 GradientBoosting(nEstimators: 20, learningRate: 0.1),
                 ytrain, ytest)
            ]

            // MARK: - Evaluate all models
            var results: [ModelResult] = []

            for (name, baseModel, ytr, yte) in models {

                print("\n--- \(name) ---")

                let model = baseModel

                var pipeline = Pipeline(steps: [
                    .transformer(StandardScaler()),
                    .model(model)
                ])

                // Train (timed)
                let trainStart = CFAbsoluteTimeGetCurrent()
                pipeline.fit(X: Xtrain, y: ytr)
                let trainTime = CFAbsoluteTimeGetCurrent() - trainStart

                // Predict (timed)
                let inferStart = CFAbsoluteTimeGetCurrent()
                let predsRaw = pipeline.predict(X: Xtest)
                let inferTime = CFAbsoluteTimeGetCurrent() - inferStart

                // Convert predictions for metrics (SVM → 0/1)
                let preds: MLXArray = (name == "SVM (Linear)")
                    ? `where`(predsRaw .> 0, MLXArray(1), MLXArray(0))
                    : predsRaw

                // Metrics ALWAYS use 0/1 labels
                let yEval: MLXArray = (name == "SVM (Linear)")
                    ? `where`(yte .> 0, MLXArray(1), MLXArray(0))
                    : yte

                let cm = ConfusionMatrix().compute(yEval, preds)

                results.append(ModelResult(
                    name: name,
                    cm: cm,
                    trainTimeSec: trainTime,
                    inferenceTimeSec: inferTime
                ))

                print("""
                  Accuracy:         \(String(format: "%.4f", cm.accuracy))
                  Precision:        \(String(format: "%.4f", cm.precision))
                  Recall:           \(String(format: "%.4f", cm.recall))
                  F1:               \(String(format: "%.4f", cm.f1))
                  BalancedAccuracy: \(String(format: "%.4f", cm.balancedAccuracy))
                  Specificity:      \(String(format: "%.4f", cm.specificity))
                  MCC:              \(String(format: "%.4f", cm.mcc))
                  TrainTime:        \(String(format: "%.4f", trainTime))s
                  InferenceTime:    \(String(format: "%.4f", inferTime))s
                  TP: \(cm.TP)  TN: \(cm.TN)  FP: \(cm.FP)  FN: \(cm.FN)
                """)
            }

            // MARK: - Write metrics.csv (compatible with Python compare-swift)
            let header = "Model,Status,Accuracy,Precision,Recall,F1,BalancedAccuracy,Specificity,MCC,AUC,TrainTimeSec,InferenceTimeSec,TN,FP,FN,TP"

            var csvLines = [header]
            for r in results {
                let line = [
                    r.name,
                    "ok",
                    String(format: "%.6f", r.cm.accuracy),
                    String(format: "%.6f", r.cm.precision),
                    String(format: "%.6f", r.cm.recall),
                    String(format: "%.6f", r.cm.f1),
                    String(format: "%.6f", r.cm.balancedAccuracy),
                    String(format: "%.6f", r.cm.specificity),
                    String(format: "%.6f", r.cm.mcc),
                    "",  // AUC — not yet implemented
                    String(format: "%.6f", r.trainTimeSec),
                    String(format: "%.6f", r.inferenceTimeSec),
                    String(r.cm.TN),
                    String(r.cm.FP),
                    String(r.cm.FN),
                    String(r.cm.TP),
                ].joined(separator: ",")
                csvLines.append(line)
            }

            let csvContent = csvLines.joined(separator: "\n") + "\n"

            let outputDir = FileManager.default.currentDirectoryPath
            let csvPath = (outputDir as NSString).appendingPathComponent("swift_metrics.csv")
            try csvContent.write(toFile: csvPath, atomically: true, encoding: .utf8)
            print("\n✅ Metrics written to: \(csvPath)")
            print("   Use: python main.py compare-swift --python-metrics <run>/metrics.csv --swift-metrics \(csvPath)")
        }
    }
}
