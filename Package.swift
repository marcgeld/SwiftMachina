// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftMachina",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        .library(
            name: "SwiftMachina",
            targets: ["SwiftMachina"]
        ),
        .executable(
            name: "SwiftMachinaExample",
            targets: ["SwiftMachinaExample"]
        ),
        .executable(
            name: "SwiftMachinaBenchmarks",
            targets: ["SwiftMachinaBenchmarks"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/marcgeld/SwiftNumerica.git", from: "0.1.0")
    ],
    targets: [
        // MARK: - SwiftMachina (ML framework)
        .target(
            name: "SwiftMachina",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "SwiftNumerica", package: "SwiftNumerica"),
            ]
        ),

        // MARK: - Example
        .executableTarget(
            name: "SwiftMachinaExample",
            dependencies: [
                "SwiftMachina",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            exclude: [
                "README.md"
            ],
            resources: [
                .process("Resources")
            ]
        ),

        // MARK: - Benchmarks
        .executableTarget(
            name: "SwiftMachinaBenchmarks",
            dependencies: [
                "SwiftMachina",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "SwiftMachinaTests",
            dependencies: [
                "SwiftMachina"
            ],
            path: "Tests/SwiftMachinaTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
