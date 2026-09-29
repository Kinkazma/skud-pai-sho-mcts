// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PaishoMPSGraph",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "PaishoMPSGraph", targets: ["PaishoMPSGraph"]),
        .executable(name: "paisho-mpsgraph-bench", targets: ["PaishoMPSGraphBench"]),
        .executable(name: "paisho-mpsgraph-service", targets: ["PaishoMPSGraphService"]),
    ],
    targets: [
        .target(
            name: "PaishoMPSGraph",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),
        .executableTarget(
            name: "PaishoMPSGraphBench",
            dependencies: ["PaishoMPSGraph"]
        ),
        .executableTarget(
            name: "PaishoMPSGraphService",
            dependencies: ["PaishoMPSGraph"]
        ),
        .testTarget(
            name: "PaishoMPSGraphTests",
            dependencies: ["PaishoMPSGraph"]
        ),
    ]
)
