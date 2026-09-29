import XCTest

@testable import PaishoMPSGraph

final class ConfigurationTests: XCTestCase {
  func testV1SchemaAndParameterCountsAreExact() {
    XCTAssertEqual(PaishoTensorSchemaV1.boardSize, 17)
    XCTAssertEqual(PaishoTensorSchemaV1.boardCells, 289)
    XCTAssertEqual(PaishoTensorSchemaV1.spatialChannels, 29)
    XCTAssertEqual(PaishoTensorSchemaV1.globalFeatures, 26)
    XCTAssertEqual(PaishoTensorSchemaV1.actionFamilies, 7)
    XCTAssertEqual(PaishoTensorSchemaV1.tileKinds, 12)
    XCTAssertEqual(PaishoNetworkConfiguration.pureV1.parameterCount, 4_698_679)
    XCTAssertEqual(PaishoNetworkConfiguration.microV1.parameterCount, 29_307)
  }

  func testInvalidConfigurationsAreRejected() {
    XCTAssertThrowsError(
      try PaishoNetworkConfiguration(
        trunkChannels: 0,
        residualBlocks: 1,
        policyEmbeddingChannels: 1,
        valueHiddenChannels: 1
      )
    )
    XCTAssertThrowsError(try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 0))
  }

  func testSyntheticBatchIsAValidPaddedDistribution() throws {
    let shape = try PaishoExecutionShape(batchSize: 3, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 42)
    XCTAssertNoThrow(try batch.validate())
    for row in 0..<shape.batchSize {
      let range = row * shape.legalActionCapacity..<(row + 1) * shape.legalActionCapacity
      XCTAssertEqual(batch.policyTargets[range].reduce(0, +), 1, accuracy: 1.0e-6)
    }
  }

  func testNegativeProbabilityIsRejectedEvenWhenRowSumsToOne() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 42)
    var targets = batch.policyTargets
    targets[0] = -0.25
    targets[1] += 0.25

    XCTAssertThrowsError(
      try PaishoTrainingBatch(
        inference: batch.inference,
        policyTargets: targets,
        valueTargets: batch.valueTargets
      )
    ) { error in
      XCTAssertEqual(
        error as? BatchError,
        .negativeProbability(name: "policyTargets", row: 0, value: -0.25)
      )
    }
  }
}
