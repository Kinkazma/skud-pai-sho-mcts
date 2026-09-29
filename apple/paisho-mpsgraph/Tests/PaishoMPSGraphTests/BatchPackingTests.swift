import XCTest

@testable import PaishoMPSGraph

final class BatchPackingTests: XCTestCase {
  func testAllActionFamiliesMapToTheExpectedPolicyComponents() throws {
    let addresses = try [
      PaishoActionAddressV1(slots: [0, 0, 289, 8]),
      PaishoActionAddressV1(slots: [1, 12, 144, 145]),
      PaishoActionAddressV1(slots: [2, 12, 289, 289]),
      PaishoActionAddressV1(slots: [3, 8, 289, 144]),
      PaishoActionAddressV1(slots: [4, 11, 144, 145]),
      PaishoActionAddressV1(slots: [5, 6, 289, 152]),
      PaishoActionAddressV1(slots: [6, 5, 289, 280]),
    ]
    let example = PaishoInferenceExampleV1(
      spatial: [Float](
        repeating: 0,
        count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      ),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: addresses
    )
    let batch = try PaishoInferenceBatch.packing([example], legalActionCapacity: 8)

    XCTAssertEqual(batch.familyIndices, [0, 1, 2, 3, 4, 5, 6, 0])
    XCTAssertEqual(batch.tileIndices, [0, 0, 0, 8, 11, 6, 5, 0])
    XCTAssertEqual(batch.tilePresence, [1, 0, 0, 1, 1, 1, 1, 0])
    XCTAssertEqual(batch.destinationIndices, [8, 145, 0, 144, 145, 152, 280, 0])
    XCTAssertEqual(batch.destinationPresence, [1, 1, 0, 1, 1, 1, 1, 0])
    XCTAssertEqual(batch.pairIndices, [0, 41_761, 0, 0, 41_761, 0, 0, 0])
    XCTAssertEqual(batch.pairPresence, [0, 1, 0, 0, 1, 0, 0, 0])
    XCTAssertEqual(batch.legalMask, [1, 1, 1, 1, 1, 1, 1, 0])
  }

  func testMalformedActionShapesAreRejected() {
    XCTAssertThrowsError(try PaishoActionAddressV1(slots: [1, 12, 5, 5]))
    XCTAssertThrowsError(try PaishoActionAddressV1(slots: [4, 10, 5, 6]))
    XCTAssertThrowsError(try PaishoActionAddressV1(slots: [5, 8, 289, 6]))
    XCTAssertThrowsError(try PaishoActionAddressV1(slots: [7, 12, 289, 289]))
    XCTAssertThrowsError(try PaishoActionAddressV1(slots: [1, 12, 0, 144]))
    XCTAssertThrowsError(try PaishoActionAddressV1(slots: [0, 0, 289, 144]))
  }

  func testInferenceBatchDoesNotRequireTrainingTargets() throws {
    let address = try PaishoActionAddressV1(slots: [2, 12, 289, 289])
    let example = PaishoInferenceExampleV1(
      spatial: [Float](
        repeating: 0,
        count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      ),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: [address]
    )
    let batch = try PaishoInferenceBatch.packing([example], legalActionCapacity: 4)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: batch.shape,
      optimization: .level0,
      seed: 127
    )
    let output = try model.inference(batch)

    XCTAssertEqual(output.policyProbabilities, [1, 0, 0, 0])
  }

  func testPackingRejectsCompensatingPerExampleFeatureLengths() throws {
    let spatialCount =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
    let action = try PaishoActionAddressV1(slots: [2, 12, 289, 289])
    let shortSpatial = PaishoInferenceExampleV1(
      spatial: [Float](repeating: 0, count: spatialCount - 1),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: [action]
    )
    let longSpatial = PaishoInferenceExampleV1(
      spatial: [Float](repeating: 0, count: spatialCount + 1),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: [action]
    )
    XCTAssertThrowsError(
      try PaishoInferenceBatch.packing(
        [shortSpatial, longSpatial], legalActionCapacity: 1
      )
    ) { error in
      XCTAssertEqual(
        error as? BatchError,
        .wrongExampleCount(
          name: "spatial", row: 0, expected: spatialCount, actual: spatialCount - 1
        )
      )
    }

    let shortGlobal = PaishoInferenceExampleV1(
      spatial: [Float](repeating: 0, count: spatialCount),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures - 1),
      legalActions: [action]
    )
    let longGlobal = PaishoInferenceExampleV1(
      spatial: [Float](repeating: 0, count: spatialCount),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures + 1),
      legalActions: [action]
    )
    XCTAssertThrowsError(
      try PaishoInferenceBatch.packing(
        [shortGlobal, longGlobal], legalActionCapacity: 1
      )
    ) { error in
      XCTAssertEqual(
        error as? BatchError,
        .wrongExampleCount(
          name: "global",
          row: 0,
          expected: PaishoTensorSchemaV1.globalFeatures,
          actual: PaishoTensorSchemaV1.globalFeatures - 1
        )
      )
    }
  }

  func testTrainingPackingRejectsCompensatingValueTargetLengths() throws {
    let inference = try validPassExample()
    let short = PaishoTrainingExampleV1(
      inference: inference,
      policyTargets: [1],
      valueTargets: [1, 0]
    )
    let long = PaishoTrainingExampleV1(
      inference: inference,
      policyTargets: [1],
      valueTargets: [0, 0, 1, 0]
    )

    XCTAssertThrowsError(
      try PaishoTrainingBatch.packing([short, long], legalActionCapacity: 1)
    ) { error in
      XCTAssertEqual(
        error as? BatchError,
        .wrongValueTargetCount(row: 0, expected: 3, actual: 2)
      )
    }
  }

  func testRawBatchRejectsNonCanonicalActionComponents() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 1)
    XCTAssertThrowsError(
      try PaishoInferenceBatch(
        shape: shape,
        spatial: [Float](
          repeating: 0,
          count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
        ),
        global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
        familyIndices: [2],
        tileIndices: [11],
        tilePresence: [1],
        destinationIndices: [145],
        destinationPresence: [1],
        pairIndices: [41_761],
        pairPresence: [1],
        legalMask: [1]
      )
    ) { error in
      XCTAssertEqual(error as? BatchError, .invalidActionComponents(row: 0, index: 0))
    }
  }

  private func validPassExample() throws -> PaishoInferenceExampleV1 {
    PaishoInferenceExampleV1(
      spatial: [Float](
        repeating: 0,
        count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      ),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: [try PaishoActionAddressV1(slots: [2, 12, 289, 289])]
    )
  }
}
