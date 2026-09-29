import CryptoKit
import Foundation
import XCTest

@testable import PaishoMPSGraph

final class WeightsWireTests: XCTestCase {
  func testWeightReplacementAcrossShapesAfterPpoAndRepeatedImport() throws {
    let sourceShape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let actorShape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let source = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: sourceShape, optimization: .level1, seed: 401)
    let actor = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: actorShape, optimization: .level1, seed: 409)
    let example = PaishoInferenceExampleV1(
      spatial: [Float](repeating: 0.1, count: 289 * PaishoTensorSchemaV1.spatialChannels),
      global: [Float](repeating: 0.2, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: [
        try PaishoActionAddressV1(slots: [0, 0, 289, 8]),
        try PaishoActionAddressV1(slots: [2, 12, 289, 289]),
      ])
    let sourceBatch = try PaishoInferenceBatch.packing([example], legalActionCapacity: 8)
    let actorBatch = try PaishoInferenceBatch.packing([example, example], legalActionCapacity: 16)
    let parameters = try PaishoTerminalPpoParametersV1(
      clipEpsilon: 0.2,
      valueLossWeight: 0.5, entropyWeight: 0.01)
    // Existing actor Adam must survive importing unrelated learner weights.
    let actorPpo = try PaishoTerminalPpoBatch(
      inference: actorBatch,
      playedActionIndices: [0, 0], behaviorProbabilities: [0.5, 0.5],
      terminalValues: [.win, .loss], actorValues: [0, 0])
    _ = try actor.trainTerminalPpo(actorPpo, learningRate: 1.0e-4, parameters: parameters)
    let actorAdam = try actor.snapshotParameters()
    _ = try actor.servingInference(actorBatch)  // Populate cache before replacement.
    for iteration in 0..<2 {
      let ppo = try PaishoTerminalPpoBatch(
        inference: sourceBatch,
        playedActionIndices: [0], behaviorProbabilities: [0.5],
        terminalValues: [.win], actorValues: [0])
      _ = try source.trainTerminalPpo(ppo, learningRate: 1.0e-4, parameters: parameters)
      let packet = try source.exportWeights().encode()
      let snapshot = try PaishoWeightSnapshot.decode(packet)
      XCTAssertEqual(snapshot.trainingStep, UInt64(iteration + 1))
      XCTAssertEqual(packet.suffix(32), Data(SHA256.hash(data: packet.dropLast(32))))
      try actor.importWeights(snapshot)
      XCTAssertEqual(actor.executionShape, actorShape)
      XCTAssertEqual(actor.trainingStep, source.trainingStep)
      let imported = try actor.snapshotParameters()
      XCTAssertEqual(imported.map(\.momentum), actorAdam.map(\.momentum))
      XCTAssertEqual(imported.map(\.velocity), actorAdam.map(\.velocity))
      XCTAssertEqual(imported.map(\.values), snapshot.parameters.map(\.values))
      let expected = try source.servingInference(sourceBatch)
      let actual = try actor.servingInference(actorBatch)
      for row in 0..<2 {
        for index in 0..<2 {
          XCTAssertEqual(
            actual.policyProbabilities[row * 16 + index],
            expected.policyProbabilities[index], accuracy: 1.0e-5)
        }
        for index in 0..<3 {
          XCTAssertEqual(
            actual.valueProbabilities[row * 3 + index],
            expected.valueProbabilities[index], accuracy: 1.0e-5)
        }
      }
    }
    let good = try source.exportWeights()
    var malformed = good.parameters
    let last = malformed.removeLast()
    malformed.append(
      PaishoWeightArray(name: last.name, shape: [last.values.count, 1], values: last.values))
    let before = try actor.snapshotParameters()
    XCTAssertThrowsError(
      try actor.importWeights(
        PaishoWeightSnapshot(
          configuration: good.configuration, trainingStep: 99, parameters: malformed)))
    XCTAssertEqual(try actor.snapshotParameters(), before)
    XCTAssertEqual(actor.trainingStep, 2)
    var corrupt = try good.encode()
    corrupt[50] ^= 1
    XCTAssertThrowsError(try PaishoWeightSnapshot.decode(corrupt))
  }

  func testRequestFramingWithoutMetal() throws {
    var export = Data("PSW1".utf8)
    export.append(contentsOf: [7, 0, 0, 0, 0, 0, 0, 0])
    let decoded = try PaishoWeightsWire.decodeRequest(export)
    XCTAssertEqual(decoded.id, 7)
    XCTAssertNil(decoded.snapshot)
    XCTAssertThrowsError(try PaishoWeightsWire.decodeRequest(export + Data([0])))
    XCTAssertThrowsError(try PaishoWeightsWire.decodeRequest(Data("PSW2".utf8)))
    XCTAssertEqual(PaishoWeightsWire.requestID(export), 7)
    XCTAssertEqual(
      PaishoWeightsWire.encodeError(id: 7, error: WeightsWireError.invalidPacket).prefix(4),
      Data("PSWE".utf8))
  }
}
