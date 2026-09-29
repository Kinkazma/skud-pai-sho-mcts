import Foundation
import XCTest

@testable import PaishoMPSGraph

final class TerminalPpoTests: XCTestCase {
  func testDurationUtilityWireKeepsLossAndWdlUnchanged() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    var payload = try requestPayload(shape: shape, requestID: 41, expectedStep: 7)
    payload.replaceSubrange(0..<8, with: Data("PSTREQ03".utf8))
    var value = Float(-1).bitPattern.littleEndian
    withUnsafeBytes(of: &value) { payload.append(contentsOf: $0) }
    let decoded = try PaishoTerminalPpoWireV1.decodeRequest(payload, expectedShape: shape)
    XCTAssertEqual(decoded.batch.advantages, [-0.75])
    XCTAssertEqual(decoded.batch.valueTargets, [0,0,1])
    payload.removeLast(4)
    value = Float(0.95).bitPattern.littleEndian
    withUnsafeBytes(of: &value) { payload.append(contentsOf: $0) }
    XCTAssertThrowsError(try PaishoTerminalPpoWireV1.decodeRequest(payload, expectedShape: shape))
  }

  func testWinUtilityChangesPolicyButNotValueTarget() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 101)
    let batch = try PaishoTerminalPpoBatch(inference: source.inference,
      playedActionIndices: [0], behaviorProbabilities: [0.2], terminalValues: [.win],
      actorValues: [0.25], policyReturns: [0.95])
    XCTAssertEqual(batch.advantages[0], 0.70, accuracy: 1e-6)
    XCTAssertEqual(batch.valueTargets, [1,0,0])
  }

  func testBatchCarriesOnePlayedActionAndBothTerminalPerspectives() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 101)
    let batch = try PaishoTerminalPpoBatch(
      inference: source.inference,
      playedActionIndices: [0, 1],
      behaviorProbabilities: [0.2, 0.3],
      terminalValues: [.win, .loss],
      actorValues: [0.25, -0.5]
    )

    XCTAssertEqual(batch.playedActionMask[0], 1)
    XCTAssertEqual(batch.playedActionMask[shape.legalActionCapacity + 1], 1)
    XCTAssertEqual(batch.playedActionMask.reduce(0, +), 2)
    XCTAssertEqual(batch.advantages, [0.75, -0.5])
    XCTAssertEqual(batch.valueTargets, [1, 0, 0, 0, 0, 1])
  }

  func testBatchRejectsInvalidBehaviorAndPlayedAction() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 103)

    XCTAssertThrowsError(
      try PaishoTerminalPpoBatch(
        inference: source.inference,
        playedActionIndices: [0],
        behaviorProbabilities: [0],
        terminalValues: [.win],
        actorValues: [0]
      )
    ) {
      XCTAssertEqual(
        $0 as? TerminalPpoBatchError,
        .invalidBehaviorProbability(row: 0, value: 0)
      )
    }
    XCTAssertThrowsError(
      try PaishoTerminalPpoBatch(
        inference: source.inference,
        playedActionIndices: [shape.legalActionCapacity - 1],
        behaviorProbabilities: [0.5],
        terminalValues: [.loss],
        actorValues: [0]
      )
    ) {
      XCTAssertEqual(
        $0 as? TerminalPpoBatchError,
        .playedActionNotLegal(row: 0, index: shape.legalActionCapacity - 1)
      )
    }
  }

  func testWireDecodesRustOrderAndEncodesMetrics() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let payload = try requestPayload(shape: shape, requestID: 41, expectedStep: 7)

    let request = try PaishoTerminalPpoWireV1.decodeRequest(payload, expectedShape: shape)
    XCTAssertEqual(request.requestID, 41)
    XCTAssertEqual(request.expectedTrainingStep, 7)
    XCTAssertEqual(request.learningRate, 1.0e-4)
    XCTAssertEqual(
      request.parameters,
      try PaishoTerminalPpoParametersV1(
        policyTemperature: 0.8,
        uniformMix: 0.05,
        clipEpsilon: 0.2,
        valueLossWeight: 0.5,
        entropyWeight: 0.01
      )
    )
    XCTAssertEqual(request.replaySnapshotSHA256, String(repeating: "ab", count: 32))
    XCTAssertEqual(request.startReplayIndex, 90)
    XCTAssertEqual(request.nextReplayIndex, 91)
    XCTAssertEqual(request.batch.playedActionMask[1], 1)
    XCTAssertEqual(request.batch.behaviorProbabilities, [0.75])
    XCTAssertEqual(request.batch.advantages, [-0.75])
    XCTAssertEqual(request.batch.valueTargets, [0, 0, 1])
    XCTAssertTrue(PaishoTerminalPpoWireV1.isRequest(payload))
    XCTAssertEqual(PaishoTerminalPpoWireV1.requestIDIfPresent(in: payload), 41)
    XCTAssertLessThanOrEqual(
      payload.count,
      try PaishoTerminalPpoWireV1.maximumRequestPayloadSize(shape: shape)
    )

    let response = PaishoTerminalPpoWireV1.encodeResponse(
      requestID: 41,
      completedReplayIndex: 91,
      result: PaishoTerminalPpoResult(
        step: 8,
        policyLoss: -0.25,
        valueLoss: 0.75,
        entropy: 1.5,
        totalLoss: 0.11,
        meanAdvantage: -0.75,
        meanImportanceRatio: 1.25,
        meanSquaredRatioDeviation: 0.0625
      )
    )
    var reader = TerminalPpoTestReader(response)
    XCTAssertEqual(try reader.bytes(count: 8), PaishoTerminalPpoWireV1.responseMagic)
    XCTAssertEqual(try reader.uint64(), 41)
    XCTAssertEqual(try reader.uint64(), 8)
    XCTAssertEqual(try reader.uint64(), 91)
    XCTAssertEqual(try reader.float(), -0.25)
    XCTAssertEqual(try reader.float(), 0.75)
    XCTAssertEqual(try reader.float(), 1.5)
    XCTAssertEqual(try reader.float(), 0.11)
    XCTAssertEqual(try reader.float(), -0.75)
    XCTAssertEqual(try reader.float(), 1.25)
    XCTAssertEqual(try reader.float(), 0.0625)
    XCTAssertTrue(reader.isAtEnd)
  }

  func testPositiveAndNegativeReturnsMovePlayedProbabilityInOppositeDirections() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 107)
    let parameters = try PaishoTerminalPpoParametersV1(
      clipEpsilon: 0.2,
      valueLossWeight: 0,
      entropyWeight: 0
    )

    let winner = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 109
    )
    let winnerBefore = try winner.inference(source).policyProbabilities[0]
    let winnerBatch = try terminalBatch(
      inference: source.inference,
      behaviorProbability: winnerBefore,
      terminalValue: .win
    )
    let winnerStep = try winner.trainTerminalPpo(
      winnerBatch,
      learningRate: 1.0e-3,
      parameters: parameters
    )
    let winnerAfter = try winner.inference(source).policyProbabilities[0]

    let loser = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 109
    )
    let loserBefore = try loser.inference(source).policyProbabilities[0]
    let loserBatch = try terminalBatch(
      inference: source.inference,
      behaviorProbability: loserBefore,
      terminalValue: .loss
    )
    let loserStep = try loser.trainTerminalPpo(
      loserBatch,
      learningRate: 1.0e-3,
      parameters: parameters
    )
    let loserAfter = try loser.inference(source).policyProbabilities[0]

    XCTAssertEqual(winnerBefore, loserBefore, accuracy: 1.0e-7)
    XCTAssertGreaterThan(winnerAfter, winnerBefore)
    XCTAssertLessThan(loserAfter, loserBefore)
    XCTAssertEqual(winnerStep.meanAdvantage, 1, accuracy: 1.0e-6)
    XCTAssertEqual(loserStep.meanAdvantage, -1, accuracy: 1.0e-6)
    XCTAssertTrue(winnerStep.totalLoss.isFinite)
    XCTAssertTrue(loserStep.totalLoss.isFinite)
  }

  func testNonDefaultBehaviorTransformStartsAtUnitImportanceRatio() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 149)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 151
    )
    let policy = try model.inference(source).policyProbabilities
    let temperature: Float = 0.8
    let uniformMix: Float = 0.05
    let behavior = transformedBehaviorPolicy(
      policy,
      legalCount: Int(source.inference.legalMask.prefix(shape.legalActionCapacity).reduce(0, +)),
      temperature: temperature,
      uniformMix: uniformMix
    )
    let batch = try terminalBatch(
      inference: source.inference,
      behaviorProbability: behavior[0],
      terminalValue: .win
    )
    let result = try model.trainTerminalPpo(
      batch,
      learningRate: 1.0e-4,
      parameters: try PaishoTerminalPpoParametersV1(
        policyTemperature: temperature,
        uniformMix: uniformMix,
        clipEpsilon: 0.2,
        valueLossWeight: 0,
        entropyWeight: 0
      )
    )

    XCTAssertEqual(result.policyLoss, -1, accuracy: 2.0e-4)
    XCTAssertEqual(result.meanImportanceRatio, 1, accuracy: 2.0e-4)
    XCTAssertEqual(result.meanSquaredRatioDeviation, 0, accuracy: 2.0e-4)
  }

  func testExtremeFiniteTemperatureKeepsPaddingOutOfBehaviorPolicy() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 153)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 157
    )
    let policy = try model.inference(source).policyProbabilities
    let legalCount = Int(
      source.inference.legalMask.prefix(shape.legalActionCapacity).reduce(0, +)
    )
    let temperature: Float = 1.0e9
    let behavior = transformedBehaviorPolicy(
      policy,
      legalCount: legalCount,
      temperature: temperature,
      uniformMix: 0
    )
    let result = try model.trainTerminalPpo(
      try terminalBatch(
        inference: source.inference,
        behaviorProbability: behavior[0],
        terminalValue: .win
      ),
      learningRate: 1.0e-4,
      parameters: try PaishoTerminalPpoParametersV1(
        policyTemperature: temperature,
        uniformMix: 0,
        clipEpsilon: 0.2,
        valueLossWeight: 0,
        entropyWeight: 0
      )
    )

    XCTAssertEqual(result.policyLoss, -1, accuracy: 2.0e-4)
  }

  func testPositiveSurrogateIsClippedAgainstBehaviorProbability() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 113)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 127
    )
    let currentProbability = try model.inference(source).policyProbabilities[0]
    let batch = try terminalBatch(
      inference: source.inference,
      behaviorProbability: currentProbability / 2,
      terminalValue: .win
    )
    let result = try model.trainTerminalPpo(
      batch,
      learningRate: 1.0e-5,
      parameters: try PaishoTerminalPpoParametersV1(
        clipEpsilon: 0.2,
        valueLossWeight: 0,
        entropyWeight: 0
      )
    )

    XCTAssertEqual(result.policyLoss, -1.2, accuracy: 1.0e-4)
    XCTAssertEqual(result.totalLoss, result.policyLoss, accuracy: 1.0e-5)
  }

  func testClippedPositiveSurrogateStaysFiniteWhenRawRatioOverflows() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 163)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 167
    )
    let result = try model.trainTerminalPpo(
      try terminalBatch(
        inference: source.inference,
        behaviorProbability: Float.leastNonzeroMagnitude,
        terminalValue: .win
      ),
      learningRate: 1.0e-5,
      parameters: try PaishoTerminalPpoParametersV1(
        clipEpsilon: 0.2,
        valueLossWeight: 0,
        entropyWeight: 0
      )
    )

    XCTAssertTrue(result.policyLoss.isFinite)
    XCTAssertEqual(result.policyLoss, -1.2, accuracy: 1.0e-4)
    XCTAssertEqual(result.totalLoss, result.policyLoss, accuracy: 1.0e-5)
  }

  func testTerminalPpoCheckpointRestoresExactAdamContinuation() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 131)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 137
    )
    let actor = try model.inference(source)
    let actorWdl = actor.valueProbabilities
    let batch = try PaishoTerminalPpoBatch(
      inference: source.inference,
      playedActionIndices: [0],
      behaviorProbabilities: [actor.policyProbabilities[0]],
      terminalValues: [.win],
      actorValues: [actorWdl[0] - actorWdl[2]]
    )
    let parameters = try PaishoTerminalPpoParametersV1(
      clipEpsilon: 0.2,
      valueLossWeight: 0.5,
      entropyWeight: 0.01
    )
    _ = try model.trainTerminalPpo(batch, learningRate: 1.0e-4, parameters: parameters)

    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "paisho-terminal-ppo-checkpoint-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: directory) }
    let checkpointURL = directory.appendingPathComponent("step-0001.psckpt")
    let progress = try PaishoTrainingProgress(
      generation: 0,
      replayIndex: 1,
      replaySnapshotSHA256: String(repeating: "ef", count: 32),
      scheduler: try PaishoLearningRateScheduler(
        learningRate: 1.0e-4,
        completedSteps: model.trainingStep
      ),
      randomStates: [try PaishoNamedRandomState(name: "learner", state: 17)]
    )
    try model.writeCheckpoint(to: checkpointURL, progress: progress)

    let originalSecond = try model.trainTerminalPpo(
      batch,
      learningRate: 1.0e-4,
      parameters: parameters
    )
    let originalParameters = try model.snapshotParameters()
    let restored = try PaishoMPSGraphModel.restoringCheckpoint(
      from: checkpointURL,
      replaySnapshotSHA256: String(repeating: "ef", count: 32)
    )
    let restoredSecond = try restored.model.trainTerminalPpo(
      batch,
      learningRate: 1.0e-4,
      parameters: parameters
    )

    XCTAssertEqual(originalSecond.step, 2)
    XCTAssertEqual(originalSecond.policyLoss, restoredSecond.policyLoss, accuracy: 0)
    XCTAssertEqual(originalSecond.valueLoss, restoredSecond.valueLoss, accuracy: 0)
    XCTAssertEqual(originalSecond.entropy, restoredSecond.entropy, accuracy: 0)
    XCTAssertEqual(originalSecond.totalLoss, restoredSecond.totalLoss, accuracy: 0)
    assertSnapshotsEqual(
      originalParameters,
      try restored.model.snapshotParameters(),
      accuracy: 0
    )
  }

  func testCompiledTerminalPpoMatchesSequentialUpdates() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let source = try PaishoTrainingBatch.synthetic(shape: shape, seed: 131)
    let legacy = try PaishoMPSGraphModel(configuration: .microV1,
      executionShape: shape, optimization: .level1, seed: 137)
    let compiled = try PaishoMPSGraphModel(configuration: .microV1,
      executionShape: shape, optimization: .level1, seed: 137)
    compiled.useCompiledTerminalPpo = true
    let actor = try legacy.inference(source)
    let batch = try terminalBatch(inference: source.inference,
      behaviorProbability: actor.policyProbabilities[0], terminalValue: .win)
    let parameters = try PaishoTerminalPpoParametersV1(
      clipEpsilon: 0.2, valueLossWeight: 0.5, entropyWeight: 0.01)
    legacy.profileTerminalPpo = true
    let beforeProbe = try legacy.snapshotParameters()
    for probe: PaishoTerminalPpoProbe in [.loss, .gradients] {
      let measured = try legacy.probeTerminalPpo(batch, parameters: parameters, probe: probe)
      XCTAssertEqual(measured.step, 0)
      XCTAssertEqual(legacy.trainingStep, 0)
      assertSnapshotsEqual(beforeProbe, try legacy.snapshotParameters(), accuracy: 0)
      let timing = try XCTUnwrap(legacy.lastTerminalPpoTiming)
      XCTAssertGreaterThan(timing.executionSeconds, 0)
      XCTAssertGreaterThanOrEqual(timing.feedsSeconds, 0)
    }
    for _ in 0..<5 {
      let a = try legacy.trainTerminalPpo(batch, learningRate: 1e-4, parameters: parameters)
      let b = try compiled.trainTerminalPpo(batch, learningRate: 1e-4, parameters: parameters)
      XCTAssertNil(compiled.lastTerminalPpoTiming)
      XCTAssertEqual(a.totalLoss, b.totalLoss, accuracy: 1e-5)
      XCTAssertEqual(a.step, b.step)
      let x = try legacy.inference(source)
      let y = try compiled.inference(source)
      for (u, v) in zip(x.policyProbabilities, y.policyProbabilities) {
        XCTAssertEqual(u, v, accuracy: 1e-5)
      }
      for (u, v) in zip(x.valueProbabilities, y.valueProbabilities) {
        XCTAssertEqual(u, v, accuracy: 1e-5)
      }
      assertSnapshotsEqual(try legacy.snapshotParameters(),
                           try compiled.snapshotParameters(), accuracy: 1e-5)
    }
  }

  private func terminalBatch(
    inference: PaishoInferenceBatch,
    behaviorProbability: Float,
    terminalValue: PaishoValueClassV1
  ) throws -> PaishoTerminalPpoBatch {
    try PaishoTerminalPpoBatch(
      inference: inference,
      playedActionIndices: [0],
      behaviorProbabilities: [behaviorProbability],
      terminalValues: [terminalValue],
      actorValues: [0]
    )
  }

  private func requestPayload(
    shape: PaishoExecutionShape,
    requestID: UInt64,
    expectedStep: UInt64
  ) throws -> Data {
    let actions = [
      try PaishoActionAddressV1(
        slots: [
          2, PaishoActionAddressV1.noTile, PaishoActionAddressV1.noCoordinate,
          PaishoActionAddressV1.noCoordinate,
        ]
      ),
      try PaishoActionAddressV1(
        slots: [1, PaishoActionAddressV1.noTile, 8, 25]
      ),
    ]
    var writer = TerminalPpoTestWriter()
    writer.bytes(PaishoTerminalPpoWireV1.requestMagic)
    writer.uint64(requestID)
    writer.uint64(expectedStep)
    writer.float(1.0e-4)
    writer.float(0.8)
    writer.float(0.05)
    writer.float(0.2)
    writer.float(0.5)
    writer.float(0.01)
    writer.uint32(UInt32(shape.batchSize))
    writer.uint32(UInt32(shape.legalActionCapacity))
    writer.bytes([UInt8](repeating: 0xab, count: 32))
    writer.uint64(90)
    writer.uint64(91)
    writer.floats(
      [Float](
        repeating: 0,
        count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      )
    )
    writer.floats([Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures))
    writer.uint32(UInt32(actions.count))
    for action in actions {
      for slot in action.slots { writer.uint16(slot) }
    }
    writer.uint32(1)
    writer.float(0.75)
    writer.uint32(UInt32(PaishoValueClassV1.loss.rawValue))
    writer.float(-0.25)
    return writer.data
  }

  private func assertSnapshotsEqual(
    _ left: [PaishoParameterSnapshot],
    _ right: [PaishoParameterSnapshot],
    accuracy: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(left.map(\.name), right.map(\.name), file: file, line: line)
    for (lhs, rhs) in zip(left, right) {
      XCTAssertEqual(lhs.shape, rhs.shape, file: file, line: line)
      for (old, new) in zip(lhs.values, rhs.values) {
        XCTAssertEqual(old, new, accuracy: accuracy, file: file, line: line)
      }
      for (old, new) in zip(lhs.momentum, rhs.momentum) {
        XCTAssertEqual(old, new, accuracy: accuracy, file: file, line: line)
      }
      for (old, new) in zip(lhs.velocity, rhs.velocity) {
        XCTAssertEqual(old, new, accuracy: accuracy, file: file, line: line)
      }
    }
  }

  private func transformedBehaviorPolicy(
    _ policy: [Float],
    legalCount: Int,
    temperature: Float,
    uniformMix: Float
  ) -> [Float] {
    let inverseTemperature = 1.0 / Double(temperature)
    let powered = policy.prefix(legalCount).map {
      pow(Double($0), inverseTemperature)
    }
    let total = powered.reduce(0, +)
    let uniform = 1.0 / Double(legalCount)
    return powered.map {
      Float((1.0 - Double(uniformMix)) * ($0 / total) + Double(uniformMix) * uniform)
    }
  }
}

private struct TerminalPpoTestWriter {
  var data = Data()

  mutating func bytes(_ values: [UInt8]) {
    data.append(contentsOf: values)
  }

  mutating func uint16(_ value: UInt16) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  mutating func uint32(_ value: UInt32) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  mutating func uint64(_ value: UInt64) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  mutating func float(_ value: Float) {
    uint32(value.bitPattern)
  }

  mutating func floats(_ values: [Float]) {
    for value in values { float(value) }
  }
}

private struct TerminalPpoTestReader {
  let data: Data
  var cursor = 0

  init(_ data: Data) {
    self.data = data
  }

  var isAtEnd: Bool { cursor == data.count }

  mutating func bytes(count: Int) throws -> [UInt8] {
    guard cursor + count <= data.count else {
      throw PaishoTrainingWireError.truncatedPayload
    }
    let result = Array(data[cursor..<cursor + count])
    cursor += count
    return result
  }

  mutating func uint32() throws -> UInt32 {
    let value = try bytes(count: 4)
    return value.enumerated().reduce(0) { result, item in
      result | UInt32(item.element) << UInt32(item.offset * 8)
    }
  }

  mutating func uint64() throws -> UInt64 {
    let value = try bytes(count: 8)
    return value.enumerated().reduce(0) { result, item in
      result | UInt64(item.element) << UInt64(item.offset * 8)
    }
  }

  mutating func float() throws -> Float {
    Float(bitPattern: try uint32())
  }
}
