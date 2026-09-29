import Foundation
import XCTest

@testable import PaishoMPSGraph

final class CheckpointTests: XCTestCase {
  func testProgressRequiresAnExactReplaySnapshotDigest() throws {
    let scheduler = try PaishoLearningRateScheduler(learningRate: 1.0e-4)
    XCTAssertThrowsError(
      try PaishoTrainingProgress(
        generation: 0,
        replayIndex: 0,
        replaySnapshotSHA256: String(repeating: "A", count: 64),
        scheduler: scheduler,
        randomStates: [try PaishoNamedRandomState(name: "learner", state: 1)]
      )
    ) {
      XCTAssertEqual($0 as? PaishoCheckpointError, .invalidReplaySnapshotSHA256)
    }
  }

  func testCheckpointRestoresExactAdamContinuation() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 101)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 103
    )
    var scheduler = try PaishoLearningRateScheduler(learningRate: 1.0e-4)
    _ = try model.train(batch, scheduler: &scheduler)

    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "paisho-checkpoint-test-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: directory) }
    let checkpointURL = directory.appendingPathComponent("generation-0004-step-0001.psckpt")
    let progress = try PaishoTrainingProgress(
      generation: 4,
      replayIndex: 1_234,
      replaySnapshotSHA256: String(repeating: "ab", count: 32),
      scheduler: scheduler,
      randomStates: [
        try PaishoNamedRandomState(name: "actor-seed-cursor", state: 77),
        try PaishoNamedRandomState(name: "replay-sampler", state: 88),
      ]
    )
    let contentDigest = try model.writeCheckpoint(to: checkpointURL, progress: progress)
    XCTAssertEqual(contentDigest.count, 32)
    XCTAssertEqual(contentDigest, Array(try Data(contentsOf: checkpointURL).suffix(32)))
    XCTAssertEqual(
      try model.writeCheckpointIdempotently(to: checkpointURL, progress: progress),
      contentDigest
    )
    let conflictingProgress = try PaishoTrainingProgress(
      generation: 5,
      replayIndex: progress.replayIndex,
      replaySnapshotSHA256: progress.replaySnapshotSHA256,
      scheduler: scheduler,
      randomStates: progress.randomStates
    )
    XCTAssertThrowsError(
      try model.writeCheckpointIdempotently(to: checkpointURL, progress: conflictingProgress)
    ) {
      guard case PaishoCheckpointError.destinationExists = $0 else {
        return XCTFail("unexpected error: \($0)")
      }
    }

    let originalSecond = try model.train(batch, scheduler: &scheduler)
    let originalParameters = try model.snapshotParameters()
    XCTAssertThrowsError(
      try PaishoMPSGraphModel.restoringCheckpoint(
        from: checkpointURL,
        replaySnapshotSHA256: String(repeating: "cd", count: 32)
      )
    ) {
      XCTAssertEqual(
        $0 as? PaishoCheckpointError,
        .replaySnapshotMismatch(
          expected: String(repeating: "cd", count: 32),
          actual: String(repeating: "ab", count: 32)
        )
      )
    }
    let restored = try PaishoMPSGraphModel.restoringCheckpoint(
      from: checkpointURL,
      replaySnapshotSHA256: String(repeating: "ab", count: 32)
    )
    var restoredScheduler = restored.progress.scheduler
    let restoredSecond = try restored.model.train(batch, scheduler: &restoredScheduler)
    let restoredParameters = try restored.model.snapshotParameters()

    XCTAssertEqual(restored.progress.generation, 4)
    XCTAssertEqual(restored.progress.replayIndex, 1_234)
    XCTAssertEqual(restored.progress.replaySnapshotSHA256, String(repeating: "ab", count: 32))
    XCTAssertEqual(
      restored.progress.randomStates.map(\.name),
      [
        "actor-seed-cursor", "replay-sampler",
      ])
    XCTAssertEqual(originalSecond.step, 2)
    XCTAssertEqual(restoredSecond.step, 2)
    XCTAssertEqual(originalSecond.totalLoss, restoredSecond.totalLoss, accuracy: 0)
    XCTAssertEqual(scheduler, restoredScheduler)
    assertSnapshotsEqual(originalParameters, restoredParameters, accuracy: 0)
  }

  func testCheckpointRejectsOverwriteAndCorruption() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 109
    )
    let scheduler = try PaishoLearningRateScheduler(learningRate: 1.0e-4)
    let progress = try PaishoTrainingProgress(
      generation: 0,
      replayIndex: 0,
      replaySnapshotSHA256: String(repeating: "00", count: 32),
      scheduler: scheduler,
      randomStates: [try PaishoNamedRandomState(name: "learner", state: 1)]
    )
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "paisho-checkpoint-test-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: directory) }
    let checkpointURL = directory.appendingPathComponent("step-0000.psckpt")
    try model.writeCheckpoint(to: checkpointURL, progress: progress)
    XCTAssertThrowsError(try model.writeCheckpoint(to: checkpointURL, progress: progress)) {
      guard case PaishoCheckpointError.destinationExists = $0 else {
        return XCTFail("unexpected error: \($0)")
      }
    }

    var corrupted = try Data(contentsOf: checkpointURL)
    corrupted[corrupted.startIndex + 20] ^= 0x01
    let corruptedURL = directory.appendingPathComponent("corrupted.psckpt")
    try corrupted.write(to: corruptedURL)
    XCTAssertThrowsError(try PaishoTrainingCheckpoint.read(from: corruptedURL)) {
      XCTAssertEqual($0 as? PaishoCheckpointError, .checksumMismatch)
    }
  }

  func testCheckpointWeightsCanServeDifferentInferenceShapes() throws {
    let sourceShape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let source = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: sourceShape,
      optimization: .level1,
      seed: 131
    )
    let scheduler = try PaishoLearningRateScheduler(learningRate: 1.0e-4)
    let progress = try PaishoTrainingProgress(
      generation: 0,
      replayIndex: 0,
      replaySnapshotSHA256: String(repeating: "00", count: 32),
      scheduler: scheduler,
      randomStates: [try PaishoNamedRandomState(name: "learner", state: 1)]
    )
    let checkpoint = try source.checkpoint(progress: progress)
    let actions = try [
      PaishoActionAddressV1(slots: [0, 0, 289, 8]),
      PaishoActionAddressV1(slots: [1, 12, 144, 145]),
      PaishoActionAddressV1(slots: [2, 12, 289, 289]),
    ]
    let example = PaishoInferenceExampleV1(
      spatial: [Float](
        repeating: 0,
        count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      ),
      global: [Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures),
      legalActions: actions
    )
    let sourceBatch = try PaishoInferenceBatch.packing(
      [example], legalActionCapacity: sourceShape.legalActionCapacity
    )
    let sourceOutput = try source.inference(sourceBatch)

    let servingShape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 32)
    let seededServing = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: servingShape,
      optimization: .level1,
      seed: 131
    )
    assertSnapshotsEqual(
      try source.snapshotParameters(), try seededServing.snapshotParameters(), accuracy: 0
    )
    let serving = try PaishoMPSGraphModel(
      configuration: checkpoint.metadata.configuration,
      executionShape: servingShape,
      optimization: .level1,
      trainingStep: checkpoint.metadata.trainingStep,
      restoredParameters: checkpoint.parameters
    )
    let servingBatch = try PaishoInferenceBatch.packing(
      [example, example], legalActionCapacity: servingShape.legalActionCapacity
    )
    let servingOutput = try serving.inference(servingBatch)

    assertArraysEqual(
      Array(sourceOutput.policyProbabilities.prefix(actions.count)),
      Array(servingOutput.policyProbabilities.prefix(actions.count)),
      accuracy: 1.0e-6
    )
    assertArraysEqual(
      sourceOutput.valueProbabilities,
      Array(servingOutput.valueProbabilities.prefix(PaishoTensorSchemaV1.valueClasses)),
      accuracy: 1.0e-6
    )
  }

  func testDuplicateRestoredParameterIsAnErrorNotATrap() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let initialized = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 113
    )
    let first = try XCTUnwrap(initialized.snapshotParameters().first)
    XCTAssertThrowsError(
      try PaishoMPSGraphModel(
        configuration: .microV1,
        executionShape: shape,
        optimization: .level0,
        restoredParameters: [first, first]
      )
    ) { error in
      XCTAssertEqual(
        String(describing: error),
        "checkpoint contains duplicate parameter \(first.name)"
      )
    }
  }

  func testResumedTrainingRejectsPartialAdamState() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let initialized = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 127
    )
    var snapshots = try initialized.snapshotParameters()
    let missingName = try XCTUnwrap(snapshots.popLast()?.name)

    XCTAssertThrowsError(
      try PaishoMPSGraphModel(
        configuration: .microV1,
        executionShape: shape,
        optimization: .level0,
        trainingStep: 1,
        restoredParameters: snapshots
      )
    ) { error in
      guard case PaishoMPSGraphError.missingCheckpointParameter(let name) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertEqual(name, missingName)
    }
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
      assertArraysEqual(lhs.values, rhs.values, accuracy: accuracy, file: file, line: line)
      assertArraysEqual(lhs.momentum, rhs.momentum, accuracy: accuracy, file: file, line: line)
      assertArraysEqual(lhs.velocity, rhs.velocity, accuracy: accuracy, file: file, line: line)
    }
  }

  private func assertArraysEqual(
    _ left: [Float],
    _ right: [Float],
    accuracy: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(left.count, right.count, file: file, line: line)
    guard left.count == right.count else { return }
    let maximumError = zip(left, right).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(maximumError, accuracy, file: file, line: line)
  }
}
