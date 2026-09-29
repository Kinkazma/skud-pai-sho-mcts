import Foundation
import XCTest

@testable import PaishoMPSGraph

final class TrainingCycleTests: XCTestCase {
  private let firstSnapshot = String(repeating: "ab", count: 32)
  private let secondSnapshot = String(repeating: "cd", count: 32)

  private func progress(generation: UInt64, step: UInt64, snapshot: String) throws
    -> PaishoTrainingProgress
  {
    try PaishoTrainingProgress(
      generation: generation, replayIndex: 0, replaySnapshotSHA256: snapshot,
      scheduler: PaishoLearningRateScheduler(learningRate: 1.0e-4, completedSteps: step),
      randomStates: [PaishoNamedRandomState(name: "replay-sampler", state: generation)])
  }

  func testRamCycleMatchesCheckpointNewGenerationIncludingAdam() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let first = try PaishoTrainingBatch.synthetic(shape: shape, seed: 101)
    let second = try PaishoTrainingBatch.synthetic(shape: shape, seed: 107)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1, executionShape: shape, optimization: .level1, seed: 103)
    try model.beginNewGeneration(
      progress: progress(generation: 1, step: 0, snapshot: firstSnapshot))
    _ = try model.train(first, replaySnapshotSHA256: firstSnapshot, startReplayIndex: 0)
    let completed = try XCTUnwrap(model.trainingProgress)
    XCTAssertEqual(completed.replayIndex, 1)
    XCTAssertEqual(completed.scheduler.completedSteps, 1)

    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "paisho-cycle-test-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let checkpoint = directory.appendingPathComponent("cycle-1.psckpt")
    try model.writeCheckpoint(to: checkpoint, progress: completed)
    let before = try model.snapshotParameters()
    let next = try progress(generation: 2, step: 1, snapshot: secondSnapshot)
    try model.beginNewGeneration(progress: next)
    XCTAssertEqual(model.trainingStep, 1)
    XCTAssertEqual(model.trainingProgress, next)
    XCTAssertEqual(try model.snapshotParameters(), before)

    let restored = try PaishoMPSGraphModel.restoringForNewGeneration(
      from: checkpoint, progress: next)
    XCTAssertEqual(restored.trainingProgress, next)
    XCTAssertEqual(restored.trainingStep, 1)
    XCTAssertEqual(try restored.snapshotParameters(), before)
    let ramResult = try model.train(
      second, replaySnapshotSHA256: secondSnapshot, startReplayIndex: 0)
    let diskResult = try restored.train(
      second, replaySnapshotSHA256: secondSnapshot, startReplayIndex: 0)
    XCTAssertEqual(ramResult.step, 2)
    XCTAssertEqual(diskResult.step, 2)
    XCTAssertEqual(ramResult.totalLoss, diskResult.totalLoss, accuracy: 0)
    // PaishoParameterSnapshot equality includes values, momentum and velocity.
    XCTAssertEqual(try model.snapshotParameters(), try restored.snapshotParameters())
    XCTAssertEqual(model.trainingProgress, restored.trainingProgress)
    XCTAssertEqual(model.trainingProgress?.replayIndex, 1)
  }

  func testSnapshotMismatchWithoutTransitionDoesNotTrain() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 131)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1, executionShape: shape, optimization: .level1, seed: 137)
    try model.beginNewGeneration(
      progress: progress(generation: 1, step: 0, snapshot: firstSnapshot))
    _ = try model.train(batch, replaySnapshotSHA256: firstSnapshot, startReplayIndex: 0)
    let before = try model.snapshotParameters()
    let completed = model.trainingProgress
    XCTAssertThrowsError(
      try model.train(batch, replaySnapshotSHA256: secondSnapshot, startReplayIndex: 1)
    ) {
      XCTAssertEqual(
        $0 as? PaishoCheckpointError,
        .replaySnapshotMismatch(expected: self.firstSnapshot, actual: self.secondSnapshot))
    }
    XCTAssertEqual(model.trainingStep, 1)
    XCTAssertEqual(model.trainingProgress, completed)
    XCTAssertEqual(try model.snapshotParameters(), before)

    XCTAssertThrowsError(
      try model.beginNewGeneration(
        progress: progress(generation: 2, step: 0, snapshot: secondSnapshot))
    ) {
      XCTAssertEqual($0 as? PaishoCheckpointError, .schedulerStepMismatch(model: 1, scheduler: 0))
    }
    XCTAssertThrowsError(
      try model.beginNewGeneration(
        progress: progress(generation: 1, step: 1, snapshot: secondSnapshot))
    ) {
      XCTAssertEqual($0 as? PaishoTrainingCycleError, .generationMustAdvance(previous: 1, next: 1))
    }
    XCTAssertThrowsError(
      try model.train(batch, replaySnapshotSHA256: firstSnapshot, startReplayIndex: 0)
    ) {
      XCTAssertEqual($0 as? PaishoTrainingCycleError, .replayIndexMismatch(expected: 1, actual: 0))
    }
    XCTAssertEqual(model.trainingProgress, completed)
    // A rejected request leaves the original cycle usable.
    let result = try model.train(batch, replaySnapshotSHA256: firstSnapshot, startReplayIndex: 1)
    XCTAssertEqual(result.step, 2)
    XCTAssertEqual(model.trainingProgress?.replayIndex, 2)
  }
}
