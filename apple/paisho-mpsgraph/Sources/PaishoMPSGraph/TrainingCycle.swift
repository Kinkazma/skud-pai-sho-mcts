import Foundation

public enum PaishoTrainingCycleError: Error, Equatable {
  case noActiveCycle
  case generationMustAdvance(previous: UInt64, next: UInt64)
  case replayIndexMismatch(expected: UInt64, actual: UInt64)
  case replayIndexOverflow
}

extension PaishoMPSGraphModel {
  public var trainingProgress: PaishoTrainingProgress? { boundTrainingProgress }

  /// Start a fresh replay cycle without touching graph variables, Adam or the global step.
  /// Call only between completed training/inference operations. Use the cycle-aware
  /// training overloads below thereafter; legacy APIs do not advance this progress.
  public func beginNewGeneration(progress: PaishoTrainingProgress) throws {
    try Self.validateNewGeneration(
      progress: progress, trainingStep: trainingStep,
      previousGeneration: boundTrainingProgress?.generation)
    boundTrainingProgress = progress
  }

  /// Shared with services that own replay progress while using legacy training calls.
  /// Validation only: does not create a second, stale model-side progress binding.
  public static func validateNewGeneration(
    progress: PaishoTrainingProgress, trainingStep: UInt64, previousGeneration: UInt64?
  ) throws {
    try progress.validate()
    guard progress.scheduler.completedSteps == trainingStep else {
      throw PaishoCheckpointError.schedulerStepMismatch(
        model: trainingStep, scheduler: progress.scheduler.completedSteps)
    }
    guard progress.replayIndex == 0 else {
      throw PaishoTrainingCycleError.replayIndexMismatch(expected: 0, actual: progress.replayIndex)
    }
    if let previous = previousGeneration, progress.generation <= previous {
      throw PaishoTrainingCycleError.generationMustAdvance(
        previous: previous, next: progress.generation)
    }
  }

  /// Disk-backed equivalent of beginNewGeneration, preserving checkpoint Adam and step.
  public static func restoringForNewGeneration(
    from source: URL,
    progress: PaishoTrainingProgress
  ) throws -> PaishoMPSGraphModel {
    let checkpoint = try PaishoTrainingCheckpoint.read(from: source)
    let metadata = checkpoint.metadata
    let model = try PaishoMPSGraphModel(
      configuration: metadata.configuration,
      executionShape: metadata.executionShape,
      optimization: metadata.optimization,
      trainingStep: metadata.trainingStep,
      restoredParameters: checkpoint.parameters)
    model.boundTrainingProgress = metadata.progress
    try model.beginNewGeneration(progress: progress)
    return model
  }

  public func train(
    _ batch: PaishoTrainingBatch,
    replaySnapshotSHA256: String,
    startReplayIndex: UInt64
  ) throws -> PaishoTrainingResult {
    try inTrainingCycle(snapshot: replaySnapshotSHA256, index: startReplayIndex) { rate in
      try train(batch, learningRate: rate)
    }
  }

  public func trainTerminalPpo(
    _ batch: PaishoTerminalPpoBatch,
    replaySnapshotSHA256: String,
    startReplayIndex: UInt64,
    parameters: PaishoTerminalPpoParametersV1
  ) throws -> PaishoTerminalPpoResult {
    try inTrainingCycle(snapshot: replaySnapshotSHA256, index: startReplayIndex) { rate in
      try trainTerminalPpo(batch, learningRate: rate, parameters: parameters)
    }
  }

  private func inTrainingCycle<Result>(
    snapshot: String, index: UInt64, operation: (Float) throws -> Result
  ) throws -> Result {
    guard let progress = boundTrainingProgress else {
      throw PaishoTrainingCycleError.noActiveCycle
    }
    guard snapshot == progress.replaySnapshotSHA256 else {
      throw PaishoCheckpointError.replaySnapshotMismatch(
        expected: progress.replaySnapshotSHA256, actual: snapshot)
    }
    guard index == progress.replayIndex else {
      throw PaishoTrainingCycleError.replayIndexMismatch(
        expected: progress.replayIndex, actual: index)
    }
    guard progress.scheduler.completedSteps == trainingStep else {
      throw PaishoCheckpointError.schedulerStepMismatch(
        model: trainingStep, scheduler: progress.scheduler.completedSteps)
    }
    let (nextIndex, overflow) = index.addingReportingOverflow(UInt64(executionShape.batchSize))
    guard !overflow else { throw PaishoTrainingCycleError.replayIndexOverflow }
    var scheduler = progress.scheduler
    try scheduler.didCompleteStep()
    // Prepare all fallible metadata before modifying the graph.
    let next = try PaishoTrainingProgress(
      generation: progress.generation, replayIndex: nextIndex,
      replaySnapshotSHA256: snapshot, scheduler: scheduler, randomStates: progress.randomStates)
    let result = try operation(progress.scheduler.learningRate)
    boundTrainingProgress = next
    return result
  }
}
