import Foundation
import PaishoMPSGraph

enum Preset: String {
  case micro
  case pure

  var configuration: PaishoNetworkConfiguration {
    switch self {
    case .micro: .microV1
    case .pure: .pureV1
    }
  }
}

enum CheckpointMode: String {
  case resume
  case newGeneration = "new-generation"
}

struct Options {
  var preset = Preset.pure
  var batchSize = 1
  var legalActions = 128
  var inferenceSlots = 1
  var optimization = PaishoGraphOptimization.level1
  var seed: UInt64 = 1
  var checkpointPath: String?
  var checkpointMode = CheckpointMode.resume

  static func parse(_ arguments: [String]) throws -> Self {
    var options = Self()
    var index = 0
    while index < arguments.count {
      let flag = arguments[index]
      if flag == "--help" || flag == "-h" {
        printUsage()
        exit(0)
      }
      guard index + 1 < arguments.count else {
        throw ArgumentError.missingValue(flag)
      }
      let value = arguments[index + 1]
      switch flag {
      case "--preset":
        guard let parsed = Preset(rawValue: value) else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.preset = parsed
      case "--batch": options.batchSize = try positiveWireInt(value, flag: flag)
      case "--actions": options.legalActions = try positiveWireInt(value, flag: flag)
      case "--inference-slots":
        options.inferenceSlots = try positiveWireInt(value, flag: flag)
      case "--level":
        switch value {
        case "0": options.optimization = .level0
        case "1": options.optimization = .level1
        default: throw ArgumentError.invalidValue(flag, value)
        }
      case "--seed":
        guard let parsed = UInt64(value) else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.seed = parsed
      case "--checkpoint": options.checkpointPath = value
      case "--checkpoint-mode":
        guard let parsed = CheckpointMode(rawValue: value) else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.checkpointMode = parsed
      default: throw ArgumentError.unknownFlag(flag)
      }
      index += 2
    }
    return options
  }
}

enum ArgumentError: Error, CustomStringConvertible {
  case missingValue(String)
  case invalidValue(String, String)
  case unknownFlag(String)
  case checkpointPresetMismatch(String)
  case newGenerationWithoutCheckpoint

  var description: String {
    switch self {
    case .missingValue(let flag): "missing value for \(flag)"
    case .invalidValue(let flag, let value): "invalid value \(value) for \(flag)"
    case .unknownFlag(let flag): "unknown option \(flag)"
    case .checkpointPresetMismatch(let path):
      "checkpoint \(path) does not match the selected network preset"
    case .newGenerationWithoutCheckpoint:
      "checkpoint mode new-generation requires --checkpoint"
    }
  }
}

enum FrameError: Error, CustomStringConvertible {
  case truncatedLength
  case truncatedPayload(expected: Int, actual: Int)
  case payloadTooLarge(actual: UInt64, maximum: Int)

  var description: String {
    switch self {
    case .truncatedLength: "truncated protocol frame length"
    case .truncatedPayload(let expected, let actual):
      "truncated protocol frame: received \(actual) of \(expected) bytes"
    case .payloadTooLarge(let actual, let maximum):
      "protocol frame has \(actual) bytes; maximum is \(maximum)"
    }
  }
}

enum WireRequestKind: Sendable {
  case inference
  case training
  case terminalPpoTraining
  case checkpoint
  case trainingCycle
  case weights

  init(payload: Data) {
    if PaishoWeightsWire.isRequest(payload) {
      self = .weights
    } else if PaishoTrainingCycleWireV1.isRequest(payload) {
      self = .trainingCycle
    } else if PaishoCheckpointWireV1.isRequest(payload) {
      self = .checkpoint
    } else if PaishoTerminalPpoWireV1.isRequest(payload) {
      self = .terminalPpoTraining
    } else if PaishoTrainingWireV1.isRequest(payload) {
      self = .training
    } else {
      self = .inference
    }
  }

  func requestID(in payload: Data) -> UInt64 {
    switch self {
    case .inference: PaishoInferenceWireV1.requestIDIfPresent(in: payload)
    case .training: PaishoTrainingWireV1.requestIDIfPresent(in: payload)
    case .terminalPpoTraining:
      PaishoTerminalPpoWireV1.requestIDIfPresent(in: payload)
    case .checkpoint: PaishoCheckpointWireV1.requestIDIfPresent(in: payload)
    case .trainingCycle: PaishoTrainingCycleWireV1.requestIDIfPresent(in: payload)
    case .weights: PaishoWeightsWire.requestID(payload)
    }
  }

  func encodeError(requestID: UInt64, error: Error) -> Data {
    switch self {
    case .inference: PaishoInferenceWireV1.encodeError(requestID: requestID, error: error)
    case .training: PaishoTrainingWireV1.encodeError(requestID: requestID, error: error)
    case .terminalPpoTraining:
      PaishoTerminalPpoWireV1.encodeError(requestID: requestID, error: error)
    case .checkpoint: PaishoCheckpointWireV1.encodeError(requestID: requestID, error: error)
    case .trainingCycle: PaishoTrainingCycleWireV1.encodeError(requestID: requestID, error: error)
    case .weights: PaishoWeightsWire.encodeError(id: requestID, error: error)
    }
  }
}

func positiveWireInt(_ value: String, flag: String) throws -> Int {
  guard let parsed = Int(value), parsed > 0, UInt32(exactly: parsed) != nil else {
    throw ArgumentError.invalidValue(flag, value)
  }
  return parsed
}

func printUsage() {
  print(
    """
    usage: paisho-mpsgraph-service [options]
      --preset micro|pure
      --batch N
      --actions N
      --inference-slots N
      --level 0|1
      --seed N
      --checkpoint PATH
      --checkpoint-mode resume|new-generation

    Standard input and output carry little-endian UInt64-length-prefixed
    PSI V1 inference, PST V1 supervised training, PST V2 terminal PPO training,
    or PSC V1 checkpoint frames. Diagnostics use standard error only.
    """
  )
}

func readExactly(
  _ count: Int,
  from handle: FileHandle,
  allowCleanEnd: Bool
) throws -> Data? {
  var data = Data()
  while data.count < count {
    let chunk = try handle.read(upToCount: count - data.count) ?? Data()
    if chunk.isEmpty {
      if data.isEmpty && allowCleanEnd { return nil }
      if allowCleanEnd { throw FrameError.truncatedLength }
      throw FrameError.truncatedPayload(expected: count, actual: data.count)
    }
    data.append(chunk)
  }
  return data
}

func decodeFrameLength(_ data: Data) -> UInt64 {
  data.enumerated().reduce(0) { result, item in
    result | UInt64(item.element) << UInt64(item.offset * 8)
  }
}

func readFrame(from handle: FileHandle, maximum: Int) throws -> Data? {
  guard let lengthData = try readExactly(8, from: handle, allowCleanEnd: true) else {
    return nil
  }
  let length = decodeFrameLength(lengthData)
  guard length <= UInt64(maximum), length <= UInt64(Int.max) else {
    throw FrameError.payloadTooLarge(actual: length, maximum: maximum)
  }
  return try readExactly(Int(length), from: handle, allowCleanEnd: false)
}

func writeFrame(_ payload: Data, to handle: FileHandle) throws {
  var length = UInt64(payload.count).littleEndian
  var frame = Data(bytes: &length, count: MemoryLayout<UInt64>.size)
  frame.append(payload)
  try handle.write(contentsOf: frame)
}

final class OrderedFrameWriter: @unchecked Sendable {
  private let condition = NSCondition()
  private let handle: FileHandle
  private var nextSequence = 0
  private var pending: [Int: Data] = [:]
  private var failure: (any Error)?

  init(handle: FileHandle) {
    self.handle = handle
  }

  func submit(sequence: Int, payload: Data) {
    condition.lock()
    if failure == nil {
      pending[sequence] = payload
      flushLocked()
    }
    condition.broadcast()
    condition.unlock()
  }

  func waitUntilWritten(_ count: Int) throws {
    condition.lock()
    while nextSequence < count, failure == nil {
      condition.wait()
    }
    let observedFailure = failure
    condition.unlock()
    if let observedFailure {
      throw observedFailure
    }
  }

  private func flushLocked() {
    while let payload = pending.removeValue(forKey: nextSequence) {
      do {
        try writeFrame(payload, to: handle)
        nextSequence += 1
      } catch {
        failure = error
        pending.removeAll()
        return
      }
    }
  }
}

do {
  let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
  let shape = try PaishoExecutionShape(
    batchSize: options.batchSize,
    legalActionCapacity: options.legalActions
  )
  let maximumPayload = max(
    max(
      max(PaishoWeightsWire.maximumPayloadSize, PaishoCheckpointWireV1.maximumRequestPayloadSize),
      PaishoTrainingCycleWireV1.maximumRequestPayloadSize),
    max(
      try PaishoInferenceWireV1.maximumRequestPayloadSize(shape: shape),
      max(
        try PaishoTrainingWireV1.maximumRequestPayloadSize(shape: shape),
        try PaishoTerminalPpoWireV1.maximumRequestPayloadSize(shape: shape)
      )
    )
  )
  let checkpoint = try options.checkpointPath.map {
    try PaishoTrainingCheckpoint.read(from: URL(fileURLWithPath: $0))
  }
  if options.checkpointMode == .newGeneration, checkpoint == nil {
    throw ArgumentError.newGenerationWithoutCheckpoint
  }
  if let checkpoint, let checkpointPath = options.checkpointPath,
    checkpoint.metadata.configuration != options.preset.configuration
  {
    throw ArgumentError.checkpointPresetMismatch(checkpointPath)
  }
  let model = try PaishoMPSGraphModel(
    configuration: options.preset.configuration,
    executionShape: shape,
    optimization: options.optimization,
    seed: options.seed,
    trainingStep: checkpoint?.metadata.trainingStep ?? 0,
    restoredParameters: checkpoint?.parameters ?? []
  )
  let input = FileHandle.standardInput
  let output = FileHandle.standardOutput
  let responseWriter = OrderedFrameWriter(handle: output)
  var inferencePipeline: PaishoServingInferencePipeline?
  var responseSequence = 0
  let beginsNewGeneration = options.checkpointMode == .newGeneration
  var expectedReplaySnapshotSHA256: String? =
    beginsNewGeneration ? nil : checkpoint?.metadata.progress.replaySnapshotSHA256
  var expectedReplayIndex: UInt64? =
    beginsNewGeneration ? 0 : checkpoint?.metadata.progress.replayIndex
  var expectedLearningRate: Float? =
    beginsNewGeneration ? nil : checkpoint?.metadata.progress.scheduler.learningRate
  // Legacy PST training keeps progress here; PSG must not leave stale progress in the model.
  var latestGeneration: UInt64? = checkpoint?.metadata.progress.generation
  var transitionedGeneration: UInt64?

  while try autoreleasepool(invoking: {
    guard let payload = try readFrame(from: input, maximum: maximumPayload) else {
      return false
    }
    let requestKind = WireRequestKind(payload: payload)
    let requestID = requestKind.requestID(in: payload)
    let sequence = responseSequence
    responseSequence += 1
    if case .inference = requestKind {
      do {
        let request = try PaishoInferenceWireV1.decodeRequest(
          payload,
          expectedShape: shape
        )
        if inferencePipeline == nil {
          inferencePipeline = try model.makeServingInferencePipeline(
            slotCount: options.inferenceSlots
          )
        }
        try inferencePipeline!.submit(request.batch) { result in
          let response: Data
          do {
            response = try PaishoInferenceWireV1.encodeResponse(
              requestID: request.requestID,
              result: result.get(),
              shape: shape
            )
          } catch {
            response = requestKind.encodeError(requestID: requestID, error: error)
          }
          responseWriter.submit(sequence: sequence, payload: response)
        }
      } catch {
        let response = requestKind.encodeError(requestID: requestID, error: error)
        responseWriter.submit(sequence: sequence, payload: response)
      }
      return true
    }

    inferencePipeline?.waitUntilIdle()
    try responseWriter.waitUntilWritten(sequence)
    switch requestKind {
    case .training, .terminalPpoTraining: inferencePipeline = nil
    case .checkpoint, .inference, .trainingCycle, .weights: break
    }
    let response: Data
    do {
      switch requestKind {
      case .weights:
        let request = try PaishoWeightsWire.decodeRequest(payload)
        if let snapshot = request.snapshot, let packet = request.packet {
          try model.importWeights(snapshot)
          inferencePipeline = nil
          expectedReplaySnapshotSHA256 = nil
          expectedReplayIndex = nil
          expectedLearningRate = nil
          latestGeneration = nil
          transitionedGeneration = nil
          response = PaishoWeightsWire.importResponse(
            id: request.id, step: snapshot.trainingStep, packet: packet)
        } else {
          response = PaishoWeightsWire.exportResponse(
            id: request.id, packet: try model.exportWeights().encode())
        }
      case .trainingCycle:
        let request = try PaishoTrainingCycleWireV1.decodeRequest(payload)
        try request.validateTransition(
          trainingStep: model.trainingStep,
          previousSnapshotSHA256: expectedReplaySnapshotSHA256,
          previousGeneration: latestGeneration)
        // Encode before mutation too: errors must preserve the previous binding.
        response = try PaishoTrainingCycleWireV1.encodeResponse(
          requestID: request.requestID, progress: request.nextProgress)
        expectedReplaySnapshotSHA256 = request.nextProgress.replaySnapshotSHA256
        expectedReplayIndex = request.nextProgress.replayIndex
        expectedLearningRate = request.nextProgress.scheduler.learningRate
        latestGeneration = request.nextProgress.generation
        transitionedGeneration = request.nextProgress.generation
      case .training:
        let request = try PaishoTrainingWireV1.decodeRequest(
          payload,
          expectedShape: shape
        )
        guard request.expectedTrainingStep == model.trainingStep else {
          throw PaishoTrainingWireError.trainingStepMismatch(
            expected: request.expectedTrainingStep,
            actual: model.trainingStep
          )
        }
        if let expectedReplaySnapshotSHA256,
          request.replaySnapshotSHA256 != expectedReplaySnapshotSHA256
        {
          throw PaishoTrainingWireError.replaySnapshotMismatch(
            expected: expectedReplaySnapshotSHA256,
            actual: request.replaySnapshotSHA256
          )
        }
        if let expectedReplayIndex, request.startReplayIndex != expectedReplayIndex {
          throw PaishoTrainingWireError.replayIndexMismatch(
            expected: expectedReplayIndex,
            actual: request.startReplayIndex
          )
        }
        if let expectedLearningRate, request.learningRate != expectedLearningRate {
          throw PaishoTrainingWireError.learningRateMismatch(
            expected: expectedLearningRate,
            actual: request.learningRate
          )
        }
        let result = try model.train(request.batch, learningRate: request.learningRate)
        expectedReplaySnapshotSHA256 = request.replaySnapshotSHA256
        expectedReplayIndex = request.nextReplayIndex
        expectedLearningRate = request.learningRate
        response = PaishoTrainingWireV1.encodeResponse(
          requestID: request.requestID,
          completedReplayIndex: request.nextReplayIndex,
          result: result
        )
      case .terminalPpoTraining:
        let request = try PaishoTerminalPpoWireV1.decodeRequest(
          payload,
          expectedShape: shape
        )
        guard request.expectedTrainingStep == model.trainingStep else {
          throw PaishoTerminalPpoWireError.trainingStepMismatch(
            expected: request.expectedTrainingStep,
            actual: model.trainingStep
          )
        }
        if let expectedReplaySnapshotSHA256,
          request.replaySnapshotSHA256 != expectedReplaySnapshotSHA256
        {
          throw PaishoTerminalPpoWireError.replaySnapshotMismatch(
            expected: expectedReplaySnapshotSHA256,
            actual: request.replaySnapshotSHA256
          )
        }
        if let expectedReplayIndex, request.startReplayIndex != expectedReplayIndex {
          throw PaishoTerminalPpoWireError.replayIndexMismatch(
            expected: expectedReplayIndex,
            actual: request.startReplayIndex
          )
        }
        if let expectedLearningRate, request.learningRate != expectedLearningRate {
          throw PaishoTerminalPpoWireError.learningRateMismatch(
            expected: expectedLearningRate,
            actual: request.learningRate
          )
        }
        let result = try model.trainTerminalPpo(
          request.batch,
          learningRate: request.learningRate,
          parameters: request.parameters
        )
        expectedReplaySnapshotSHA256 = request.replaySnapshotSHA256
        expectedReplayIndex = request.nextReplayIndex
        expectedLearningRate = request.learningRate
        response = PaishoTerminalPpoWireV1.encodeResponse(
          requestID: request.requestID,
          completedReplayIndex: request.nextReplayIndex,
          result: result
        )
      case .checkpoint:
        let request = try PaishoCheckpointWireV1.decodeRequest(payload)
        if let generation = transitionedGeneration, request.generation != generation {
          throw PaishoTrainingCycleWireError.checkpointGenerationMismatch(
            expected: generation, actual: request.generation)
        }
        guard request.expectedTrainingStep == model.trainingStep else {
          throw PaishoCheckpointWireError.trainingStepMismatch(
            expected: request.expectedTrainingStep,
            actual: model.trainingStep
          )
        }
        if let expectedReplaySnapshotSHA256,
          request.replaySnapshotSHA256 != expectedReplaySnapshotSHA256
        {
          throw PaishoCheckpointWireError.replaySnapshotMismatch(
            expected: expectedReplaySnapshotSHA256,
            actual: request.replaySnapshotSHA256
          )
        }
        if let expectedReplayIndex, request.replayIndex != expectedReplayIndex {
          throw PaishoCheckpointWireError.replayIndexMismatch(
            expected: expectedReplayIndex,
            actual: request.replayIndex
          )
        }
        if let expectedLearningRate, request.learningRate != expectedLearningRate {
          throw PaishoCheckpointWireError.learningRateMismatch(
            expected: expectedLearningRate,
            actual: request.learningRate
          )
        }
        let scheduler = try PaishoLearningRateScheduler(
          learningRate: request.learningRate,
          completedSteps: model.trainingStep
        )
        let progress = try PaishoTrainingProgress(
          generation: request.generation,
          replayIndex: request.replayIndex,
          replaySnapshotSHA256: request.replaySnapshotSHA256,
          scheduler: scheduler,
          randomStates: request.randomStates
        )
        let contentSHA256 = try model.writeCheckpointIdempotently(
          to: URL(fileURLWithPath: request.destinationPath),
          progress: progress
        )
        expectedReplaySnapshotSHA256 = request.replaySnapshotSHA256
        expectedReplayIndex = request.replayIndex
        expectedLearningRate = request.learningRate
        response = try PaishoCheckpointWireV1.encodeResponse(
          requestID: request.requestID,
          completedTrainingStep: model.trainingStep,
          completedReplayIndex: request.replayIndex,
          contentSHA256: contentSHA256
        )
        latestGeneration = request.generation
      case .inference:
        let request = try PaishoInferenceWireV1.decodeRequest(
          payload,
          expectedShape: shape
        )
        let result = try model.servingInference(request.batch)
        response = try PaishoInferenceWireV1.encodeResponse(
          requestID: request.requestID,
          result: result,
          shape: shape
        )
      }
    } catch {
      response = requestKind.encodeError(requestID: requestID, error: error)
    }
    responseWriter.submit(sequence: sequence, payload: response)
    try responseWriter.waitUntilWritten(sequence + 1)
    return true
  }) {}
  inferencePipeline?.waitUntilIdle()
  try responseWriter.waitUntilWritten(responseSequence)
} catch {
  FileHandle.standardError.write(Data("error: \(error)\n".utf8))
  exit(2)
}
