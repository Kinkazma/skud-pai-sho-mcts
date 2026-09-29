import Foundation
import Metal
import MetalPerformanceShadersGraph

public struct PaishoInferenceResult: Sendable {
  public let legalLogits: [Float]
  public let policyProbabilities: [Float]
  public let valueLogits: [Float]
  public let valueProbabilities: [Float]
}

public struct PaishoTrainingResult: Sendable {
  public let step: UInt64
  public let policyLoss: Float
  public let valueLoss: Float
  public let totalLoss: Float
}

public struct PaishoTerminalPpoResult: Sendable {
  public let step: UInt64
  public let policyLoss: Float
  public let valueLoss: Float
  public let entropy: Float
  public let totalLoss: Float
  public let meanAdvantage: Float
  public let meanImportanceRatio: Float
  public let meanSquaredRatioDeviation: Float
}

public struct PaishoTerminalPpoTiming: Sendable {
  public let validationSeconds: Double
  public let feedsSeconds: Double
  // Wall time: graph submission, runtime work and waiting, NOT GPU kernel time.
  public let executionSeconds: Double
  public let readbackSeconds: Double
}

public enum PaishoTerminalPpoProbe { case loss, gradients }

public final class PaishoMPSGraphModel {
  public let configuration: PaishoNetworkConfiguration
  public let executionShape: PaishoExecutionShape
  public let optimization: PaishoGraphOptimization
  public let metalDeviceName: String
  public private(set) var trainingStep: UInt64
  // Owned by the cycle-aware API; legacy batch APIs remain available to the service.
  var boundTrainingProgress: PaishoTrainingProgress?

  private let graph: MPSGraph
  private let device: MPSGraphDevice
  private let metalDevice: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let artifacts: GraphArtifacts
  private var servingExecution: PaishoInferenceExecution?
  // Diagnostic only until parity and throughput have been measured.
  public var useCompiledTerminalPpo = false
  public var profileTerminalPpo = false
  public private(set) var lastTerminalPpoTiming: PaishoTerminalPpoTiming?
  private var terminalGradientProbe: MPSGraphTensor?
  private var terminalExecution: PaishoTerminalExecution?
  private var weightImportFeeds: [MPSGraphTensor] = []
  private var weightImportOperations: [MPSGraphOperation] = []

  public init(
    configuration: PaishoNetworkConfiguration,
    executionShape: PaishoExecutionShape,
    optimization: PaishoGraphOptimization,
    seed: UInt64 = 1,
    trainingStep: UInt64 = 0,
    restoredParameters: [PaishoParameterSnapshot] = []
  ) throws {
    guard let metalDevice = MTLCreateSystemDefaultDevice(),
      let commandQueue = metalDevice.makeCommandQueue()
    else {
      throw PaishoMPSGraphError.metalUnavailable
    }
    let graph = MPSGraph()
    graph.options = .synchronizeResults
    let builder = try PaishoGraphBuilder(
      graph: graph,
      configuration: configuration,
      executionShape: executionShape,
      seed: seed,
      restored: restoredParameters
    )
    let builtArtifacts = try builder.build()
    if trainingStep > 0 {
      let restoredNames = Set(restoredParameters.map(\.name))
      if let missing = builtArtifacts.parameters.map(\.name).first(where: {
        !restoredNames.contains($0)
      }) {
        throw PaishoMPSGraphError.missingCheckpointParameter(missing)
      }
    }

    self.configuration = configuration
    self.executionShape = executionShape
    self.optimization = optimization
    metalDeviceName = metalDevice.name
    self.trainingStep = trainingStep
    self.graph = graph
    device = MPSGraphDevice(mtlDevice: metalDevice)
    self.metalDevice = metalDevice
    self.commandQueue = commandQueue
    artifacts = builtArtifacts
    servingExecution = nil
  }

  public func inference(_ batch: PaishoInferenceBatch) throws -> PaishoInferenceResult {
    try requireShape(batch)
    let outputs = artifacts.outputs
    let result = graph.runAsync(
      with: commandQueue,
      feeds: inferenceFeeds(batch),
      targetTensors: [
        outputs.legalLogits,
        outputs.policyProbabilities,
        outputs.valueLogits,
        outputs.valueProbabilities,
      ],
      targetOperations: nil,
      executionDescriptor: executionDescriptor()
    )
    let actionCount = executionShape.batchSize * executionShape.legalActionCapacity
    let valueCount = executionShape.batchSize * PaishoTensorSchemaV1.valueClasses
    return PaishoInferenceResult(
      legalLogits: try readFloats(result[outputs.legalLogits]!, count: actionCount),
      policyProbabilities: try readFloats(
        result[outputs.policyProbabilities]!, count: actionCount
      ),
      valueLogits: try readFloats(result[outputs.valueLogits]!, count: valueCount),
      valueProbabilities: try readFloats(
        result[outputs.valueProbabilities]!, count: valueCount
      )
    )
  }

  public func inference(_ batch: PaishoTrainingBatch) throws -> PaishoInferenceResult {
    try inference(batch.inference)
  }

  public func inference(_ batch: PaishoTerminalPpoBatch) throws -> PaishoInferenceResult {
    try inference(batch.inference)
  }

  public func servingInference(
    _ batch: PaishoInferenceBatch
  ) throws -> PaishoServingInferenceResult {
    try requireShape(batch)
    let execution: PaishoInferenceExecution
    if let existing = servingExecution {
      execution = existing
    } else {
      let compiled = try makeServingExecution(slotCount: 1)
      servingExecution = compiled
      execution = compiled
    }
    return try execution.run(batch)
  }

  public func makeServingInferencePipeline(
    slotCount: Int
  ) throws -> PaishoServingInferencePipeline {
    PaishoServingInferencePipeline(
      execution: try makeServingExecution(slotCount: slotCount),
      slotCount: slotCount
    )
  }

  public func train(
    _ batch: PaishoTrainingBatch,
    learningRate: Float
  ) throws -> PaishoTrainingResult {
    try batch.validate()
    try requireShape(batch.inference)
    guard learningRate.isFinite, learningRate > 0 else {
      throw PaishoMPSGraphError.invalidLearningRate
    }
    guard trainingStep < UInt64.max else {
      throw PaishoMPSGraphError.trainingStepOverflow
    }
    servingExecution = nil
    let nextStep = trainingStep + 1
    var feeds = inferenceFeeds(batch.inference)
    let inputs = artifacts.inputs
    feeds[inputs.policyTargets] = tensorData(
      batch.policyTargets,
      shape: [executionShape.batchSize, executionShape.legalActionCapacity],
      device: device
    )
    feeds[inputs.valueTargets] = tensorData(
      batch.valueTargets,
      shape: [executionShape.batchSize, PaishoTensorSchemaV1.valueClasses],
      device: device
    )
    feeds[inputs.learningRate] = tensorData([learningRate], shape: [1], device: device)
    feeds[inputs.beta1Power] = tensorData(
      [pow(0.9, Float(nextStep))], shape: [1], device: device
    )
    feeds[inputs.beta2Power] = tensorData(
      [pow(0.999, Float(nextStep))], shape: [1], device: device
    )

    let outputs = artifacts.outputs
    let result = graph.runAsync(
      with: commandQueue,
      feeds: feeds,
      targetTensors: [outputs.policyLoss, outputs.valueLoss, outputs.totalLoss],
      targetOperations: artifacts.updateOperations,
      executionDescriptor: executionDescriptor()
    )
    trainingStep = nextStep
    return PaishoTrainingResult(
      step: nextStep,
      policyLoss: try readFloats(result[outputs.policyLoss]!, count: 1)[0],
      valueLoss: try readFloats(result[outputs.valueLoss]!, count: 1)[0],
      totalLoss: try readFloats(result[outputs.totalLoss]!, count: 1)[0]
    )
  }

  public func trainTerminalPpo(
    _ batch: PaishoTerminalPpoBatch,
    learningRate: Float,
    parameters: PaishoTerminalPpoParametersV1
  ) throws -> PaishoTerminalPpoResult {
    try executeTerminalPpo(batch, learningRate: learningRate, parameters: parameters, probe: nil)
  }

  /// Diagnostic only: evaluates loss or all gradients WITHOUT changing weights or Adam.
  public func probeTerminalPpo(
    _ batch: PaishoTerminalPpoBatch,
    parameters: PaishoTerminalPpoParametersV1,
    probe: PaishoTerminalPpoProbe
  ) throws -> PaishoTerminalPpoResult {
    try executeTerminalPpo(batch, learningRate: 1e-4, parameters: parameters, probe: probe)
  }

  private func executeTerminalPpo(
    _ batch: PaishoTerminalPpoBatch,
    learningRate: Float,
    parameters: PaishoTerminalPpoParametersV1,
    probe: PaishoTerminalPpoProbe?
  ) throws -> PaishoTerminalPpoResult {
    let profiling = profileTerminalPpo
    func stamp() -> UInt64 { profiling ? DispatchTime.now().uptimeNanoseconds : 0 }
    lastTerminalPpoTiming = nil
    let started = stamp()
    try batch.validate()
    try requireShape(batch.inference)
    guard learningRate.isFinite, learningRate > 0 else {
      throw PaishoMPSGraphError.invalidLearningRate
    }
    guard trainingStep < UInt64.max else {
      throw PaishoMPSGraphError.trainingStepOverflow
    }
    servingExecution = nil
    let nextStep = trainingStep + 1
    let validated = stamp()
    var feeds = inferenceFeeds(batch.inference)
    let inputs = artifacts.inputs
    feeds[inputs.playedActionMask] = tensorData(
      batch.playedActionMask,
      shape: [executionShape.batchSize, executionShape.legalActionCapacity],
      device: device
    )
    feeds[inputs.behaviorProbabilities] = tensorData(
      batch.behaviorProbabilities,
      shape: [executionShape.batchSize],
      device: device
    )
    feeds[inputs.advantages] = tensorData(
      batch.advantages,
      shape: [executionShape.batchSize],
      device: device
    )
    feeds[inputs.policyTemperature] = tensorData(
      [parameters.policyTemperature], shape: [1], device: device
    )
    feeds[inputs.uniformMix] = tensorData(
      [parameters.uniformMix], shape: [1], device: device
    )
    feeds[inputs.valueTargets] = tensorData(
      batch.valueTargets,
      shape: [executionShape.batchSize, PaishoTensorSchemaV1.valueClasses],
      device: device
    )
    feeds[inputs.clipEpsilon] = tensorData(
      [parameters.clipEpsilon], shape: [1], device: device
    )
    feeds[inputs.valueLossWeight] = tensorData(
      [parameters.valueLossWeight], shape: [1], device: device
    )
    feeds[inputs.entropyWeight] = tensorData(
      [parameters.entropyWeight], shape: [1], device: device
    )
    feeds[inputs.learningRate] = tensorData([learningRate], shape: [1], device: device)
    feeds[inputs.beta1Power] = tensorData(
      [pow(0.9, Float(nextStep))], shape: [1], device: device
    )
    feeds[inputs.beta2Power] = tensorData(
      [pow(0.999, Float(nextStep))], shape: [1], device: device
    )

    let outputs = artifacts.outputs
    var targets = [
        outputs.terminalPolicyLoss,
        outputs.valueLoss,
        outputs.policyEntropy,
        outputs.terminalTotalLoss,
        outputs.meanAdvantage,
        outputs.meanImportanceRatio,
        outputs.meanSquaredRatioDeviation,
      ]
    if probe == .gradients {
      if terminalGradientProbe == nil {
        let parameters = artifacts.parameters.map(\.values)
        let gradients = graph.gradients(of: outputs.terminalTotalLoss, with: parameters,
                                       name: "diagnostic/ppo_gradients")
        // A scalar checksum forces every gradient to execute without copying 19 MB
        // of gradients to the CPU. Reduction cost is included in the measurement.
        let sums = parameters.map { parameter in
          let gradient = gradients[parameter]!
          return graph.reductionSum(with: gradient, axes: nil, name: nil)
        }
        terminalGradientProbe = sums.dropFirst().reduce(sums[0]) {
          graph.addition($0, $1, name: nil)
        }
      }
      targets.append(terminalGradientProbe!)
    }
    let result: [MPSGraphTensor: MPSGraphTensorData]
    let prepared = stamp()
    if useCompiledTerminalPpo && probe == nil {
      if terminalExecution == nil {
        terminalExecution = try PaishoTerminalExecution(
          graph: graph, device: device, optimization: optimization,
          feeds: feeds, targets: targets, operations: artifacts.terminalUpdateOperations)
      }
      result = try terminalExecution!.run(queue: commandQueue, feeds: feeds)
    } else {
      result = graph.runAsync(
        with: commandQueue, feeds: feeds, targetTensors: targets,
        targetOperations: probe == nil ? artifacts.terminalUpdateOperations : nil,
        executionDescriptor: executionDescriptor())
    }
    let executed = stamp()
    if probe == nil { trainingStep = nextStep }
    let metrics = PaishoTerminalPpoResult(
      step: trainingStep,
      policyLoss: try readFloats(result[outputs.terminalPolicyLoss]!, count: 1)[0],
      valueLoss: try readFloats(result[outputs.valueLoss]!, count: 1)[0],
      entropy: try readFloats(result[outputs.policyEntropy]!, count: 1)[0],
      totalLoss: try readFloats(result[outputs.terminalTotalLoss]!, count: 1)[0],
      meanAdvantage: try readFloats(result[outputs.meanAdvantage]!, count: 1)[0],
      meanImportanceRatio: try readFloats(
        result[outputs.meanImportanceRatio]!, count: 1
      )[0],
      meanSquaredRatioDeviation: try readFloats(
        result[outputs.meanSquaredRatioDeviation]!, count: 1
      )[0]
    )
    if profiling {
      let finished = stamp()
      lastTerminalPpoTiming = PaishoTerminalPpoTiming(
        validationSeconds: Double(validated - started) / 1e9,
        feedsSeconds: Double(prepared - validated) / 1e9,
        executionSeconds: Double(executed - prepared) / 1e9,
        readbackSeconds: Double(finished - executed) / 1e9)
    }
    return metrics
  }

  public func train(
    _ batch: PaishoTrainingBatch,
    scheduler: inout PaishoLearningRateScheduler
  ) throws -> PaishoTrainingResult {
    guard scheduler.completedSteps == trainingStep else {
      throw PaishoCheckpointError.schedulerStepMismatch(
        model: trainingStep,
        scheduler: scheduler.completedSteps
      )
    }
    let result = try train(batch, learningRate: scheduler.learningRate)
    try scheduler.didCompleteStep()
    return result
  }

  public func checkpoint(progress: PaishoTrainingProgress) throws -> PaishoTrainingCheckpoint {
    let metadata = try PaishoCheckpointMetadata(
      configuration: configuration,
      executionShape: executionShape,
      optimization: optimization,
      trainingStep: trainingStep,
      progress: progress
    )
    return try PaishoTrainingCheckpoint(
      metadata: metadata,
      parameters: snapshotParameters()
    )
  }

  @discardableResult
  public func writeCheckpoint(
    to destination: URL,
    progress: PaishoTrainingProgress
  ) throws -> [UInt8] {
    try checkpoint(progress: progress).write(to: destination)
  }

  @discardableResult
  public func writeCheckpointIdempotently(
    to destination: URL,
    progress: PaishoTrainingProgress
  ) throws -> [UInt8] {
    try checkpoint(progress: progress).writeIdempotently(to: destination)
  }

  public static func restoringCheckpoint(
    from source: URL,
    replaySnapshotSHA256 expectedReplaySnapshotSHA256: String
  ) throws -> (model: PaishoMPSGraphModel, progress: PaishoTrainingProgress) {
    let checkpoint = try PaishoTrainingCheckpoint.read(from: source)
    let metadata = checkpoint.metadata
    guard metadata.progress.replaySnapshotSHA256 == expectedReplaySnapshotSHA256 else {
      throw PaishoCheckpointError.replaySnapshotMismatch(
        expected: expectedReplaySnapshotSHA256,
        actual: metadata.progress.replaySnapshotSHA256
      )
    }
    let model = try PaishoMPSGraphModel(
      configuration: metadata.configuration,
      executionShape: metadata.executionShape,
      optimization: metadata.optimization,
      trainingStep: metadata.trainingStep,
      restoredParameters: checkpoint.parameters
    )
    model.boundTrainingProgress = metadata.progress
    return (model, metadata.progress)
  }

  /// Values only: never reads Adam tensors or writes a checkpoint file.
  public func exportWeights() throws -> PaishoWeightSnapshot {
    let result = graph.run(
      with: commandQueue, feeds: [:],
      targetTensors: artifacts.parameters.map(\.values), targetOperations: nil)
    return PaishoWeightSnapshot(
      configuration: configuration, trainingStep: trainingStep,
      parameters: try artifacts.parameters.map {
        PaishoWeightArray(
          name: $0.name, shape: $0.shape,
          values: try readFloats(result[$0.values]!, count: $0.count))
      })
  }

  /// Inference replacement, not an Adam restore. Caller must drain and discard
  /// external serving pipelines. Existing graph shapes and Adam moments are retained.
  public func importWeights(_ snapshot: PaishoWeightSnapshot) throws {
    guard snapshot.configuration == configuration else {
      throw WeightsWireError.configurationMismatch
    }
    guard snapshot.parameters.count == artifacts.parameters.count else {
      throw WeightsWireError.parameterMismatch
    }
    var incoming: [String: PaishoWeightArray] = [:]
    for parameter in snapshot.parameters {
      guard incoming.updateValue(parameter, forKey: parameter.name) == nil else {
        throw WeightsWireError.parameterMismatch
      }
    }
    let ordered = try artifacts.parameters.map { parameter -> PaishoWeightArray in
      guard let value = incoming[parameter.name], value.shape == parameter.shape,
        value.values.count == parameter.count, value.values.allSatisfy(\.isFinite)
      else { throw WeightsWireError.parameterMismatch }
      return value
    }
    if weightImportFeeds.isEmpty {
      for parameter in artifacts.parameters {
        let feed = graph.placeholder(
          shape: parameter.shape.map(NSNumber.init(value:)),
          dataType: .float32, name: "\(parameter.name)/weight_import")
        weightImportFeeds.append(feed)
        weightImportOperations.append(
          graph.assign(
            parameter.values, tensor: feed,
            name: "\(parameter.name)/assign_weight_import"))
      }
    }
    let feeds = Dictionary(
      uniqueKeysWithValues: zip(weightImportFeeds, ordered).map {
        ($0.0, tensorData($0.1.values, shape: $0.1.shape, device: device))
      })
    _ = graph.run(
      with: commandQueue, feeds: feeds, targetTensors: [],
      targetOperations: weightImportOperations)
    servingExecution = nil
    trainingStep = snapshot.trainingStep
    boundTrainingProgress = nil
  }

  public func snapshotParameters() throws -> [PaishoParameterSnapshot] {
    let tensors = artifacts.parameters.flatMap { [$0.values, $0.momentum, $0.velocity] }
    // Checkpoint publication is already a blocking boundary. Using the synchronous
    // entry point here also avoids asking MPSGraph to clone an asynchronous
    // executable containing every parameter tensor solely for serialization.
    let result = graph.run(
      with: commandQueue,
      feeds: [:],
      targetTensors: tensors,
      targetOperations: nil
    )
    return try artifacts.parameters.map { parameter in
      PaishoParameterSnapshot(
        name: parameter.name,
        shape: parameter.shape,
        values: try readFloats(result[parameter.values]!, count: parameter.count),
        momentum: try readFloats(result[parameter.momentum]!, count: parameter.count),
        velocity: try readFloats(result[parameter.velocity]!, count: parameter.count)
      )
    }
  }

  private func requireShape(_ batch: PaishoInferenceBatch) throws {
    try batch.validate()
    guard batch.shape == executionShape else {
      throw PaishoMPSGraphError.batchShapeMismatch(
        expected: executionShape,
        actual: batch.shape
      )
    }
  }

  private func inferenceFeeds(
    _ batch: PaishoInferenceBatch
  ) -> [MPSGraphTensor: MPSGraphTensorData] {
    let inputs = artifacts.inputs
    let batchSize = executionShape.batchSize
    let actionCapacity = executionShape.legalActionCapacity
    let actionShape = [batchSize, actionCapacity]
    return [
      inputs.spatial: tensorData(
        batch.spatial,
        shape: [
          batchSize,
          PaishoTensorSchemaV1.boardSize,
          PaishoTensorSchemaV1.boardSize,
          PaishoTensorSchemaV1.spatialChannels,
        ],
        device: device
      ),
      inputs.global: tensorData(
        batch.global,
        shape: [batchSize, PaishoTensorSchemaV1.globalFeatures],
        device: device
      ),
      inputs.familyIndices: tensorData(
        batch.familyIndices, shape: actionShape, device: device
      ),
      inputs.tileIndices: tensorData(batch.tileIndices, shape: actionShape, device: device),
      inputs.tilePresence: tensorData(
        batch.tilePresence, shape: actionShape, device: device
      ),
      inputs.destinationIndices: tensorData(
        batch.destinationIndices, shape: actionShape, device: device
      ),
      inputs.destinationPresence: tensorData(
        batch.destinationPresence, shape: actionShape, device: device
      ),
      inputs.pairIndices: tensorData(batch.pairIndices, shape: actionShape, device: device),
      inputs.pairPresence: tensorData(
        batch.pairPresence, shape: actionShape, device: device
      ),
      inputs.legalMask: tensorData(batch.legalMask, shape: actionShape, device: device),
    ]
  }

  private func executionDescriptor() -> MPSGraphExecutionDescriptor {
    let compilation = MPSGraphCompilationDescriptor()
    compilation.optimizationLevel = optimization.mpsGraphValue
    compilation.waitForCompilationCompletion = true
    let execution = MPSGraphExecutionDescriptor()
    execution.compilationDescriptor = compilation
    execution.waitUntilCompleted = true
    return execution
  }

  private func makeServingExecution(slotCount: Int) throws -> PaishoInferenceExecution {
    try PaishoInferenceExecution(
      graph: graph,
      artifacts: artifacts,
      executionShape: executionShape,
      optimization: optimization,
      metalDevice: metalDevice,
      commandQueue: commandQueue,
      slotCount: slotCount
    )
  }
}

public enum PaishoMPSGraphError: Error, CustomStringConvertible {
  case metalUnavailable
  case invalidLearningRate
  case trainingStepOverflow
  case batchShapeMismatch(expected: PaishoExecutionShape, actual: PaishoExecutionShape)
  case duplicateParameter(String)
  case duplicateCheckpointParameter(String)
  case checkpointShapeMismatch(String)
  case nonFiniteCheckpointParameter(String)
  case unusedCheckpointParameter(String)
  case missingCheckpointParameter(String)
  case parameterCountMismatch(expected: Int, actual: Int)

  public var description: String {
    switch self {
    case .metalUnavailable: "Metal is unavailable"
    case .invalidLearningRate: "learning rate must be finite and positive"
    case .trainingStepOverflow: "training step overflow"
    case .batchShapeMismatch(let expected, let actual):
      "batch shape \(actual) does not match graph shape \(expected)"
    case .duplicateParameter(let name): "duplicate graph parameter \(name)"
    case .duplicateCheckpointParameter(let name):
      "checkpoint contains duplicate parameter \(name)"
    case .checkpointShapeMismatch(let name): "checkpoint shape mismatch for \(name)"
    case .nonFiniteCheckpointParameter(let name):
      "checkpoint contains non-finite values for \(name)"
    case .unusedCheckpointParameter(let name): "checkpoint parameter \(name) is not in the graph"
    case .missingCheckpointParameter(let name):
      "checkpoint is missing parameter \(name) for resumed training"
    case .parameterCountMismatch(let expected, let actual):
      "graph has \(actual) trainable values; configuration declares \(expected)"
    }
  }
}
