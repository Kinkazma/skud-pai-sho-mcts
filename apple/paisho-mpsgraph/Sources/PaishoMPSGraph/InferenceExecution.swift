import Foundation
import Metal
import MetalPerformanceShadersGraph

public struct PaishoServingInferenceResult: Sendable {
  public let policyProbabilities: [Float]
  public let valueProbabilities: [Float]
}

final class PaishoInferenceExecution {
  private let commandQueue: any MTLCommandQueue
  private let executable: MPSGraphExecutable
  private let slots: [PaishoInferenceSlot]

  init(
    graph: MPSGraph,
    artifacts: GraphArtifacts,
    executionShape: PaishoExecutionShape,
    optimization: PaishoGraphOptimization,
    metalDevice: any MTLDevice,
    commandQueue: any MTLCommandQueue,
    slotCount: Int = 1
  ) throws {
    guard slotCount > 0 else {
      throw PaishoInferenceExecutionError.invalidSlotCount(slotCount)
    }
    let createdSlots = try (0..<slotCount).map { _ in
      try PaishoInferenceSlot(
        artifacts: artifacts,
        executionShape: executionShape,
        metalDevice: metalDevice
      )
    }
    let template = createdSlots[0]
    let feedTypes = Dictionary(
      uniqueKeysWithValues: template.feedSlots.map { ($0.tensor, $0.shapedType) }
    )
    let compilationDescriptor = MPSGraphCompilationDescriptor()
    compilationDescriptor.optimizationLevel = optimization.mpsGraphValue
    compilationDescriptor.waitForCompilationCompletion = true
    let graphDevice = MPSGraphDevice(mtlDevice: metalDevice)
    let compiled = graph.compile(
      with: graphDevice,
      feeds: feedTypes,
      targetTensors: template.targetTensors,
      targetOperations: nil,
      compilationDescriptor: compilationDescriptor
    )

    guard let executableFeedTensors = compiled.feedTensors else {
      throw PaishoInferenceExecutionError.missingExecutableFeeds
    }
    guard let orderedTargetTensors = compiled.targetTensors else {
      throw PaishoInferenceExecutionError.missingExecutableTargets
    }
    guard executableFeedTensors.count == template.feedSlots.count else {
      throw PaishoInferenceExecutionError.executableFeedCount(
        expected: template.feedSlots.count,
        actual: executableFeedTensors.count
      )
    }
    guard orderedTargetTensors.count == template.targetTensors.count else {
      throw PaishoInferenceExecutionError.executableTargetCount(
        expected: template.targetTensors.count,
        actual: orderedTargetTensors.count
      )
    }
    for slot in createdSlots {
      try slot.order(
        feedTensors: executableFeedTensors,
        targetTensors: orderedTargetTensors
      )
    }
    compiled.specialize(
      with: graphDevice,
      inputTypes: template.orderedFeedSlots.map(\.shapedType),
      compilationDescriptor: compilationDescriptor
    )
    self.commandQueue = commandQueue
    executable = compiled
    slots = createdSlots
  }

  func run(_ batch: PaishoInferenceBatch) throws -> PaishoServingInferenceResult {
    let slot = slots[0]
    try slot.write(batch)
    let completedOutputs = executable.run(
      with: commandQueue,
      inputs: slot.orderedFeedSlots.map(\.tensorData),
      results: slot.orderedOutputSlots.map(\.tensorData),
      executionDescriptor: nil
    )
    try validateOutputCount(completedOutputs)
    let result = try slot.readResult()
    withExtendedLifetime(completedOutputs) {}
    return result
  }

  func runAsync(
    _ batch: PaishoInferenceBatch,
    slotIndex: Int,
    completion: @escaping @Sendable (Result<PaishoServingInferenceResult, any Error>) -> Void
  ) throws {
    guard slots.indices.contains(slotIndex) else {
      throw PaishoInferenceExecutionError.invalidSlotIndex(slotIndex)
    }
    let slot = slots[slotIndex]
    try slot.write(batch)
    let descriptor = MPSGraphExecutableExecutionDescriptor()
    descriptor.waitUntilCompleted = false
    descriptor.completionHandler = { completedOutputs, error in
      let result: Result<PaishoServingInferenceResult, any Error>
      if let error {
        result = .failure(error)
      } else {
        result = Result {
          try self.validateOutputCount(completedOutputs)
          return try slot.readResult()
        }
      }
      completion(result)
      withExtendedLifetime(completedOutputs) {}
      withExtendedLifetime(slot) {}
    }
    let submittedOutputs = executable.runAsync(
      with: commandQueue,
      inputs: slot.orderedFeedSlots.map(\.tensorData),
      results: slot.orderedOutputSlots.map(\.tensorData),
      executionDescriptor: descriptor
    )
    withExtendedLifetime(submittedOutputs) {}
  }

  private func validateOutputCount(_ outputs: [MPSGraphTensorData]) throws {
    guard outputs.count == 2 else {
      throw PaishoInferenceExecutionError.executableOutputCount(
        expected: 2,
        actual: outputs.count
      )
    }
  }
}

public final class PaishoServingInferencePipeline: @unchecked Sendable {
  public let slotCount: Int

  private let execution: PaishoInferenceExecution
  private let condition = NSCondition()
  private var availableSlots: [Int]
  private var inFlight = 0

  init(execution: PaishoInferenceExecution, slotCount: Int) {
    self.execution = execution
    self.slotCount = slotCount
    availableSlots = Array((0..<slotCount).reversed())
  }

  public func submit(
    _ batch: PaishoInferenceBatch,
    completion: @escaping @Sendable (Result<PaishoServingInferenceResult, any Error>) -> Void
  ) throws {
    let slotIndex = acquireSlot()
    do {
      try execution.runAsync(batch, slotIndex: slotIndex) { result in
        self.releaseSlot(slotIndex)
        completion(result)
      }
    } catch {
      releaseSlot(slotIndex)
      throw error
    }
  }

  public func waitUntilIdle() {
    condition.lock()
    while inFlight != 0 {
      condition.wait()
    }
    condition.unlock()
  }

  private func acquireSlot() -> Int {
    condition.lock()
    while availableSlots.isEmpty {
      condition.wait()
    }
    let slot = availableSlots.removeLast()
    inFlight += 1
    condition.unlock()
    return slot
  }

  private func releaseSlot(_ slot: Int) {
    condition.lock()
    availableSlots.append(slot)
    inFlight -= 1
    condition.broadcast()
    condition.unlock()
  }
}

private final class PaishoInferenceSlot {
  fileprivate let feedSlots: [PersistentMetalTensor]
  fileprivate let targetTensors: [MPSGraphTensor]
  fileprivate private(set) var orderedFeedSlots: [PersistentMetalTensor] = []
  fileprivate private(set) var orderedOutputSlots: [PersistentMetalTensor] = []

  private let spatial: PersistentMetalTensor
  private let global: PersistentMetalTensor
  private let familyIndices: PersistentMetalTensor
  private let tileIndices: PersistentMetalTensor
  private let tilePresence: PersistentMetalTensor
  private let destinationIndices: PersistentMetalTensor
  private let destinationPresence: PersistentMetalTensor
  private let pairIndices: PersistentMetalTensor
  private let pairPresence: PersistentMetalTensor
  private let legalMask: PersistentMetalTensor
  private let policyOutput: PersistentMetalTensor
  private let valueOutput: PersistentMetalTensor

  init(
    artifacts: GraphArtifacts,
    executionShape: PaishoExecutionShape,
    metalDevice: any MTLDevice
  ) throws {
    let batch = executionShape.batchSize
    let capacity = executionShape.legalActionCapacity
    let actionShape = [batch, capacity]
    let inputs = artifacts.inputs
    spatial = try PersistentMetalTensor(
      name: "spatial",
      tensor: inputs.spatial,
      shape: [
        batch,
        PaishoTensorSchemaV1.boardSize,
        PaishoTensorSchemaV1.boardSize,
        PaishoTensorSchemaV1.spatialChannels,
      ],
      dataType: .float32,
      metalDevice: metalDevice
    )
    global = try PersistentMetalTensor(
      name: "global",
      tensor: inputs.global,
      shape: [batch, PaishoTensorSchemaV1.globalFeatures],
      dataType: .float32,
      metalDevice: metalDevice
    )
    familyIndices = try PersistentMetalTensor(
      name: "familyIndices",
      tensor: inputs.familyIndices,
      shape: actionShape,
      dataType: .int32,
      metalDevice: metalDevice
    )
    tileIndices = try PersistentMetalTensor(
      name: "tileIndices",
      tensor: inputs.tileIndices,
      shape: actionShape,
      dataType: .int32,
      metalDevice: metalDevice
    )
    tilePresence = try PersistentMetalTensor(
      name: "tilePresence",
      tensor: inputs.tilePresence,
      shape: actionShape,
      dataType: .float32,
      metalDevice: metalDevice
    )
    destinationIndices = try PersistentMetalTensor(
      name: "destinationIndices",
      tensor: inputs.destinationIndices,
      shape: actionShape,
      dataType: .int32,
      metalDevice: metalDevice
    )
    destinationPresence = try PersistentMetalTensor(
      name: "destinationPresence",
      tensor: inputs.destinationPresence,
      shape: actionShape,
      dataType: .float32,
      metalDevice: metalDevice
    )
    pairIndices = try PersistentMetalTensor(
      name: "pairIndices",
      tensor: inputs.pairIndices,
      shape: actionShape,
      dataType: .int32,
      metalDevice: metalDevice
    )
    pairPresence = try PersistentMetalTensor(
      name: "pairPresence",
      tensor: inputs.pairPresence,
      shape: actionShape,
      dataType: .float32,
      metalDevice: metalDevice
    )
    legalMask = try PersistentMetalTensor(
      name: "legalMask",
      tensor: inputs.legalMask,
      shape: actionShape,
      dataType: .float32,
      metalDevice: metalDevice
    )
    let outputs = artifacts.outputs
    policyOutput = try PersistentMetalTensor(
      name: "policyOutput",
      tensor: outputs.policyProbabilities,
      shape: actionShape,
      dataType: .float32,
      metalDevice: metalDevice
    )
    valueOutput = try PersistentMetalTensor(
      name: "valueOutput",
      tensor: outputs.valueProbabilities,
      shape: [batch, PaishoTensorSchemaV1.valueClasses],
      dataType: .float32,
      metalDevice: metalDevice
    )
    feedSlots = [
      spatial,
      global,
      familyIndices,
      tileIndices,
      tilePresence,
      destinationIndices,
      destinationPresence,
      pairIndices,
      pairPresence,
      legalMask,
    ]
    targetTensors = [outputs.policyProbabilities, outputs.valueProbabilities]
  }

  func order(feedTensors: [MPSGraphTensor], targetTensors: [MPSGraphTensor]) throws {
    let feedsByTensor = Dictionary(uniqueKeysWithValues: feedSlots.map { ($0.tensor, $0) })
    orderedFeedSlots = try feedTensors.map { tensor in
      guard let slot = feedsByTensor[tensor] else {
        throw PaishoInferenceExecutionError.unknownExecutableFeed(
          String(describing: tensor)
        )
      }
      return slot
    }
    let outputSlots = [policyOutput, valueOutput]
    let outputsByTensor = Dictionary(uniqueKeysWithValues: outputSlots.map { ($0.tensor, $0) })
    orderedOutputSlots = try targetTensors.map { tensor in
      guard let slot = outputsByTensor[tensor] else {
        throw PaishoInferenceExecutionError.unknownExecutableTarget(
          String(describing: tensor)
        )
      }
      return slot
    }
  }

  func write(_ batch: PaishoInferenceBatch) throws {
    try spatial.write(batch.spatial)
    try global.write(batch.global)
    try familyIndices.write(batch.familyIndices)
    try tileIndices.write(batch.tileIndices)
    try tilePresence.write(batch.tilePresence)
    try destinationIndices.write(batch.destinationIndices)
    try destinationPresence.write(batch.destinationPresence)
    try pairIndices.write(batch.pairIndices)
    try pairPresence.write(batch.pairPresence)
    try legalMask.write(batch.legalMask)
  }

  func readResult() throws -> PaishoServingInferenceResult {
    PaishoServingInferenceResult(
      policyProbabilities: try policyOutput.readFloats(),
      valueProbabilities: try valueOutput.readFloats()
    )
  }
}

private final class PersistentMetalTensor {
  let tensor: MPSGraphTensor
  let shapedType: MPSGraphShapedType
  let tensorData: MPSGraphTensorData

  private let name: String
  private let buffer: any MTLBuffer
  private let elementCount: Int
  private let dataType: MPSDataType

  init(
    name: String,
    tensor: MPSGraphTensor,
    shape: [Int],
    dataType: MPSDataType,
    metalDevice: any MTLDevice
  ) throws {
    guard let lastDimension = shape.last else {
      throw PaishoInferenceExecutionError.invalidShape(name, shape)
    }
    let elementCount = try shape.reduce(1) { partial, dimension in
      let product = partial.multipliedReportingOverflow(by: dimension)
      guard dimension > 0, !product.overflow else {
        throw PaishoInferenceExecutionError.invalidShape(name, shape)
      }
      return product.partialValue
    }
    let bytesPerElement: Int
    switch dataType {
    case .float32, .int32: bytesPerElement = 4
    default: throw PaishoInferenceExecutionError.unsupportedDataType(name)
    }
    let denseByteCount = elementCount.multipliedReportingOverflow(by: bytesPerElement)
    let rowByteCount = lastDimension.multipliedReportingOverflow(by: bytesPerElement)
    guard !denseByteCount.overflow, !rowByteCount.overflow else {
      throw PaishoInferenceExecutionError.invalidShape(name, shape)
    }
    let alignedByteCount = denseByteCount.partialValue.addingReportingOverflow(15)
    guard !alignedByteCount.overflow else {
      throw PaishoInferenceExecutionError.invalidShape(name, shape)
    }
    let allocationLength = alignedByteCount.partialValue & ~15
    guard
      let buffer = metalDevice.makeBuffer(
        length: allocationLength,
        options: .storageModeShared
      )
    else {
      throw PaishoInferenceExecutionError.bufferAllocation(name, allocationLength)
    }
    let mpsShape = shape.map(NSNumber.init(value:))
    self.name = name
    self.tensor = tensor
    self.buffer = buffer
    self.elementCount = elementCount
    self.dataType = dataType
    shapedType = MPSGraphShapedType(shape: mpsShape, dataType: dataType)
    tensorData = MPSGraphTensorData(
      buffer,
      shape: mpsShape,
      dataType: dataType,
      rowBytes: rowByteCount.partialValue
    )
  }

  func write(_ values: [Float]) throws {
    guard dataType == .float32 else {
      throw PaishoInferenceExecutionError.unsupportedDataType(name)
    }
    try validateElementCount(values.count, stride: MemoryLayout<Float>.stride)
    let retainedBuffer = buffer
    let destination = retainedBuffer.contents()
    let byteCount = values.count * MemoryLayout<Float>.stride
    values.withUnsafeBufferPointer { source in
      guard let address = source.baseAddress else { return }
      destination.copyMemory(
        from: UnsafeRawPointer(address),
        byteCount: byteCount
      )
    }
    withExtendedLifetime(retainedBuffer) {}
  }

  func write(_ values: [Int32]) throws {
    guard dataType == .int32 else {
      throw PaishoInferenceExecutionError.unsupportedDataType(name)
    }
    try validateElementCount(values.count, stride: MemoryLayout<Int32>.stride)
    let retainedBuffer = buffer
    let destination = retainedBuffer.contents()
    let byteCount = values.count * MemoryLayout<Int32>.stride
    values.withUnsafeBufferPointer { source in
      guard let address = source.baseAddress else { return }
      destination.copyMemory(
        from: UnsafeRawPointer(address),
        byteCount: byteCount
      )
    }
    withExtendedLifetime(retainedBuffer) {}
  }

  func readFloats() throws -> [Float] {
    guard dataType == .float32 else {
      throw PaishoInferenceExecutionError.unsupportedDataType(name)
    }
    let retainedBuffer = buffer
    let source = retainedBuffer.contents()
    let byteCount = elementCount * MemoryLayout<Float>.stride
    let values = [Float](unsafeUninitializedCapacity: elementCount) { destination, count in
      guard let address = destination.baseAddress else {
        count = 0
        return
      }
      UnsafeMutableRawPointer(address).copyMemory(
        from: UnsafeRawPointer(source),
        byteCount: byteCount
      )
      count = elementCount
    }
    withExtendedLifetime(retainedBuffer) {}
    return values
  }

  private func validateElementCount(_ count: Int, stride: Int) throws {
    guard count == elementCount else {
      throw PaishoInferenceExecutionError.elementCount(
        name,
        expected: elementCount,
        actual: count
      )
    }
    let byteCount = count.multipliedReportingOverflow(by: stride)
    guard !byteCount.overflow, byteCount.partialValue <= buffer.length else {
      throw PaishoInferenceExecutionError.bufferCapacity(
        name,
        required: byteCount.partialValue,
        available: buffer.length
      )
    }
  }
}

enum PaishoInferenceExecutionError: Error, CustomStringConvertible {
  case invalidSlotCount(Int)
  case invalidSlotIndex(Int)
  case invalidShape(String, [Int])
  case unsupportedDataType(String)
  case bufferAllocation(String, Int)
  case bufferCapacity(String, required: Int, available: Int)
  case elementCount(String, expected: Int, actual: Int)
  case missingExecutableFeeds
  case missingExecutableTargets
  case unknownExecutableFeed(String)
  case unknownExecutableTarget(String)
  case executableFeedCount(expected: Int, actual: Int)
  case executableTargetCount(expected: Int, actual: Int)
  case executableOutputCount(expected: Int, actual: Int)

  var description: String {
    switch self {
    case .invalidSlotCount(let count):
      "inference execution needs at least one slot; received \(count)"
    case .invalidSlotIndex(let index): "invalid inference slot index \(index)"
    case .invalidShape(let name, let shape): "invalid shape for \(name): \(shape)"
    case .unsupportedDataType(let name): "unsupported data type for \(name)"
    case .bufferAllocation(let name, let bytes):
      "could not allocate \(bytes) shared Metal bytes for \(name)"
    case .bufferCapacity(let name, let required, let available):
      "\(name) requires \(required) buffer bytes; only \(available) are available"
    case .elementCount(let name, let expected, let actual):
      "\(name) has \(actual) elements; expected \(expected)"
    case .missingExecutableFeeds: "compiled inference executable has no feed ordering"
    case .missingExecutableTargets: "compiled inference executable has no target ordering"
    case .unknownExecutableFeed(let name): "compiled inference has unknown feed \(name)"
    case .unknownExecutableTarget(let name): "compiled inference has unknown target \(name)"
    case .executableFeedCount(let expected, let actual):
      "compiled inference has \(actual) feeds; expected \(expected)"
    case .executableTargetCount(let expected, let actual):
      "compiled inference has \(actual) targets; expected \(expected)"
    case .executableOutputCount(let expected, let actual):
      "compiled inference returned \(actual) outputs; expected \(expected)"
    }
  }
}
