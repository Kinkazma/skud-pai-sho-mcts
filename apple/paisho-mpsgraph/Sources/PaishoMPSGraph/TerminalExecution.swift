import Metal
import MetalPerformanceShadersGraph

// Sequential Adam execution: no overlapping updates or changed batch semantics.
final class PaishoTerminalExecution {
  private let executable: MPSGraphExecutable
  private let inputs: [MPSGraphTensor]
  private let targets: [MPSGraphTensor]

  init(graph: MPSGraph, device: MPSGraphDevice,
       optimization: PaishoGraphOptimization,
       feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor],
       operations: [MPSGraphOperation]) throws {
    let types = feeds.mapValues { MPSGraphShapedType(shape: $0.shape, dataType: $0.dataType) }
    let descriptor = MPSGraphCompilationDescriptor()
    descriptor.optimizationLevel = optimization.mpsGraphValue
    descriptor.waitForCompilationCompletion = true
    executable = graph.compile(with: device, feeds: types, targetTensors: targets,
                               targetOperations: operations, compilationDescriptor: descriptor)
    guard let orderedInputs = executable.feedTensors,
          let orderedTargets = executable.targetTensors,
          Set(orderedInputs) == Set(feeds.keys), Set(orderedTargets) == Set(targets) else {
      throw ExecutionError.tensorOrder
    }
    inputs = orderedInputs
    self.targets = orderedTargets
    executable.specialize(with: device, inputTypes: inputs.map { types[$0]! },
                          compilationDescriptor: descriptor)
  }

  func run(queue: any MTLCommandQueue,
           feeds: [MPSGraphTensor: MPSGraphTensorData]) throws
    -> [MPSGraphTensor: MPSGraphTensorData] {
    let values = executable.run(with: queue, inputs: inputs.map { feeds[$0]! },
                                results: nil, executionDescriptor: nil)
    guard values.count == targets.count else { throw ExecutionError.tensorOrder }
    return Dictionary(uniqueKeysWithValues: zip(targets, values))
  }

  private enum ExecutionError: Error { case tensorOrder }
}
