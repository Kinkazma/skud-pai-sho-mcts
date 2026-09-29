import Foundation
import MetalPerformanceShadersGraph

public struct PaishoParameterSnapshot: Codable, Equatable, Sendable {
  public let name: String
  public let shape: [Int]
  public let values: [Float]
  public let momentum: [Float]
  public let velocity: [Float]

  public init(
    name: String,
    shape: [Int],
    values: [Float],
    momentum: [Float],
    velocity: [Float]
  ) {
    self.name = name
    self.shape = shape
    self.values = values
    self.momentum = momentum
    self.velocity = velocity
  }
}

final class GraphParameter {
  let name: String
  let shape: [Int]
  let count: Int
  let values: MPSGraphTensor
  let momentum: MPSGraphTensor
  let velocity: MPSGraphTensor

  init(
    name: String,
    shape: [Int],
    values: MPSGraphTensor,
    momentum: MPSGraphTensor,
    velocity: MPSGraphTensor
  ) {
    self.name = name
    self.shape = shape
    count = shape.reduce(1, *)
    self.values = values
    self.momentum = momentum
    self.velocity = velocity
  }
}

enum ParameterInitialization {
  case zeros
  case ones
  case heUniform(fanIn: Int)
  case scaledUniform(fanIn: Int, outputScale: Float)
}

final class ParameterStore {
  private let graph: MPSGraph
  private var rng: SplitMix64
  private let restored: [String: PaishoParameterSnapshot]
  private var usedRestoredNames = Set<String>()
  private(set) var parameters: [GraphParameter] = []

  init(
    graph: MPSGraph,
    seed: UInt64,
    restored: [PaishoParameterSnapshot] = []
  ) throws {
    self.graph = graph
    rng = SplitMix64(seed: seed)
    var dictionary: [String: PaishoParameterSnapshot] = [:]
    for snapshot in restored {
      guard dictionary.updateValue(snapshot, forKey: snapshot.name) == nil else {
        throw PaishoMPSGraphError.duplicateCheckpointParameter(snapshot.name)
      }
    }
    self.restored = dictionary
  }

  func make(
    name: String,
    shape: [Int],
    initialization: ParameterInitialization
  ) throws -> MPSGraphTensor {
    guard !parameters.contains(where: { $0.name == name }) else {
      throw PaishoMPSGraphError.duplicateParameter(name)
    }
    let count = shape.reduce(1, *)
    let initialValues: [Float]
    let initialMomentum: [Float]
    let initialVelocity: [Float]

    if let snapshot = restored[name] {
      guard snapshot.shape == shape,
        snapshot.values.count == count,
        snapshot.momentum.count == count,
        snapshot.velocity.count == count
      else {
        throw PaishoMPSGraphError.checkpointShapeMismatch(name)
      }
      guard snapshot.values.allSatisfy(\.isFinite),
        snapshot.momentum.allSatisfy(\.isFinite),
        snapshot.velocity.allSatisfy(\.isFinite)
      else {
        throw PaishoMPSGraphError.nonFiniteCheckpointParameter(name)
      }
      initialValues = snapshot.values
      initialMomentum = snapshot.momentum
      initialVelocity = snapshot.velocity
      usedRestoredNames.insert(name)
    } else {
      initialValues = initialize(count: count, initialization: initialization)
      initialMomentum = [Float](repeating: 0, count: count)
      initialVelocity = [Float](repeating: 0, count: count)
    }

    let numberShape = shape.map(NSNumber.init(value:))
    let values = graph.variable(
      with: data(from: initialValues),
      shape: numberShape,
      dataType: .float32,
      name: name
    )
    let momentum = graph.variable(
      with: data(from: initialMomentum),
      shape: numberShape,
      dataType: .float32,
      name: "\(name)/adam_m"
    )
    let velocity = graph.variable(
      with: data(from: initialVelocity),
      shape: numberShape,
      dataType: .float32,
      name: "\(name)/adam_v"
    )
    parameters.append(
      GraphParameter(
        name: name,
        shape: shape,
        values: values,
        momentum: momentum,
        velocity: velocity
      )
    )
    return values
  }

  func validateRestorationComplete() throws {
    let unused = Set(restored.keys).subtracting(usedRestoredNames)
    if let name = unused.sorted().first {
      throw PaishoMPSGraphError.unusedCheckpointParameter(name)
    }
  }

  private func initialize(
    count: Int,
    initialization: ParameterInitialization
  ) -> [Float] {
    switch initialization {
    case .zeros:
      [Float](repeating: 0, count: count)
    case .ones:
      [Float](repeating: 1, count: count)
    case .heUniform(let fanIn):
      (0..<count).map { _ in
        let limit = sqrt(6 / Float(fanIn))
        return rng.nextFloat(in: -limit...limit)
      }
    case .scaledUniform(let fanIn, let outputScale):
      (0..<count).map { _ in
        let limit = outputScale * sqrt(3 / Float(fanIn))
        return rng.nextFloat(in: -limit...limit)
      }
    }
  }
}
