import Foundation

public struct PaishoTrainingExampleV1: Sendable {
  public let inference: PaishoInferenceExampleV1
  public let policyTargets: [Float]
  public let valueTargets: [Float]

  public init(
    inference: PaishoInferenceExampleV1,
    policyTargets: [Float],
    valueTargets: [Float]
  ) {
    self.inference = inference
    self.policyTargets = policyTargets
    self.valueTargets = valueTargets
  }
}

public struct PaishoTrainingBatch: Sendable {
  public let inference: PaishoInferenceBatch
  public let policyTargets: [Float]
  public let valueTargets: [Float]

  public var shape: PaishoExecutionShape { inference.shape }

  public init(
    inference: PaishoInferenceBatch,
    policyTargets: [Float],
    valueTargets: [Float]
  ) throws {
    self.inference = inference
    self.policyTargets = policyTargets
    self.valueTargets = valueTargets
    try validate()
  }

  public static func packing(
    _ examples: [PaishoTrainingExampleV1],
    legalActionCapacity: Int
  ) throws -> Self {
    let inference = try PaishoInferenceBatch.packing(
      examples.map(\.inference),
      legalActionCapacity: legalActionCapacity
    )
    var policyTargets = [Float](
      repeating: 0,
      count: examples.count * legalActionCapacity
    )
    var valueTargets: [Float] = []
    valueTargets.reserveCapacity(examples.count * PaishoTensorSchemaV1.valueClasses)
    for (row, example) in examples.enumerated() {
      guard example.policyTargets.count == example.inference.legalActions.count else {
        throw BatchError.wrongPolicyTargetCount(
          row: row,
          expected: example.inference.legalActions.count,
          actual: example.policyTargets.count
        )
      }
      guard example.valueTargets.count == PaishoTensorSchemaV1.valueClasses else {
        throw BatchError.wrongValueTargetCount(
          row: row,
          expected: PaishoTensorSchemaV1.valueClasses,
          actual: example.valueTargets.count
        )
      }
      policyTargets.replaceSubrange(
        row * legalActionCapacity..<(row * legalActionCapacity + example.policyTargets.count),
        with: example.policyTargets
      )
      valueTargets.append(contentsOf: example.valueTargets)
    }
    return try Self(
      inference: inference,
      policyTargets: policyTargets,
      valueTargets: valueTargets
    )
  }

  public func validate() throws {
    try inference.validate()
    let batch = shape.batchSize
    let capacity = shape.legalActionCapacity
    try expectCount(policyTargets, batch * capacity, "policyTargets")
    try expectCount(
      valueTargets,
      batch * PaishoTensorSchemaV1.valueClasses,
      "valueTargets"
    )
    try requireFinite(policyTargets, name: "policyTargets")
    try requireFinite(valueTargets, name: "valueTargets")

    for row in 0..<batch {
      let actions = row * capacity..<(row + 1) * capacity
      try requireNonNegative(policyTargets[actions], name: "policyTargets", row: row)
      let policySum = policyTargets[actions].reduce(0, +)
      guard abs(policySum - 1) <= 1.0e-5 else {
        throw BatchError.invalidDistribution(
          name: "policyTargets",
          row: row,
          sum: policySum
        )
      }
      for index in actions
      where inference.legalMask[index] == 0 && policyTargets[index] != 0 {
        throw BatchError.targetOnPadding(row: row, index: index - row * capacity)
      }

      let valueStart = row * PaishoTensorSchemaV1.valueClasses
      let valueRange = valueStart..<(valueStart + PaishoTensorSchemaV1.valueClasses)
      try requireNonNegative(valueTargets[valueRange], name: "valueTargets", row: row)
      let valueSum = valueTargets[valueRange].reduce(0, +)
      guard abs(valueSum - 1) <= 1.0e-5 else {
        throw BatchError.invalidDistribution(
          name: "valueTargets",
          row: row,
          sum: valueSum
        )
      }
    }
  }

  public static func synthetic(shape: PaishoExecutionShape, seed: UInt64 = 1) throws -> Self {
    var rng = SplitMix64(seed: seed)
    let spatialCount =
      PaishoTensorSchemaV1.boardCells
      * PaishoTensorSchemaV1.spatialChannels
    let examples = try (0..<shape.batchSize).map { row in
      let spatial = (0..<spatialCount).map { _ in rng.nextFloat(in: -0.25...0.25) }
      let global = (0..<PaishoTensorSchemaV1.globalFeatures).map {
        _ in rng.nextFloat(in: 0...1)
      }
      let legalCount = min(shape.legalActionCapacity, 7 + row % 11)
      let actions = try (0..<legalCount).map { action in
        try syntheticAction(row: row, action: action)
      }
      var value = [Float](repeating: 0, count: PaishoTensorSchemaV1.valueClasses)
      value[row % PaishoTensorSchemaV1.valueClasses] = 1
      return PaishoTrainingExampleV1(
        inference: PaishoInferenceExampleV1(
          spatial: spatial,
          global: global,
          legalActions: actions
        ),
        policyTargets: [Float](repeating: 1 / Float(legalCount), count: legalCount),
        valueTargets: value
      )
    }
    return try packing(examples, legalActionCapacity: shape.legalActionCapacity)
  }
}

public enum BatchError: Error, Equatable, CustomStringConvertible {
  case wrongCount(name: String, expected: Int, actual: Int)
  case wrongExampleCount(name: String, row: Int, expected: Int, actual: Int)
  case wrongPolicyTargetCount(row: Int, expected: Int, actual: Int)
  case wrongValueTargetCount(row: Int, expected: Int, actual: Int)
  case nonBinary(name: String, value: Float)
  case nonFinite(name: String)
  case negativeProbability(name: String, row: Int, value: Float)
  case noLegalAction(row: Int)
  case tooManyLegalActions(row: Int, capacity: Int, actual: Int)
  case invalidDistribution(name: String, row: Int, sum: Float)
  case targetOnPadding(row: Int, index: Int)
  case indexOutOfRange(name: String, value: Int32)
  case invalidActionComponents(row: Int, index: Int)

  public var description: String {
    switch self {
    case .wrongCount(let name, let expected, let actual):
      "\(name) has \(actual) values; expected \(expected)"
    case .wrongExampleCount(let name, let row, let expected, let actual):
      "\(name) row \(row) has \(actual) values; expected \(expected)"
    case .wrongPolicyTargetCount(let row, let expected, let actual):
      "policy target row \(row) has \(actual) values; expected \(expected)"
    case .wrongValueTargetCount(let row, let expected, let actual):
      "value target row \(row) has \(actual) values; expected \(expected)"
    case .nonBinary(let name, let value): "\(name) contains non-binary value \(value)"
    case .nonFinite(let name): "\(name) contains a non-finite value"
    case .negativeProbability(let name, let row, let value):
      "\(name) row \(row) contains negative probability \(value)"
    case .noLegalAction(let row): "batch row \(row) has no legal action"
    case .tooManyLegalActions(let row, let capacity, let actual):
      "batch row \(row) has \(actual) legal actions; capacity is \(capacity)"
    case .invalidDistribution(let name, let row, let sum):
      "\(name) row \(row) sums to \(sum), not 1"
    case .targetOnPadding(let row, let index):
      "policy target uses padded action \(index) in row \(row)"
    case .indexOutOfRange(let name, let value):
      "\(name) contains out-of-range index \(value)"
    case .invalidActionComponents(let row, let index):
      "action \(index) in row \(row) is not a canonical V1 action"
    }
  }
}

func expectCount<T>(_ values: [T], _ expected: Int, _ name: String) throws {
  guard values.count == expected else {
    throw BatchError.wrongCount(name: name, expected: expected, actual: values.count)
  }
}

func requireBinary(_ values: ArraySlice<Float>, name: String) throws {
  if let value = values.first(where: { $0 != 0 && $0 != 1 }) {
    throw BatchError.nonBinary(name: name, value: value)
  }
}

func requireFinite(_ values: [Float], name: String) throws {
  guard values.allSatisfy(\.isFinite) else {
    throw BatchError.nonFinite(name: name)
  }
}

private func requireNonNegative(
  _ values: ArraySlice<Float>,
  name: String,
  row: Int
) throws {
  if let value = values.first(where: { $0 < 0 }) {
    throw BatchError.negativeProbability(name: name, row: row, value: value)
  }
}

private func syntheticAction(row: Int, action: Int) throws -> PaishoActionAddressV1 {
  let family = UInt16(action % PaishoTensorSchemaV1.actionFamilies)
  let source = syntheticPlayableSlot(startingAt: action * 17 + row * 3)
  var destination = syntheticPlayableSlot(startingAt: action * 31 + row * 7 + 1)
  if destination == source {
    destination = syntheticPlayableSlot(startingAt: Int(destination) + 1)
  }
  let gates: [UInt16] = [8, 152, 280, 136]
  let noTile = PaishoActionAddressV1.noTile
  let noCoordinate = PaishoActionAddressV1.noCoordinate
  let slots: [UInt16]
  switch family {
  case 0, 6:
    slots = [family, UInt16((action + row) % 6), noCoordinate, gates[(action + row) % 4]]
  case 1: slots = [family, noTile, source, destination]
  case 2: slots = [family, noTile, noCoordinate, noCoordinate]
  case 3:
    while gates.contains(destination) {
      destination = syntheticPlayableSlot(startingAt: Int(destination) + 1)
    }
    slots = [family, UInt16(8 + (action + row) % 4), noCoordinate, destination]
  case 4: slots = [family, 11, source, destination]
  case 5:
    slots = [family, UInt16(6 + (action + row) % 2), noCoordinate, gates[(action + row) % 4]]
  default: preconditionFailure("family is reduced modulo seven")
  }
  return try PaishoActionAddressV1(slots: slots)
}

private func syntheticPlayableSlot(startingAt start: Int) -> UInt16 {
  var slot = start % PaishoTensorSchemaV1.boardCells
  while true {
    let row = slot / PaishoTensorSchemaV1.boardSize
    let column = slot % PaishoTensorSchemaV1.boardSize
    if abs(column - 8) + abs(8 - row) <= 12 {
      return UInt16(slot)
    }
    slot = (slot + 1) % PaishoTensorSchemaV1.boardCells
  }
}

struct SplitMix64 {
  private var state: UInt64

  init(seed: UInt64) {
    state = seed
  }

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var value = state
    value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
    value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
    return value ^ (value >> 31)
  }

  mutating func nextUnitFloat() -> Float {
    Float(Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0))
  }

  mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
    range.lowerBound + (range.upperBound - range.lowerBound) * nextUnitFloat()
  }
}
