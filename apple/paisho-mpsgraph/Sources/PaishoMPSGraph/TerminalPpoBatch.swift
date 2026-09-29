import Foundation

public struct PaishoTerminalPpoParametersV1: Equatable, Sendable {
  public let policyTemperature: Float
  public let uniformMix: Float
  public let clipEpsilon: Float
  public let valueLossWeight: Float
  public let entropyWeight: Float

  public init(
    policyTemperature: Float = 1,
    uniformMix: Float = 0,
    clipEpsilon: Float,
    valueLossWeight: Float,
    entropyWeight: Float
  ) throws {
    guard policyTemperature.isFinite, policyTemperature > 0 else {
      throw TerminalPpoParameterError.invalidPolicyTemperature(policyTemperature)
    }
    guard uniformMix.isFinite, (0...1).contains(uniformMix) else {
      throw TerminalPpoParameterError.invalidUniformMix(uniformMix)
    }
    guard clipEpsilon.isFinite, clipEpsilon > 0, clipEpsilon < 1 else {
      throw TerminalPpoParameterError.invalidClipEpsilon(clipEpsilon)
    }
    guard valueLossWeight.isFinite, valueLossWeight >= 0 else {
      throw TerminalPpoParameterError.invalidWeight(name: "value", value: valueLossWeight)
    }
    guard entropyWeight.isFinite, entropyWeight >= 0 else {
      throw TerminalPpoParameterError.invalidWeight(name: "entropy", value: entropyWeight)
    }
    self.policyTemperature = policyTemperature
    self.uniformMix = uniformMix
    self.clipEpsilon = clipEpsilon
    self.valueLossWeight = valueLossWeight
    self.entropyWeight = entropyWeight
  }
}

public enum TerminalPpoParameterError: Error, Equatable, CustomStringConvertible {
  case invalidPolicyTemperature(Float)
  case invalidUniformMix(Float)
  case invalidClipEpsilon(Float)
  case invalidWeight(name: String, value: Float)

  public var description: String {
    switch self {
    case .invalidPolicyTemperature(let value):
      "terminal PPO policy temperature \(value) is not positive and finite"
    case .invalidUniformMix(let value):
      "terminal PPO uniform mix \(value) is not in [0, 1]"
    case .invalidClipEpsilon(let value):
      "terminal PPO clip epsilon \(value) is not in (0, 1)"
    case .invalidWeight(let name, let value):
      "terminal PPO \(name) weight \(value) is not finite and non-negative"
    }
  }
}

public struct PaishoTerminalPpoExampleV1: Sendable {
  public let inference: PaishoInferenceExampleV1
  public let playedActionIndex: Int
  public let behaviorProbability: Float
  public let terminalValue: PaishoValueClassV1
  public let actorValue: Float
  public let policyReturn: Float?

  public init(
    inference: PaishoInferenceExampleV1,
    playedActionIndex: Int,
    behaviorProbability: Float,
    terminalValue: PaishoValueClassV1,
    actorValue: Float,
    policyReturn: Float? = nil
  ) {
    self.inference = inference
    self.playedActionIndex = playedActionIndex
    self.behaviorProbability = behaviorProbability
    self.terminalValue = terminalValue
    self.actorValue = actorValue
    self.policyReturn = policyReturn
  }
}

public struct PaishoTerminalPpoBatch: Sendable {
  public let inference: PaishoInferenceBatch
  public let playedActionMask: [Float]
  public let behaviorProbabilities: [Float]
  public let advantages: [Float]
  public let valueTargets: [Float]

  public var shape: PaishoExecutionShape { inference.shape }

  public init(
    inference: PaishoInferenceBatch,
    playedActionIndices: [Int],
    behaviorProbabilities: [Float],
    terminalValues: [PaishoValueClassV1],
    actorValues: [Float],
    policyReturns: [Float]? = nil
  ) throws {
    try inference.validate()
    let batch = inference.shape.batchSize
    let capacity = inference.shape.legalActionCapacity
    guard playedActionIndices.count == batch else {
      throw TerminalPpoBatchError.wrongCount(
        name: "playedActionIndices", expected: batch, actual: playedActionIndices.count
      )
    }
    guard behaviorProbabilities.count == batch else {
      throw TerminalPpoBatchError.wrongCount(
        name: "behaviorProbabilities", expected: batch, actual: behaviorProbabilities.count
      )
    }
    guard terminalValues.count == batch else {
      throw TerminalPpoBatchError.wrongCount(
        name: "terminalValues", expected: batch, actual: terminalValues.count
      )
    }
    guard actorValues.count == batch else {
      throw TerminalPpoBatchError.wrongCount(
        name: "actorValues", expected: batch, actual: actorValues.count
      )
    }

    if let returns = policyReturns, returns.count != batch {
      throw TerminalPpoBatchError.wrongCount(name: "policyReturns", expected: batch, actual: returns.count)
    }
    var playedActionMask = [Float](repeating: 0, count: batch * capacity)
    var advantages: [Float] = []
    var valueTargets: [Float] = []
    advantages.reserveCapacity(batch)
    valueTargets.reserveCapacity(batch * PaishoTensorSchemaV1.valueClasses)
    for row in 0..<batch {
      let played = playedActionIndices[row]
      guard played >= 0, played < capacity,
        inference.legalMask[row * capacity + played] == 1
      else {
        throw TerminalPpoBatchError.playedActionNotLegal(row: row, index: played)
      }
      let probability = behaviorProbabilities[row]
      guard probability.isFinite, probability > 0, probability <= 1 else {
        throw TerminalPpoBatchError.invalidBehaviorProbability(row: row, value: probability)
      }
      let actorValue = actorValues[row]
      guard actorValue.isFinite, (-1...1).contains(actorValue) else {
        throw TerminalPpoBatchError.invalidActorValue(row: row, value: actorValue)
      }
      playedActionMask[row * capacity + played] = 1
      let utility = policyReturns?[row] ?? terminalValues[row].signedReturn
      let terminal = terminalValues[row].signedReturn
      guard utility.isFinite, (terminal > 0 ? (0.9...1).contains(utility) : utility == terminal) else {
        throw TerminalPpoBatchError.invalidAdvantage(row: row, value: utility)
      }
      advantages.append(utility - actorValue)
      valueTargets.append(contentsOf: terminalValues[row].oneHot)
    }
    self.inference = inference
    self.playedActionMask = playedActionMask
    self.behaviorProbabilities = behaviorProbabilities
    self.advantages = advantages
    self.valueTargets = valueTargets
    try validate()
  }

  public static func packing(
    _ examples: [PaishoTerminalPpoExampleV1],
    legalActionCapacity: Int
  ) throws -> Self {
    try Self(
      inference: PaishoInferenceBatch.packing(
        examples.map(\.inference), legalActionCapacity: legalActionCapacity
      ),
      playedActionIndices: examples.map(\.playedActionIndex),
      behaviorProbabilities: examples.map(\.behaviorProbability),
      terminalValues: examples.map(\.terminalValue),
      actorValues: examples.map(\.actorValue),
      policyReturns: examples.map { $0.policyReturn ?? $0.terminalValue.signedReturn }
    )
  }

  public func validate() throws {
    try inference.validate()
    let batch = shape.batchSize
    let capacity = shape.legalActionCapacity
    try expectCount(playedActionMask, batch * capacity, "playedActionMask")
    try expectCount(behaviorProbabilities, batch, "behaviorProbabilities")
    try expectCount(advantages, batch, "advantages")
    try expectCount(
      valueTargets, batch * PaishoTensorSchemaV1.valueClasses, "terminalValueTargets"
    )
    try requireFinite(playedActionMask, name: "playedActionMask")
    try requireFinite(behaviorProbabilities, name: "behaviorProbabilities")
    try requireFinite(advantages, name: "advantages")
    try requireFinite(valueTargets, name: "terminalValueTargets")

    for row in 0..<batch {
      let actionRange = row * capacity..<(row + 1) * capacity
      let selected = playedActionMask[actionRange]
      guard selected.allSatisfy({ $0 == 0 || $0 == 1 }), selected.reduce(0, +) == 1 else {
        throw TerminalPpoBatchError.invalidPlayedActionMask(row: row)
      }
      for index in actionRange
      where playedActionMask[index] == 1 && inference.legalMask[index] != 1 {
        throw TerminalPpoBatchError.playedActionNotLegal(
          row: row, index: index - row * capacity
        )
      }
      let probability = behaviorProbabilities[row]
      guard probability > 0, probability <= 1 else {
        throw TerminalPpoBatchError.invalidBehaviorProbability(row: row, value: probability)
      }
      guard (-2...2).contains(advantages[row]) else {
        throw TerminalPpoBatchError.invalidAdvantage(row: row, value: advantages[row])
      }
      let valueStart = row * PaishoTensorSchemaV1.valueClasses
      let values = valueTargets[valueStart..<(valueStart + PaishoTensorSchemaV1.valueClasses)]
      guard values.allSatisfy({ $0 == 0 || $0 == 1 }), values.reduce(0, +) == 1 else {
        throw TerminalPpoBatchError.invalidTerminalValue(row: row)
      }
    }
  }
}

public enum TerminalPpoBatchError: Error, Equatable, CustomStringConvertible {
  case wrongCount(name: String, expected: Int, actual: Int)
  case playedActionNotLegal(row: Int, index: Int)
  case invalidPlayedActionMask(row: Int)
  case invalidBehaviorProbability(row: Int, value: Float)
  case invalidActorValue(row: Int, value: Float)
  case invalidAdvantage(row: Int, value: Float)
  case invalidTerminalValue(row: Int)

  public var description: String {
    switch self {
    case .wrongCount(let name, let expected, let actual):
      "terminal PPO \(name) has \(actual) values; expected \(expected)"
    case .playedActionNotLegal(let row, let index):
      "terminal PPO played action \(index) is not legal in row \(row)"
    case .invalidPlayedActionMask(let row):
      "terminal PPO row \(row) does not select exactly one played action"
    case .invalidBehaviorProbability(let row, let value):
      "terminal PPO behavior probability \(value) is not in (0, 1] in row \(row)"
    case .invalidActorValue(let row, let value):
      "terminal PPO actor value \(value) is not in [-1, 1] in row \(row)"
    case .invalidAdvantage(let row, let value):
      "terminal PPO advantage \(value) is not in [-2, 2] in row \(row)"
    case .invalidTerminalValue(let row):
      "terminal PPO value target is not one-hot in row \(row)"
    }
  }
}
