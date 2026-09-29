import Foundation
import MetalPerformanceShadersGraph

public enum PaishoValueClassV1: Int, CaseIterable, Sendable {
  case win = 0
  case draw = 1
  case loss = 2

  public var signedReturn: Float {
    switch self {
    case .win: 1
    case .draw: 0
    case .loss: -1
    }
  }

  public var oneHot: [Float] {
    var values = [Float](repeating: 0, count: PaishoTensorSchemaV1.valueClasses)
    values[rawValue] = 1
    return values
  }
}

public enum PaishoTensorSchemaV1 {
  public static let boardSize = 17
  public static let boardCells = 289
  public static let spatialChannels = 29
  public static let globalFeatures = 26
  public static let actionFamilies = 7
  public static let tileKinds = 12
  public static let valueClasses = PaishoValueClassV1.allCases.count
}

public struct PaishoNetworkConfiguration: Codable, Equatable, Sendable {
  public let trunkChannels: Int
  public let residualBlocks: Int
  public let policyEmbeddingChannels: Int
  public let valueHiddenChannels: Int
  public let normalizationEpsilon: Float

  public init(
    trunkChannels: Int,
    residualBlocks: Int,
    policyEmbeddingChannels: Int,
    valueHiddenChannels: Int,
    normalizationEpsilon: Float = 1.0e-5
  ) throws {
    guard trunkChannels > 0 else {
      throw ConfigurationError.nonPositive("trunkChannels")
    }
    guard residualBlocks > 0 else {
      throw ConfigurationError.nonPositive("residualBlocks")
    }
    guard policyEmbeddingChannels > 0 else {
      throw ConfigurationError.nonPositive("policyEmbeddingChannels")
    }
    guard valueHiddenChannels > 0 else {
      throw ConfigurationError.nonPositive("valueHiddenChannels")
    }
    guard normalizationEpsilon.isFinite, normalizationEpsilon > 0 else {
      throw ConfigurationError.invalidNormalizationEpsilon
    }
    guard
      Self.computeParameterCount(
        trunkChannels: trunkChannels,
        residualBlocks: residualBlocks,
        policyEmbeddingChannels: policyEmbeddingChannels,
        valueHiddenChannels: valueHiddenChannels
      ) != nil
    else {
      throw ConfigurationError.parameterCountOverflow
    }
    self.trunkChannels = trunkChannels
    self.residualBlocks = residualBlocks
    self.policyEmbeddingChannels = policyEmbeddingChannels
    self.valueHiddenChannels = valueHiddenChannels
    self.normalizationEpsilon = normalizationEpsilon
  }

  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    do {
      try self.init(
        trunkChannels: values.decode(Int.self, forKey: .trunkChannels),
        residualBlocks: values.decode(Int.self, forKey: .residualBlocks),
        policyEmbeddingChannels: values.decode(Int.self, forKey: .policyEmbeddingChannels),
        valueHiddenChannels: values.decode(Int.self, forKey: .valueHiddenChannels),
        normalizationEpsilon: values.decode(Float.self, forKey: .normalizationEpsilon)
      )
    } catch let error as ConfigurationError {
      throw DecodingError.dataCorruptedError(
        forKey: .trunkChannels,
        in: values,
        debugDescription: error.description
      )
    }
  }

  public func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(trunkChannels, forKey: .trunkChannels)
    try values.encode(residualBlocks, forKey: .residualBlocks)
    try values.encode(policyEmbeddingChannels, forKey: .policyEmbeddingChannels)
    try values.encode(valueHiddenChannels, forKey: .valueHiddenChannels)
    try values.encode(normalizationEpsilon, forKey: .normalizationEpsilon)
  }

  public static let pureV1 = try! PaishoNetworkConfiguration(
    trunkChannels: 160,
    residualBlocks: 10,
    policyEmbeddingChannels: 32,
    valueHiddenChannels: 128
  )

  public static let microV1 = try! PaishoNetworkConfiguration(
    trunkChannels: 20,
    residualBlocks: 3,
    policyEmbeddingChannels: 8,
    valueHiddenChannels: 32
  )

  /// Trainable model values. Adam momentum and velocity are deliberately
  /// excluded because they are optimizer state, not network parameters.
  public var parameterCount: Int {
    Self.computeParameterCount(
      trunkChannels: trunkChannels,
      residualBlocks: residualBlocks,
      policyEmbeddingChannels: policyEmbeddingChannels,
      valueHiddenChannels: valueHiddenChannels
    )!
  }

  private static func computeParameterCount(
    trunkChannels: Int,
    residualBlocks: Int,
    policyEmbeddingChannels: Int,
    valueHiddenChannels: Int
  ) -> Int? {
    let channels = trunkChannels
    let embedding = policyEmbeddingChannels
    let valueHidden = valueHiddenChannels

    guard
      let stemConvolution = checkedSum([
        checkedProduct([3, 3, PaishoTensorSchemaV1.spatialChannels, channels]), channels,
      ]),
      let globalProjection = checkedSum([
        checkedProduct([PaishoTensorSchemaV1.globalFeatures, channels]), channels,
      ]),
      let stemNormalization = checkedProduct([2, channels]),
      let residualBlock = checkedSum([
        checkedProduct([18, channels, channels]), checkedProduct([6, channels]),
      ]),
      let residualTrunk = checkedProduct([residualBlocks, residualBlock]),
      let familyHead = checkedSum([
        checkedProduct([channels, PaishoTensorSchemaV1.actionFamilies]),
        PaishoTensorSchemaV1.actionFamilies,
      ]),
      let tileHead = checkedSum([
        checkedProduct([channels, PaishoTensorSchemaV1.tileKinds]),
        PaishoTensorSchemaV1.tileKinds,
      ]),
      let destinationHead = checkedSum([channels, 1]),
      let onePairEmbedding = checkedSum([
        checkedProduct([channels, embedding]), embedding,
      ]),
      let pairEmbeddings = checkedProduct([2, onePairEmbedding]),
      let valueHead = checkedSum([
        checkedProduct([channels, valueHidden]),
        valueHidden,
        checkedProduct([valueHidden, PaishoTensorSchemaV1.valueClasses]),
        PaishoTensorSchemaV1.valueClasses,
      ])
    else {
      return nil
    }
    return checkedSum([
      stemConvolution,
      globalProjection,
      stemNormalization,
      residualTrunk,
      familyHead,
      tileHead,
      destinationHead,
      pairEmbeddings,
      valueHead,
    ])
  }

  private enum CodingKeys: String, CodingKey {
    case trunkChannels
    case residualBlocks
    case policyEmbeddingChannels
    case valueHiddenChannels
    case normalizationEpsilon
  }
}

public struct PaishoExecutionShape: Codable, Equatable, Sendable {
  public let batchSize: Int
  public let legalActionCapacity: Int

  public init(batchSize: Int, legalActionCapacity: Int) throws {
    guard batchSize > 0 else {
      throw ConfigurationError.nonPositive("batchSize")
    }
    guard legalActionCapacity > 0 else {
      throw ConfigurationError.nonPositive("legalActionCapacity")
    }
    guard checkedProduct([batchSize, legalActionCapacity]) != nil,
      checkedProduct([
        batchSize,
        PaishoTensorSchemaV1.boardCells,
        PaishoTensorSchemaV1.spatialChannels,
      ]) != nil
    else {
      throw ConfigurationError.executionShapeOverflow
    }
    self.batchSize = batchSize
    self.legalActionCapacity = legalActionCapacity
  }

  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    do {
      try self.init(
        batchSize: values.decode(Int.self, forKey: .batchSize),
        legalActionCapacity: values.decode(Int.self, forKey: .legalActionCapacity)
      )
    } catch let error as ConfigurationError {
      throw DecodingError.dataCorruptedError(
        forKey: .batchSize,
        in: values,
        debugDescription: error.description
      )
    }
  }

  public func encode(to encoder: Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    try values.encode(batchSize, forKey: .batchSize)
    try values.encode(legalActionCapacity, forKey: .legalActionCapacity)
  }

  private enum CodingKeys: String, CodingKey {
    case batchSize
    case legalActionCapacity
  }
}

public enum PaishoGraphOptimization: String, CaseIterable, Codable, Sendable {
  case level0
  case level1

  var mpsGraphValue: MPSGraphOptimization {
    switch self {
    case .level0: .level0
    case .level1: .level1
    }
  }
}

public enum ConfigurationError: Error, Equatable, CustomStringConvertible {
  case nonPositive(String)
  case invalidNormalizationEpsilon
  case parameterCountOverflow
  case executionShapeOverflow

  public var description: String {
    switch self {
    case .nonPositive(let name): "\(name) must be positive"
    case .invalidNormalizationEpsilon: "normalizationEpsilon must be finite and positive"
    case .parameterCountOverflow: "network parameter count overflows Int"
    case .executionShapeOverflow: "execution shape value count overflows Int"
    }
  }
}

private func checkedProduct(_ values: [Int]) -> Int? {
  var result = 1
  for value in values {
    let (product, overflow) = result.multipliedReportingOverflow(by: value)
    guard !overflow else { return nil }
    result = product
  }
  return result
}

private func checkedSum(_ values: [Int?]) -> Int? {
  var result = 0
  for value in values {
    guard let value else { return nil }
    let (sum, overflow) = result.addingReportingOverflow(value)
    guard !overflow else { return nil }
    result = sum
  }
  return result
}
