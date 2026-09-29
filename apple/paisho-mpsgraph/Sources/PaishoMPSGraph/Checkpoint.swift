import CryptoKit
import Foundation

public struct PaishoNamedRandomState: Codable, Equatable, Sendable {
  public let name: String
  public let state: UInt64

  public init(name: String, state: UInt64) throws {
    guard !name.isEmpty else {
      throw PaishoCheckpointError.emptyRandomStateName
    }
    self.name = name
    self.state = state
  }
}

/// The first executable scheduler is deliberately small: a constant learning
/// rate and its exact completed-step cursor. Additional schedules can be added
/// as new tagged cases without changing parameter storage.
public struct PaishoLearningRateScheduler: Codable, Equatable, Sendable {
  public let learningRate: Float
  public private(set) var completedSteps: UInt64

  public init(learningRate: Float, completedSteps: UInt64 = 0) throws {
    self.learningRate = learningRate
    self.completedSteps = completedSteps
    try validate()
  }

  func validate() throws {
    guard learningRate.isFinite, learningRate > 0 else {
      throw PaishoCheckpointError.invalidSchedulerLearningRate
    }
  }

  mutating func didCompleteStep() throws {
    guard completedSteps < UInt64.max else {
      throw PaishoCheckpointError.schedulerStepOverflow
    }
    completedSteps += 1
  }
}

public struct PaishoTrainingProgress: Codable, Equatable, Sendable {
  public let generation: UInt64
  public let replayIndex: UInt64
  public let replaySnapshotSHA256: String
  public var scheduler: PaishoLearningRateScheduler
  public let randomStates: [PaishoNamedRandomState]

  public init(
    generation: UInt64,
    replayIndex: UInt64,
    replaySnapshotSHA256: String,
    scheduler: PaishoLearningRateScheduler,
    randomStates: [PaishoNamedRandomState]
  ) throws {
    self.generation = generation
    self.replayIndex = replayIndex
    self.replaySnapshotSHA256 = replaySnapshotSHA256
    self.scheduler = scheduler
    self.randomStates = randomStates.sorted { $0.name < $1.name }
    try validate()
  }

  func validate() throws {
    try scheduler.validate()
    guard replaySnapshotSHA256.count == 64,
      replaySnapshotSHA256.utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      })
    else {
      throw PaishoCheckpointError.invalidReplaySnapshotSHA256
    }
    guard !randomStates.isEmpty else {
      throw PaishoCheckpointError.missingRandomState
    }
    var names = Set<String>()
    for randomState in randomStates {
      guard !randomState.name.isEmpty else {
        throw PaishoCheckpointError.emptyRandomStateName
      }
      guard names.insert(randomState.name).inserted else {
        throw PaishoCheckpointError.duplicateRandomState(randomState.name)
      }
    }
  }
}

public struct PaishoCheckpointMetadata: Codable, Equatable, Sendable {
  public static let formatVersion: UInt32 = 2
  public static let tensorSchema = "paisho-neural-encoding-v1"
  public static let ruleProfile = "skud-pai-sho-2022-03-14"

  public let formatVersion: UInt32
  public let tensorSchema: String
  public let ruleProfile: String
  public let configuration: PaishoNetworkConfiguration
  public let executionShape: PaishoExecutionShape
  public let optimization: PaishoGraphOptimization
  public let trainingStep: UInt64
  public let progress: PaishoTrainingProgress

  init(
    configuration: PaishoNetworkConfiguration,
    executionShape: PaishoExecutionShape,
    optimization: PaishoGraphOptimization,
    trainingStep: UInt64,
    progress: PaishoTrainingProgress
  ) throws {
    guard progress.scheduler.completedSteps == trainingStep else {
      throw PaishoCheckpointError.schedulerStepMismatch(
        model: trainingStep,
        scheduler: progress.scheduler.completedSteps
      )
    }
    formatVersion = Self.formatVersion
    tensorSchema = Self.tensorSchema
    ruleProfile = Self.ruleProfile
    self.configuration = configuration
    self.executionShape = executionShape
    self.optimization = optimization
    self.trainingStep = trainingStep
    self.progress = progress
  }

  func validate() throws {
    guard formatVersion == Self.formatVersion else {
      throw PaishoCheckpointError.unsupportedFormatVersion(formatVersion)
    }
    guard tensorSchema == Self.tensorSchema else {
      throw PaishoCheckpointError.unsupportedTensorSchema(tensorSchema)
    }
    guard ruleProfile == Self.ruleProfile else {
      throw PaishoCheckpointError.unsupportedRuleProfile(ruleProfile)
    }
    guard progress.scheduler.completedSteps == trainingStep else {
      throw PaishoCheckpointError.schedulerStepMismatch(
        model: trainingStep,
        scheduler: progress.scheduler.completedSteps
      )
    }
    try progress.validate()
  }
}

public struct PaishoTrainingCheckpoint: Sendable {
  public let metadata: PaishoCheckpointMetadata
  public let parameters: [PaishoParameterSnapshot]

  public init(
    metadata: PaishoCheckpointMetadata,
    parameters: [PaishoParameterSnapshot]
  ) throws {
    self.metadata = metadata
    self.parameters = parameters
    try validate()
  }

  @discardableResult
  public func write(to destination: URL) throws -> [UInt8] {
    try publish(to: destination, acceptingIdenticalExistingFile: false)
  }

  @discardableResult
  public func writeIdempotently(to destination: URL) throws -> [UInt8] {
    try publish(to: destination, acceptingIdenticalExistingFile: true)
  }

  private func publish(
    to destination: URL,
    acceptingIdenticalExistingFile: Bool
  ) throws -> [UInt8] {
    let encoded = try encode()
    let manager = FileManager.default
    let parent = destination.deletingLastPathComponent()
    try manager.createDirectory(at: parent, withIntermediateDirectories: true)
    if manager.fileExists(atPath: destination.path) {
      return try resolveExistingPublication(
        at: destination,
        encoded: encoded,
        acceptingIdentical: acceptingIdenticalExistingFile
      )
    }

    let temporary = parent.appendingPathComponent(
      ".\(destination.lastPathComponent).tmp-\(UUID().uuidString)",
      isDirectory: false
    )
    defer { try? manager.removeItem(at: temporary) }
    try encoded.write(to: temporary, options: .withoutOverwriting)
    let handle = try FileHandle(forWritingTo: temporary)
    try handle.synchronize()
    try handle.close()
    do {
      try manager.moveItem(at: temporary, to: destination)
    } catch {
      guard acceptingIdenticalExistingFile, manager.fileExists(atPath: destination.path) else {
        throw error
      }
      return try resolveExistingPublication(
        at: destination,
        encoded: encoded,
        acceptingIdentical: true
      )
    }
    return Array(encoded.suffix(32))
  }

  private func resolveExistingPublication(
    at destination: URL,
    encoded: Data,
    acceptingIdentical: Bool
  ) throws -> [UInt8] {
    guard acceptingIdentical else {
      throw PaishoCheckpointError.destinationExists(destination.path)
    }
    let existing = try Data(contentsOf: destination, options: .mappedIfSafe)
    guard existing == encoded else {
      throw PaishoCheckpointError.destinationExists(destination.path)
    }
    return Array(encoded.suffix(32))
  }

  public static func read(from source: URL) throws -> Self {
    let data = try Data(contentsOf: source, options: .mappedIfSafe)
    return try decode(data)
  }

  private func validate() throws {
    try metadata.validate()
    guard !parameters.isEmpty else {
      throw PaishoCheckpointError.missingParameters
    }
    var names = Set<String>()
    var totalValues = 0
    for parameter in parameters {
      guard !parameter.name.isEmpty else {
        throw PaishoCheckpointError.emptyParameterName
      }
      guard names.insert(parameter.name).inserted else {
        throw PaishoCheckpointError.duplicateParameter(parameter.name)
      }
      let expected = try checkedElementCount(parameter.shape, parameter: parameter.name)
      guard parameter.values.count == expected,
        parameter.momentum.count == expected,
        parameter.velocity.count == expected
      else {
        throw PaishoCheckpointError.parameterShapeMismatch(parameter.name)
      }
      guard parameter.values.allSatisfy(\.isFinite),
        parameter.momentum.allSatisfy(\.isFinite),
        parameter.velocity.allSatisfy(\.isFinite)
      else {
        throw PaishoCheckpointError.nonFiniteParameter(parameter.name)
      }
      let (sum, overflow) = totalValues.addingReportingOverflow(expected)
      guard !overflow else {
        throw PaishoCheckpointError.invalidParameterShape(parameter.name)
      }
      totalValues = sum
    }
    guard totalValues == metadata.configuration.parameterCount else {
      throw PaishoCheckpointError.parameterCountMismatch(
        expected: metadata.configuration.parameterCount,
        actual: totalValues
      )
    }
  }

  private func encode() throws -> Data {
    try validate()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let metadataData = try encoder.encode(metadata)
    guard metadataData.count <= Int(UInt32.max), parameters.count <= Int(UInt32.max) else {
      throw PaishoCheckpointError.fileTooLarge
    }

    var writer = BinaryWriter()
    writer.append(Self.magic)
    writer.append(UInt32(metadataData.count))
    writer.append(metadataData)
    writer.append(UInt32(parameters.count))
    for parameter in parameters {
      let name = Data(parameter.name.utf8)
      guard name.count <= Int(UInt32.max), parameter.shape.count <= Int(UInt32.max) else {
        throw PaishoCheckpointError.fileTooLarge
      }
      writer.append(UInt32(name.count))
      writer.append(name)
      writer.append(UInt32(parameter.shape.count))
      for dimension in parameter.shape {
        writer.append(UInt64(dimension))
      }
      writer.append(UInt64(parameter.values.count))
      writer.append(parameter.values)
      writer.append(parameter.momentum)
      writer.append(parameter.velocity)
    }
    let digest = SHA256.hash(data: writer.data)
    writer.append(Data(digest))
    return writer.data
  }

  private static func decode(_ data: Data) throws -> Self {
    guard data.count >= magic.count + 32 else {
      throw PaishoCheckpointError.truncatedFile
    }
    let contentLength = data.count - 32
    let expectedDigest = data.suffix(32)
    let actualDigest = SHA256.hash(data: data.prefix(contentLength))
    guard Data(actualDigest) == expectedDigest else {
      throw PaishoCheckpointError.checksumMismatch
    }

    var reader = BinaryReader(data: data, limit: contentLength)
    guard try reader.readData(count: magic.count) == magic else {
      throw PaishoCheckpointError.invalidMagic
    }
    let metadataLength = try reader.readUInt32()
    guard metadataLength <= 16 * 1_024 * 1_024 else {
      throw PaishoCheckpointError.fileTooLarge
    }
    let metadataData = try reader.readData(count: Int(metadataLength))
    let metadata = try JSONDecoder().decode(PaishoCheckpointMetadata.self, from: metadataData)
    try metadata.validate()

    let parameterCount = try reader.readUInt32()
    guard parameterCount <= 4_096 else {
      throw PaishoCheckpointError.fileTooLarge
    }
    var parameters: [PaishoParameterSnapshot] = []
    parameters.reserveCapacity(Int(parameterCount))
    for _ in 0..<parameterCount {
      let nameLength = try reader.readUInt32()
      guard nameLength <= 4_096 else {
        throw PaishoCheckpointError.fileTooLarge
      }
      let nameData = try reader.readData(count: Int(nameLength))
      guard let name = String(data: nameData, encoding: .utf8) else {
        throw PaishoCheckpointError.invalidParameterName
      }
      let rank = try reader.readUInt32()
      guard rank > 0, rank <= 16 else {
        throw PaishoCheckpointError.invalidParameterShape(name)
      }
      var shape: [Int] = []
      shape.reserveCapacity(Int(rank))
      for _ in 0..<rank {
        let dimension = try reader.readUInt64()
        guard dimension > 0, dimension <= UInt64(Int.max) else {
          throw PaishoCheckpointError.invalidParameterShape(name)
        }
        shape.append(Int(dimension))
      }
      let storedCount = try reader.readUInt64()
      let expectedCount = try checkedElementCount(shape, parameter: name)
      guard storedCount == UInt64(expectedCount) else {
        throw PaishoCheckpointError.parameterShapeMismatch(name)
      }
      let values = try reader.readFloats(count: expectedCount)
      let momentum = try reader.readFloats(count: expectedCount)
      let velocity = try reader.readFloats(count: expectedCount)
      parameters.append(
        PaishoParameterSnapshot(
          name: name,
          shape: shape,
          values: values,
          momentum: momentum,
          velocity: velocity
        )
      )
    }
    guard reader.isAtEnd else {
      throw PaishoCheckpointError.trailingData
    }
    return try Self(metadata: metadata, parameters: parameters)
  }

  private static let magic = Data("PAISHO-CKPT-V2\n".utf8)
}

public enum PaishoCheckpointError: Error, Equatable, CustomStringConvertible {
  case emptyRandomStateName
  case missingRandomState
  case duplicateRandomState(String)
  case invalidReplaySnapshotSHA256
  case replaySnapshotMismatch(expected: String, actual: String)
  case invalidSchedulerLearningRate
  case schedulerStepOverflow
  case schedulerStepMismatch(model: UInt64, scheduler: UInt64)
  case unsupportedFormatVersion(UInt32)
  case unsupportedTensorSchema(String)
  case unsupportedRuleProfile(String)
  case missingParameters
  case emptyParameterName
  case invalidParameterName
  case duplicateParameter(String)
  case invalidParameterShape(String)
  case parameterShapeMismatch(String)
  case nonFiniteParameter(String)
  case parameterCountMismatch(expected: Int, actual: Int)
  case destinationExists(String)
  case truncatedFile
  case invalidMagic
  case checksumMismatch
  case trailingData
  case fileTooLarge

  public var description: String {
    switch self {
    case .emptyRandomStateName: "random-state names cannot be empty"
    case .missingRandomState: "checkpoint progress requires at least one random state"
    case .duplicateRandomState(let name): "duplicate random state \(name)"
    case .invalidReplaySnapshotSHA256:
      "replay snapshot SHA-256 must contain 64 lowercase hexadecimal characters"
    case .replaySnapshotMismatch(let expected, let actual):
      "checkpoint replay snapshot " + actual + " does not match expected snapshot " + expected
    case .invalidSchedulerLearningRate: "scheduler learning rate must be finite and positive"
    case .schedulerStepOverflow: "scheduler step overflow"
    case .schedulerStepMismatch(let model, let scheduler):
      "model step \(model) does not match scheduler step \(scheduler)"
    case .unsupportedFormatVersion(let version):
      "unsupported checkpoint format version \(version)"
    case .unsupportedTensorSchema(let schema): "unsupported tensor schema \(schema)"
    case .unsupportedRuleProfile(let profile): "unsupported rule profile \(profile)"
    case .missingParameters: "checkpoint contains no parameters"
    case .emptyParameterName: "checkpoint contains an empty parameter name"
    case .invalidParameterName: "checkpoint contains a non-UTF-8 parameter name"
    case .duplicateParameter(let name): "duplicate checkpoint parameter \(name)"
    case .invalidParameterShape(let name): "invalid shape for checkpoint parameter \(name)"
    case .parameterShapeMismatch(let name):
      "stored value count does not match the shape of checkpoint parameter \(name)"
    case .nonFiniteParameter(let name): "checkpoint parameter \(name) is non-finite"
    case .parameterCountMismatch(let expected, let actual):
      "checkpoint has \(actual) model values; expected \(expected)"
    case .destinationExists(let path): "checkpoint destination already exists: \(path)"
    case .truncatedFile: "checkpoint file is truncated"
    case .invalidMagic: "checkpoint file magic is invalid"
    case .checksumMismatch: "checkpoint SHA-256 verification failed"
    case .trailingData: "checkpoint contains unexpected trailing data"
    case .fileTooLarge: "checkpoint field exceeds the supported size"
    }
  }
}

private struct BinaryWriter {
  private(set) var data = Data()

  mutating func append(_ bytes: Data) {
    data.append(bytes)
  }

  mutating func append(_ value: UInt32) {
    var littleEndian = value.littleEndian
    Swift.withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  mutating func append(_ value: UInt64) {
    var littleEndian = value.littleEndian
    Swift.withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  mutating func append(_ values: [Float]) {
    #if _endian(little)
      values.withUnsafeBytes { data.append(contentsOf: $0) }
    #else
      for value in values {
        append(UInt32(value.bitPattern))
      }
    #endif
  }
}

private struct BinaryReader {
  private let data: Data
  private let limit: Int
  private var offset = 0

  init(data: Data, limit: Int) {
    self.data = data
    self.limit = limit
  }

  var isAtEnd: Bool { offset == limit }

  mutating func readData(count: Int) throws -> Data {
    guard count >= 0, offset <= limit, count <= limit - offset else {
      throw PaishoCheckpointError.truncatedFile
    }
    defer { offset += count }
    return data.subdata(in: offset..<(offset + count))
  }

  mutating func readUInt32() throws -> UInt32 {
    var value: UInt32 = 0
    let bytes = try readData(count: MemoryLayout<UInt32>.size)
    _ = withUnsafeMutableBytes(of: &value) { bytes.copyBytes(to: $0) }
    return UInt32(littleEndian: value)
  }

  mutating func readUInt64() throws -> UInt64 {
    var value: UInt64 = 0
    let bytes = try readData(count: MemoryLayout<UInt64>.size)
    _ = withUnsafeMutableBytes(of: &value) { bytes.copyBytes(to: $0) }
    return UInt64(littleEndian: value)
  }

  mutating func readFloats(count: Int) throws -> [Float] {
    guard count >= 0, count <= (limit - offset) / MemoryLayout<Float>.size else {
      throw PaishoCheckpointError.truncatedFile
    }
    var values = [Float](repeating: 0, count: count)
    let bytes = try readData(count: count * MemoryLayout<Float>.size)
    values.withUnsafeMutableBytes { _ = bytes.copyBytes(to: $0) }
    #if _endian(big)
      for index in values.indices {
        values[index] = Float(bitPattern: values[index].bitPattern.byteSwapped)
      }
    #endif
    return values
  }
}

private func checkedElementCount(_ shape: [Int], parameter: String) throws -> Int {
  guard !shape.isEmpty else {
    throw PaishoCheckpointError.invalidParameterShape(parameter)
  }
  var count = 1
  for dimension in shape {
    guard dimension > 0 else {
      throw PaishoCheckpointError.invalidParameterShape(parameter)
    }
    let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
    guard !overflow else {
      throw PaishoCheckpointError.invalidParameterShape(parameter)
    }
    count = product
  }
  return count
}
