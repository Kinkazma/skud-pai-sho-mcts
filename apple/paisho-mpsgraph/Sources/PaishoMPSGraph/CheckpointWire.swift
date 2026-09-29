import Foundation

public struct PaishoCheckpointRequestV1: Sendable {
  public let requestID: UInt64
  public let expectedTrainingStep: UInt64
  public let replaySnapshotSHA256: String
  public let replayIndex: UInt64
  public let generation: UInt64
  public let learningRate: Float
  public let randomStates: [PaishoNamedRandomState]
  public let destinationPath: String
}

public enum PaishoCheckpointWireV1 {
  public static let requestMagic = Array("PSCREQ01".utf8)
  public static let responseMagic = Array("PSCRSP01".utf8)
  public static let errorMagic = Array("PSCERR01".utf8)
  public static let maximumRequestPayloadSize = 65_536

  private static let maximumRandomStates = 1_024
  private static let maximumRandomStateNameBytes = 4_096
  private static let maximumDestinationBytes = 16_384

  public static func isRequest(_ payload: Data) -> Bool {
    payload.count >= requestMagic.count && Array(payload.prefix(requestMagic.count)) == requestMagic
  }

  public static func decodeRequest(_ payload: Data) throws -> PaishoCheckpointRequestV1 {
    guard payload.count <= maximumRequestPayloadSize else {
      throw PaishoCheckpointWireError.payloadTooLarge(payload.count)
    }
    var reader = CheckpointWireReader(payload)
    guard try reader.bytes(count: 8) == requestMagic else {
      throw PaishoCheckpointWireError.invalidRequestMagic
    }
    let requestID = try reader.uint64()
    let expectedTrainingStep = try reader.uint64()
    let replaySnapshotSHA256 = try reader.bytes(count: 32)
      .map { String(format: "%02x", $0) }
      .joined()
    let replayIndex = try reader.uint64()
    let generation = try reader.uint64()
    let learningRate = try reader.float()
    guard learningRate.isFinite, learningRate > 0 else {
      throw PaishoCheckpointWireError.invalidLearningRate(learningRate)
    }
    let randomStateCount = Int(try reader.uint32())
    guard randomStateCount > 0 else {
      throw PaishoCheckpointWireError.missingRandomState
    }
    guard randomStateCount <= maximumRandomStates else {
      throw PaishoCheckpointWireError.tooManyRandomStates(randomStateCount)
    }
    var randomStates: [PaishoNamedRandomState] = []
    randomStates.reserveCapacity(randomStateCount)
    var randomStateNames = Set<String>()
    for _ in 0..<randomStateCount {
      let name = try reader.string(maximumBytes: maximumRandomStateNameBytes)
      guard !name.isEmpty else {
        throw PaishoCheckpointWireError.emptyRandomStateName
      }
      guard randomStateNames.insert(name).inserted else {
        throw PaishoCheckpointWireError.duplicateRandomState(name)
      }
      randomStates.append(try PaishoNamedRandomState(name: name, state: reader.uint64()))
    }
    let destinationPath = try reader.string(maximumBytes: maximumDestinationBytes)
    guard !destinationPath.isEmpty, !destinationPath.contains("\0") else {
      throw PaishoCheckpointWireError.invalidDestination
    }
    try reader.finish()
    return PaishoCheckpointRequestV1(
      requestID: requestID,
      expectedTrainingStep: expectedTrainingStep,
      replaySnapshotSHA256: replaySnapshotSHA256,
      replayIndex: replayIndex,
      generation: generation,
      learningRate: learningRate,
      randomStates: randomStates,
      destinationPath: destinationPath
    )
  }

  public static func encodeResponse(
    requestID: UInt64,
    completedTrainingStep: UInt64,
    completedReplayIndex: UInt64,
    contentSHA256: [UInt8]
  ) throws -> Data {
    guard contentSHA256.count == 32 else {
      throw PaishoCheckpointWireError.invalidContentDigest(contentSHA256.count)
    }
    var writer = CheckpointWireWriter()
    writer.bytes(responseMagic)
    writer.uint64(requestID)
    writer.uint64(completedTrainingStep)
    writer.uint64(completedReplayIndex)
    writer.bytes(contentSHA256)
    return writer.data
  }

  public static func encodeError(requestID: UInt64, error: Error) -> Data {
    let message = Data(String(describing: error).prefix(16_384).utf8)
    var writer = CheckpointWireWriter()
    writer.bytes(errorMagic)
    writer.uint64(requestID)
    writer.uint32(UInt32(message.count))
    writer.data.append(message)
    return writer.data
  }

  public static func requestIDIfPresent(in payload: Data) -> UInt64 {
    guard payload.count >= 16, isRequest(payload) else { return 0 }
    var reader = CheckpointWireReader(payload)
    _ = try? reader.bytes(count: 8)
    return (try? reader.uint64()) ?? 0
  }
}

public enum PaishoCheckpointWireError: Error, Equatable, CustomStringConvertible {
  case truncatedPayload
  case trailingBytes(Int)
  case invalidRequestMagic
  case payloadTooLarge(Int)
  case invalidLearningRate(Float)
  case missingRandomState
  case tooManyRandomStates(Int)
  case stringTooLong(actual: Int, maximum: Int)
  case invalidUTF8
  case emptyRandomStateName
  case duplicateRandomState(String)
  case invalidDestination
  case invalidContentDigest(Int)
  case trainingStepMismatch(expected: UInt64, actual: UInt64)
  case replaySnapshotMismatch(expected: String, actual: String)
  case replayIndexMismatch(expected: UInt64, actual: UInt64)
  case learningRateMismatch(expected: Float, actual: Float)

  public var description: String {
    switch self {
    case .truncatedPayload: "truncated checkpoint payload"
    case .trailingBytes(let count): "checkpoint payload has \(count) trailing bytes"
    case .invalidRequestMagic: "invalid checkpoint request magic"
    case .payloadTooLarge(let count): "checkpoint payload has \(count) bytes"
    case .invalidLearningRate(let value):
      "checkpoint learning rate \(value) is not positive and finite"
    case .missingRandomState: "checkpoint requires at least one random state"
    case .tooManyRandomStates(let count): "checkpoint has \(count) random states"
    case .stringTooLong(let actual, let maximum):
      "checkpoint string has \(actual) bytes; maximum is \(maximum)"
    case .invalidUTF8: "checkpoint string is not valid UTF-8"
    case .emptyRandomStateName: "checkpoint random-state name is empty"
    case .duplicateRandomState(let name): "checkpoint repeats random state \(name)"
    case .invalidDestination:
      "checkpoint destination must be non-empty and contain no NUL"
    case .invalidContentDigest(let count):
      "checkpoint content digest has \(count) bytes; expected 32"
    case .trainingStepMismatch(let expected, let actual):
      "checkpoint request expected step \(expected); model is at \(actual)"
    case .replaySnapshotMismatch(let expected, let actual):
      "checkpoint snapshot \(actual) does not match \(expected)"
    case .replayIndexMismatch(let expected, let actual):
      "checkpoint replay index \(actual) does not match \(expected)"
    case .learningRateMismatch(let expected, let actual):
      "checkpoint learning rate \(actual) does not match \(expected)"
    }
  }
}

private struct CheckpointWireReader {
  let bytes: [UInt8]
  var cursor = 0

  init(_ data: Data) {
    bytes = Array(data)
  }

  mutating func bytes(count: Int) throws -> [UInt8] {
    let (end, overflow) = cursor.addingReportingOverflow(count)
    guard !overflow, cursor >= 0, end <= bytes.count else {
      throw PaishoCheckpointWireError.truncatedPayload
    }
    let values = Array(bytes[cursor..<end])
    cursor = end
    return values
  }

  mutating func uint32() throws -> UInt32 {
    let value = try bytes(count: 4)
    return value.enumerated().reduce(0) { result, item in
      result | UInt32(item.element) << UInt32(item.offset * 8)
    }
  }

  mutating func uint64() throws -> UInt64 {
    let value = try bytes(count: 8)
    return value.enumerated().reduce(0) { result, item in
      result | UInt64(item.element) << UInt64(item.offset * 8)
    }
  }

  mutating func float() throws -> Float {
    Float(bitPattern: try uint32())
  }

  mutating func string(maximumBytes: Int) throws -> String {
    let length = Int(try uint32())
    guard length <= maximumBytes else {
      throw PaishoCheckpointWireError.stringTooLong(actual: length, maximum: maximumBytes)
    }
    guard let value = String(bytes: try bytes(count: length), encoding: .utf8) else {
      throw PaishoCheckpointWireError.invalidUTF8
    }
    return value
  }

  func finish() throws {
    guard cursor == bytes.count else {
      throw PaishoCheckpointWireError.trailingBytes(bytes.count - cursor)
    }
  }
}

private struct CheckpointWireWriter {
  var data = Data()

  mutating func bytes(_ values: [UInt8]) {
    data.append(contentsOf: values)
  }

  mutating func uint32(_ value: UInt32) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  mutating func uint64(_ value: UInt64) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }
}
