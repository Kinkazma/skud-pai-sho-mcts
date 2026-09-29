import Foundation

public enum PaishoTrainingWireV1 {
  public static let requestMagic = Array("PSTREQ01".utf8)
  public static let responseMagic = Array("PSTRSP01".utf8)
  public static let errorMagic = Array("PSTERR01".utf8)

  public static func isRequest(_ payload: Data) -> Bool {
    payload.count >= requestMagic.count && Array(payload.prefix(requestMagic.count)) == requestMagic
  }

  public static func decodeRequest(
    _ payload: Data,
    expectedShape: PaishoExecutionShape
  ) throws -> (
    requestID: UInt64,
    expectedTrainingStep: UInt64,
    learningRate: Float,
    replaySnapshotSHA256: String,
    startReplayIndex: UInt64,
    nextReplayIndex: UInt64,
    batch: PaishoTrainingBatch
  ) {
    var reader = TrainingWireReader(payload)
    guard try reader.bytes(count: 8) == requestMagic else {
      throw PaishoTrainingWireError.invalidRequestMagic
    }
    let requestID = try reader.uint64()
    let expectedTrainingStep = try reader.uint64()
    guard expectedTrainingStep < UInt64.max else {
      throw PaishoTrainingWireError.trainingStepOverflow
    }
    let learningRate = try reader.float()
    guard learningRate.isFinite, learningRate > 0 else {
      throw PaishoTrainingWireError.invalidLearningRate(learningRate)
    }
    let batchSize = Int(try reader.uint32())
    let legalActionCapacity = Int(try reader.uint32())
    guard batchSize == expectedShape.batchSize,
      legalActionCapacity == expectedShape.legalActionCapacity
    else {
      throw PaishoTrainingWireError.requestShapeMismatch(
        expected: expectedShape,
        batch: batchSize,
        capacity: legalActionCapacity
      )
    }
    let replaySnapshotSHA256 = try reader.bytes(count: 32)
      .map { String(format: "%02x", $0) }
      .joined()
    let startReplayIndex = try reader.uint64()
    let nextReplayIndex = try reader.uint64()
    let (expectedNextReplayIndex, replayOverflow) = startReplayIndex.addingReportingOverflow(
      UInt64(batchSize)
    )
    guard !replayOverflow, nextReplayIndex == expectedNextReplayIndex else {
      throw PaishoTrainingWireError.invalidReplayRange(
        start: startReplayIndex,
        next: nextReplayIndex,
        batch: batchSize
      )
    }

    let spatialCount =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
    var examples: [PaishoTrainingExampleV1] = []
    examples.reserveCapacity(batchSize)
    for row in 0..<batchSize {
      let spatial = try reader.floats(count: spatialCount)
      let global = try reader.floats(count: PaishoTensorSchemaV1.globalFeatures)
      let legalCount = Int(try reader.uint32())
      guard legalCount <= legalActionCapacity else {
        throw PaishoTrainingWireError.tooManyLegalActions(
          row: row,
          capacity: legalActionCapacity,
          actual: legalCount
        )
      }
      var legalActions: [PaishoActionAddressV1] = []
      legalActions.reserveCapacity(legalCount)
      for _ in 0..<legalCount {
        legalActions.append(
          try PaishoActionAddressV1(
            slots: [
              try reader.uint16(),
              try reader.uint16(),
              try reader.uint16(),
              try reader.uint16(),
            ]
          )
        )
      }
      examples.append(
        PaishoTrainingExampleV1(
          inference: PaishoInferenceExampleV1(
            spatial: spatial,
            global: global,
            legalActions: legalActions
          ),
          policyTargets: try reader.floats(count: legalCount),
          valueTargets: try reader.floats(count: PaishoTensorSchemaV1.valueClasses)
        )
      )
    }
    try reader.finish()
    return (
      requestID,
      expectedTrainingStep,
      learningRate,
      replaySnapshotSHA256,
      startReplayIndex,
      nextReplayIndex,
      try PaishoTrainingBatch.packing(
        examples,
        legalActionCapacity: legalActionCapacity
      )
    )
  }

  public static func encodeResponse(
    requestID: UInt64,
    completedReplayIndex: UInt64,
    result: PaishoTrainingResult
  ) -> Data {
    var writer = TrainingWireWriter()
    writer.bytes(responseMagic)
    writer.uint64(requestID)
    writer.uint64(result.step)
    writer.uint64(completedReplayIndex)
    writer.float(result.policyLoss)
    writer.float(result.valueLoss)
    writer.float(result.totalLoss)
    return writer.data
  }

  public static func encodeError(requestID: UInt64, error: Error) -> Data {
    let message = Data(String(describing: error).prefix(16_384).utf8)
    var writer = TrainingWireWriter()
    writer.bytes(errorMagic)
    writer.uint64(requestID)
    writer.uint32(UInt32(message.count))
    writer.data.append(message)
    return writer.data
  }

  public static func requestIDIfPresent(in payload: Data) -> UInt64 {
    guard payload.count >= 16, isRequest(payload) else { return 0 }
    var reader = TrainingWireReader(payload)
    _ = try? reader.bytes(count: 8)
    return (try? reader.uint64()) ?? 0
  }

  public static func maximumRequestPayloadSize(shape: PaishoExecutionShape) throws -> Int {
    guard UInt32(exactly: shape.batchSize) != nil,
      UInt32(exactly: shape.legalActionCapacity) != nil
    else {
      throw PaishoTrainingWireError.shapeExceedsWireFormat(
        batch: shape.batchSize,
        capacity: shape.legalActionCapacity
      )
    }
    let stateFloats =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      + PaishoTensorSchemaV1.globalFeatures
    guard let stateBytes = checkedTrainingProduct(stateFloats, 4),
      let actionAndPolicyBytes = checkedTrainingProduct(shape.legalActionCapacity, 12),
      let valueBytes = checkedTrainingProduct(PaishoTensorSchemaV1.valueClasses, 4),
      let rowBytes = checkedTrainingSum(stateBytes, 4, actionAndPolicyBytes, valueBytes),
      let allRows = checkedTrainingProduct(shape.batchSize, rowBytes),
      let total = checkedTrainingSum(84, allRows)
    else {
      throw PaishoTrainingWireError.payloadSizeOverflow
    }
    return total
  }
}

public enum PaishoTrainingWireError: Error, Equatable, CustomStringConvertible {
  case truncatedPayload
  case trailingBytes(Int)
  case invalidRequestMagic
  case trainingStepOverflow
  case invalidLearningRate(Float)
  case requestShapeMismatch(expected: PaishoExecutionShape, batch: Int, capacity: Int)
  case tooManyLegalActions(row: Int, capacity: Int, actual: Int)
  case invalidReplayRange(start: UInt64, next: UInt64, batch: Int)
  case trainingStepMismatch(expected: UInt64, actual: UInt64)
  case replaySnapshotMismatch(expected: String, actual: String)
  case replayIndexMismatch(expected: UInt64, actual: UInt64)
  case learningRateMismatch(expected: Float, actual: Float)
  case shapeExceedsWireFormat(batch: Int, capacity: Int)
  case payloadSizeOverflow

  public var description: String {
    switch self {
    case .truncatedPayload: "truncated training payload"
    case .trailingBytes(let count): "training payload has \(count) trailing bytes"
    case .invalidRequestMagic: "invalid training request magic"
    case .trainingStepOverflow: "training step would overflow"
    case .invalidLearningRate(let value):
      "training learning rate \(value) is not positive and finite"
    case .requestShapeMismatch(let expected, let batch, let capacity):
      "training request shape \(batch)x\(capacity) does not match \(expected)"
    case .tooManyLegalActions(let row, let capacity, let actual):
      "training row \(row) has \(actual) legal actions; capacity is \(capacity)"
    case .invalidReplayRange(let start, let next, let batch):
      "training replay range \(start)..<\(next) does not contain batch \(batch)"
    case .trainingStepMismatch(let expected, let actual):
      "training request expected step \(expected); model is at \(actual)"
    case .replaySnapshotMismatch(let expected, let actual):
      "training snapshot \(actual) does not match \(expected)"
    case .replayIndexMismatch(let expected, let actual):
      "training replay index \(actual) does not match \(expected)"
    case .learningRateMismatch(let expected, let actual):
      "training learning rate \(actual) does not match \(expected)"
    case .shapeExceedsWireFormat(let batch, let capacity):
      "training shape \(batch)x\(capacity) exceeds the UInt32 wire format"
    case .payloadSizeOverflow: "training payload size overflows Int"
    }
  }
}

struct TrainingWireReader {
  let data: Data
  var cursor = 0

  init(_ data: Data) {
    self.data = data
  }

  mutating func bytes(count: Int) throws -> [UInt8] {
    let range = try consume(count: count)
    return Array(data[range])
  }

  mutating func uint16() throws -> UInt16 {
    let range = try consume(count: MemoryLayout<UInt16>.size)
    return data.withUnsafeBytes { source in
      UInt16(littleEndian: source.loadUnaligned(fromByteOffset: range.lowerBound, as: UInt16.self))
    }
  }

  mutating func uint32() throws -> UInt32 {
    let range = try consume(count: MemoryLayout<UInt32>.size)
    return data.withUnsafeBytes { source in
      UInt32(littleEndian: source.loadUnaligned(fromByteOffset: range.lowerBound, as: UInt32.self))
    }
  }

  mutating func uint64() throws -> UInt64 {
    let range = try consume(count: MemoryLayout<UInt64>.size)
    return data.withUnsafeBytes { source in
      UInt64(littleEndian: source.loadUnaligned(fromByteOffset: range.lowerBound, as: UInt64.self))
    }
  }

  mutating func float() throws -> Float {
    Float(bitPattern: try uint32())
  }

  mutating func floats(count: Int) throws -> [Float] {
    guard let byteCount = checkedTrainingProduct(count, MemoryLayout<Float>.size) else {
      throw PaishoTrainingWireError.truncatedPayload
    }
    let range = try consume(count: byteCount)
    return [Float](unsafeUninitializedCapacity: count) { destination, initializedCount in
      data.withUnsafeBytes { source in
        let sourceSlice = UnsafeRawBufferPointer(
          start: source.baseAddress?.advanced(by: range.lowerBound),
          count: byteCount
        )
        let destinationBytes = UnsafeMutableRawBufferPointer(
          start: destination.baseAddress,
          count: byteCount
        )
        destinationBytes.copyMemory(from: sourceSlice)
      }
      initializedCount = count
    }
  }

  func finish() throws {
    guard cursor == data.count else {
      throw PaishoTrainingWireError.trailingBytes(data.count - cursor)
    }
  }

  private mutating func consume(count: Int) throws -> Range<Int> {
    let (end, overflow) = cursor.addingReportingOverflow(count)
    guard !overflow, count >= 0, cursor >= 0, end <= data.count else {
      throw PaishoTrainingWireError.truncatedPayload
    }
    let range = cursor..<end
    cursor = end
    return range
  }
}

struct TrainingWireWriter {
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

  mutating func float(_ value: Float) {
    uint32(value.bitPattern)
  }
}

func checkedTrainingProduct(_ left: Int, _ right: Int) -> Int? {
  let (result, overflow) = left.multipliedReportingOverflow(by: right)
  return overflow ? nil : result
}

func checkedTrainingSum(_ values: Int...) -> Int? {
  var result = 0
  for value in values {
    let (sum, overflow) = result.addingReportingOverflow(value)
    guard !overflow else { return nil }
    result = sum
  }
  return result
}
