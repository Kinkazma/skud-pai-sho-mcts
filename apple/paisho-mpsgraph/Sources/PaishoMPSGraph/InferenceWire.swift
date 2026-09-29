import Foundation

public enum PaishoInferenceWireV1 {
  public static let requestMagic = Array("PSIREQ01".utf8)
  public static let responseMagic = Array("PSIRSP01".utf8)
  public static let errorMagic = Array("PSIERR01".utf8)

  public static func decodeRequest(
    _ payload: Data,
    expectedShape: PaishoExecutionShape
  ) throws -> (requestID: UInt64, batch: PaishoInferenceBatch) {
    var reader = WireReader(payload)
    guard try reader.bytes(count: 8) == requestMagic else {
      throw PaishoInferenceWireError.invalidRequestMagic
    }
    let requestID = try reader.uint64()
    let batchSize = Int(try reader.uint32())
    let legalActionCapacity = Int(try reader.uint32())
    guard batchSize == expectedShape.batchSize,
      legalActionCapacity == expectedShape.legalActionCapacity
    else {
      throw PaishoInferenceWireError.requestShapeMismatch(
        expected: expectedShape,
        batch: batchSize,
        capacity: legalActionCapacity
      )
    }

    let spatialCount =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
    var examples: [PaishoInferenceExampleV1] = []
    examples.reserveCapacity(batchSize)
    for row in 0..<batchSize {
      let spatial = try reader.floats(count: spatialCount)
      let global = try reader.floats(count: PaishoTensorSchemaV1.globalFeatures)
      let legalCount = Int(try reader.uint32())
      guard legalCount <= legalActionCapacity else {
        throw PaishoInferenceWireError.tooManyLegalActions(
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
        PaishoInferenceExampleV1(
          spatial: spatial,
          global: global,
          legalActions: legalActions
        )
      )
    }
    try reader.finish()
    return (
      requestID,
      try PaishoInferenceBatch.packing(
        examples,
        legalActionCapacity: legalActionCapacity
      )
    )
  }

  public static func encodeResponse(
    requestID: UInt64,
    result: PaishoInferenceResult,
    shape: PaishoExecutionShape
  ) throws -> Data {
    try encodeResponse(
      requestID: requestID,
      policyProbabilities: result.policyProbabilities,
      valueProbabilities: result.valueProbabilities,
      shape: shape
    )
  }

  public static func encodeResponse(
    requestID: UInt64,
    result: PaishoServingInferenceResult,
    shape: PaishoExecutionShape
  ) throws -> Data {
    try encodeResponse(
      requestID: requestID,
      policyProbabilities: result.policyProbabilities,
      valueProbabilities: result.valueProbabilities,
      shape: shape
    )
  }

  private static func encodeResponse(
    requestID: UInt64,
    policyProbabilities: [Float],
    valueProbabilities: [Float],
    shape: PaishoExecutionShape
  ) throws -> Data {
    let policyCount = shape.batchSize * shape.legalActionCapacity
    let valueCount = shape.batchSize * PaishoTensorSchemaV1.valueClasses
    guard let wireBatchSize = UInt32(exactly: shape.batchSize),
      let wireActionCapacity = UInt32(exactly: shape.legalActionCapacity)
    else {
      throw PaishoInferenceWireError.shapeExceedsWireFormat(
        batch: shape.batchSize,
        capacity: shape.legalActionCapacity
      )
    }
    guard policyProbabilities.count == policyCount else {
      throw PaishoInferenceWireError.resultCountMismatch(
        name: "policyProbabilities",
        expected: policyCount,
        actual: policyProbabilities.count
      )
    }
    guard valueProbabilities.count == valueCount else {
      throw PaishoInferenceWireError.resultCountMismatch(
        name: "valueProbabilities",
        expected: valueCount,
        actual: valueProbabilities.count
      )
    }
    var writer = WireWriter()
    writer.bytes(responseMagic)
    writer.uint64(requestID)
    writer.uint32(wireBatchSize)
    writer.uint32(wireActionCapacity)
    writer.floats(policyProbabilities)
    writer.floats(valueProbabilities)
    return writer.data
  }

  public static func encodeError(requestID: UInt64, error: Error) -> Data {
    let message = Data(String(describing: error).prefix(16_384).utf8)
    var writer = WireWriter()
    writer.bytes(errorMagic)
    writer.uint64(requestID)
    writer.uint32(UInt32(message.count))
    writer.data.append(message)
    return writer.data
  }

  public static func requestIDIfPresent(in payload: Data) -> UInt64 {
    guard payload.count >= 16, Array(payload.prefix(8)) == requestMagic else { return 0 }
    var reader = WireReader(payload)
    _ = try? reader.bytes(count: 8)
    return (try? reader.uint64()) ?? 0
  }

  public static func maximumRequestPayloadSize(
    shape: PaishoExecutionShape
  ) throws -> Int {
    guard UInt32(exactly: shape.batchSize) != nil,
      UInt32(exactly: shape.legalActionCapacity) != nil
    else {
      throw PaishoInferenceWireError.shapeExceedsWireFormat(
        batch: shape.batchSize,
        capacity: shape.legalActionCapacity
      )
    }
    let stateFloats =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      + PaishoTensorSchemaV1.globalFeatures
    guard let stateBytes = checkedProduct(stateFloats, 4),
      let actionBytes = checkedProduct(shape.legalActionCapacity, 8),
      let rowBytes = checkedSum(stateBytes, 4, actionBytes),
      let allRows = checkedProduct(shape.batchSize, rowBytes),
      let total = checkedSum(24, allRows)
    else {
      throw PaishoInferenceWireError.payloadSizeOverflow
    }
    return total
  }
}

public enum PaishoInferenceWireError: Error, Equatable, CustomStringConvertible {
  case truncatedPayload
  case trailingBytes(Int)
  case invalidRequestMagic
  case requestShapeMismatch(expected: PaishoExecutionShape, batch: Int, capacity: Int)
  case tooManyLegalActions(row: Int, capacity: Int, actual: Int)
  case resultCountMismatch(name: String, expected: Int, actual: Int)
  case shapeExceedsWireFormat(batch: Int, capacity: Int)
  case payloadSizeOverflow

  public var description: String {
    switch self {
    case .truncatedPayload: "truncated inference payload"
    case .trailingBytes(let count): "inference payload has \(count) trailing bytes"
    case .invalidRequestMagic: "invalid inference request magic"
    case .requestShapeMismatch(let expected, let batch, let capacity):
      "request shape \(batch)x\(capacity) does not match \(expected)"
    case .tooManyLegalActions(let row, let capacity, let actual):
      "row \(row) has \(actual) legal actions; capacity is \(capacity)"
    case .resultCountMismatch(let name, let expected, let actual):
      "\(name) has \(actual) values; expected \(expected)"
    case .shapeExceedsWireFormat(let batch, let capacity):
      "shape \(batch)x\(capacity) exceeds the UInt32 wire format"
    case .payloadSizeOverflow: "inference payload size overflows Int"
    }
  }
}

private struct WireReader {
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

  mutating func floats(count: Int) throws -> [Float] {
    guard let byteCount = checkedProduct(count, MemoryLayout<Float>.size) else {
      throw PaishoInferenceWireError.truncatedPayload
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
      throw PaishoInferenceWireError.trailingBytes(data.count - cursor)
    }
  }

  private mutating func consume(count: Int) throws -> Range<Int> {
    let (end, overflow) = cursor.addingReportingOverflow(count)
    guard !overflow, count >= 0, cursor >= 0, end <= data.count else {
      throw PaishoInferenceWireError.truncatedPayload
    }
    let range = cursor..<end
    cursor = end
    return range
  }
}

private struct WireWriter {
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

  mutating func floats(_ values: [Float]) {
    for value in values {
      uint32(value.bitPattern)
    }
  }
}

private func checkedProduct(_ left: Int, _ right: Int) -> Int? {
  let (result, overflow) = left.multipliedReportingOverflow(by: right)
  return overflow ? nil : result
}

private func checkedSum(_ values: Int...) -> Int? {
  var result = 0
  for value in values {
    let (sum, overflow) = result.addingReportingOverflow(value)
    guard !overflow else { return nil }
    result = sum
  }
  return result
}
