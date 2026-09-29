import Foundation
import XCTest

@testable import PaishoMPSGraph

final class InferenceWireTests: XCTestCase {
  func testRequestDecodesIntoCanonicalInferenceBatch() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 4)
    let payload = requestPayload(shape: shape, requestID: 77)
    let decoded = try PaishoInferenceWireV1.decodeRequest(
      payload,
      expectedShape: shape
    )

    XCTAssertEqual(decoded.requestID, 77)
    XCTAssertEqual(decoded.batch.shape, shape)
    XCTAssertEqual(decoded.batch.familyIndices, [2, 1, 0, 0])
    XCTAssertEqual(decoded.batch.pairIndices, [0, 41_761, 0, 0])
    XCTAssertEqual(decoded.batch.legalMask, [1, 1, 0, 0])
  }

  func testResponseUsesTheRustProtocolOrder() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 4)
    let result = PaishoInferenceResult(
      legalLogits: [4, 3, -1.0e9, -1.0e9],
      policyProbabilities: [0.75, 0.25, 0, 0],
      valueLogits: [1, 2, 3],
      valueProbabilities: [0.2, 0.3, 0.5]
    )
    let payload = try PaishoInferenceWireV1.encodeResponse(
      requestID: 81,
      result: result,
      shape: shape
    )
    var reader = TestReader(payload)

    XCTAssertEqual(try reader.bytes(8), Array("PSIRSP01".utf8))
    XCTAssertEqual(try reader.uint64(), 81)
    XCTAssertEqual(try reader.uint32(), 1)
    XCTAssertEqual(try reader.uint32(), 4)
    XCTAssertEqual(try reader.floats(4), [0.75, 0.25, 0, 0])
    XCTAssertEqual(try reader.floats(3), [0.2, 0.3, 0.5])
    XCTAssertEqual(reader.remaining, 0)
  }

  func testMalformedRequestsReturnStableErrorsAndRecoverTheirID() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 4)
    var trailing = requestPayload(shape: shape, requestID: 91)
    trailing.append(0)
    XCTAssertThrowsError(
      try PaishoInferenceWireV1.decodeRequest(trailing, expectedShape: shape)
    ) { error in
      XCTAssertEqual(error as? PaishoInferenceWireError, .trailingBytes(1))
    }
    XCTAssertEqual(PaishoInferenceWireV1.requestIDIfPresent(in: trailing), 91)

    var wrongShape = requestPayload(shape: shape, requestID: 92)
    wrongShape.replaceSubrange(16..<20, with: UInt32(2).littleEndianBytes)
    XCTAssertThrowsError(
      try PaishoInferenceWireV1.decodeRequest(wrongShape, expectedShape: shape)
    ) { error in
      XCTAssertEqual(
        error as? PaishoInferenceWireError,
        .requestShapeMismatch(expected: shape, batch: 2, capacity: 4)
      )
    }

    XCTAssertEqual(
      try PaishoInferenceWireV1.maximumRequestPayloadSize(shape: shape),
      24
        + (PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
          + PaishoTensorSchemaV1.globalFeatures) * 4 + 4 + 4 * 8
    )

    let oversized = try PaishoExecutionShape(
      batchSize: Int(UInt32.max) + 1,
      legalActionCapacity: 1
    )
    XCTAssertThrowsError(
      try PaishoInferenceWireV1.maximumRequestPayloadSize(shape: oversized)
    ) { error in
      XCTAssertEqual(
        error as? PaishoInferenceWireError,
        .shapeExceedsWireFormat(batch: Int(UInt32.max) + 1, capacity: 1)
      )
    }
  }

  private func requestPayload(
    shape: PaishoExecutionShape,
    requestID: UInt64
  ) -> Data {
    var data = Data(PaishoInferenceWireV1.requestMagic)
    data.append(contentsOf: requestID.littleEndianBytes)
    data.append(contentsOf: UInt32(shape.batchSize).littleEndianBytes)
    data.append(contentsOf: UInt32(shape.legalActionCapacity).littleEndianBytes)
    let spatialCount =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
    for _ in 0..<spatialCount + PaishoTensorSchemaV1.globalFeatures {
      data.append(contentsOf: Float(0).bitPattern.littleEndianBytes)
    }
    data.append(contentsOf: UInt32(2).littleEndianBytes)
    for slots: [UInt16] in [[2, 12, 289, 289], [1, 12, 144, 145]] {
      for slot in slots {
        data.append(contentsOf: slot.littleEndianBytes)
      }
    }
    return data
  }
}

private struct TestReader {
  let data: Data
  var cursor = 0

  init(_ data: Data) {
    self.data = data
  }

  var remaining: Int { data.count - cursor }

  mutating func bytes(_ count: Int) throws -> [UInt8] {
    let end = cursor + count
    guard end <= data.count else { throw PaishoInferenceWireError.truncatedPayload }
    defer { cursor = end }
    return Array(data[cursor..<end])
  }

  mutating func uint32() throws -> UInt32 {
    let values = try bytes(4)
    return values.enumerated().reduce(0) { result, item in
      result | UInt32(item.element) << UInt32(item.offset * 8)
    }
  }

  mutating func uint64() throws -> UInt64 {
    let values = try bytes(8)
    return values.enumerated().reduce(0) { result, item in
      result | UInt64(item.element) << UInt64(item.offset * 8)
    }
  }

  mutating func floats(_ count: Int) throws -> [Float] {
    try (0..<count).map { _ in Float(bitPattern: try uint32()) }
  }
}

extension FixedWidthInteger {
  fileprivate var littleEndianBytes: [UInt8] {
    var value = littleEndian
    return withUnsafeBytes(of: &value) { Array($0) }
  }
}
