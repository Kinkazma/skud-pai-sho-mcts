import Foundation
import XCTest

@testable import PaishoMPSGraph

final class TrainingWireTests: XCTestCase {
  func testRequestDecodesIntoCanonicalTrainingBatch() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    let payload = try requestPayload(shape: shape, requestID: 41, expectedStep: 7)

    let request = try PaishoTrainingWireV1.decodeRequest(payload, expectedShape: shape)
    XCTAssertEqual(request.requestID, 41)
    XCTAssertEqual(request.expectedTrainingStep, 7)
    XCTAssertEqual(request.learningRate, 1.0e-4)
    XCTAssertEqual(request.replaySnapshotSHA256, String(repeating: "ab", count: 32))
    XCTAssertEqual(request.startReplayIndex, 90)
    XCTAssertEqual(request.nextReplayIndex, 91)
    XCTAssertEqual(request.batch.shape, shape)
    XCTAssertEqual(request.batch.policyTargets[0], 0.25)
    XCTAssertEqual(request.batch.policyTargets[1], 0.75)
    XCTAssertTrue(request.batch.policyTargets[2...].allSatisfy { $0 == 0 })
    XCTAssertEqual(request.batch.valueTargets, [1, 0, 0])
    XCTAssertTrue(PaishoTrainingWireV1.isRequest(payload))
    XCTAssertEqual(PaishoTrainingWireV1.requestIDIfPresent(in: payload), 41)
    XCTAssertLessThanOrEqual(
      payload.count,
      try PaishoTrainingWireV1.maximumRequestPayloadSize(shape: shape)
    )
  }

  func testResponseUsesRustProtocolOrder() throws {
    let result = PaishoTrainingResult(
      step: 8,
      policyLoss: 1.25,
      valueLoss: 0.75,
      totalLoss: 2
    )
    let response = PaishoTrainingWireV1.encodeResponse(
      requestID: 51,
      completedReplayIndex: 16,
      result: result
    )
    var reader = TestTrainingWireReader(response)
    XCTAssertEqual(try reader.bytes(count: 8), PaishoTrainingWireV1.responseMagic)
    XCTAssertEqual(try reader.uint64(), 51)
    XCTAssertEqual(try reader.uint64(), 8)
    XCTAssertEqual(try reader.uint64(), 16)
    XCTAssertEqual(try reader.float(), 1.25)
    XCTAssertEqual(try reader.float(), 0.75)
    XCTAssertEqual(try reader.float(), 2)
    XCTAssertTrue(reader.isAtEnd)
  }

  func testMalformedRequestsAreRejectedAndKeepTheirRequestID() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 8)
    var payload = try requestPayload(shape: shape, requestID: 61, expectedStep: 0)
    payload.append(0)
    XCTAssertThrowsError(try PaishoTrainingWireV1.decodeRequest(payload, expectedShape: shape)) {
      XCTAssertEqual($0 as? PaishoTrainingWireError, .trailingBytes(1))
    }
    XCTAssertEqual(PaishoTrainingWireV1.requestIDIfPresent(in: payload), 61)

    let error = PaishoTrainingWireV1.encodeError(
      requestID: 61,
      error: PaishoTrainingWireError.trainingStepMismatch(expected: 0, actual: 1)
    )
    XCTAssertEqual(Array(error.prefix(8)), PaishoTrainingWireV1.errorMagic)
  }

  private func requestPayload(
    shape: PaishoExecutionShape,
    requestID: UInt64,
    expectedStep: UInt64
  ) throws -> Data {
    let actions = [
      try PaishoActionAddressV1(
        slots: [
          2, PaishoActionAddressV1.noTile, PaishoActionAddressV1.noCoordinate,
          PaishoActionAddressV1.noCoordinate,
        ]
      ),
      try PaishoActionAddressV1(
        slots: [1, PaishoActionAddressV1.noTile, 8, 25]
      ),
    ]
    var writer = TestTrainingWireWriter()
    writer.bytes(PaishoTrainingWireV1.requestMagic)
    writer.uint64(requestID)
    writer.uint64(expectedStep)
    writer.float(1.0e-4)
    writer.uint32(UInt32(shape.batchSize))
    writer.uint32(UInt32(shape.legalActionCapacity))
    writer.bytes([UInt8](repeating: 0xab, count: 32))
    writer.uint64(90)
    writer.uint64(91)
    writer.floats(
      [Float](
        repeating: 0,
        count: PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      )
    )
    writer.floats([Float](repeating: 0, count: PaishoTensorSchemaV1.globalFeatures))
    writer.uint32(UInt32(actions.count))
    for action in actions {
      for slot in action.slots {
        writer.uint16(slot)
      }
    }
    writer.floats([0.25, 0.75])
    writer.floats([1, 0, 0])
    return writer.data
  }
}

private struct TestTrainingWireWriter {
  var data = Data()

  mutating func bytes(_ values: [UInt8]) {
    data.append(contentsOf: values)
  }

  mutating func uint16(_ value: UInt16) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
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

  mutating func floats(_ values: [Float]) {
    for value in values { float(value) }
  }
}

private struct TestTrainingWireReader {
  let data: Data
  var cursor = 0

  init(_ data: Data) {
    self.data = data
  }

  var isAtEnd: Bool { cursor == data.count }

  mutating func bytes(count: Int) throws -> [UInt8] {
    guard cursor + count <= data.count else {
      throw PaishoTrainingWireError.truncatedPayload
    }
    let result = Array(data[cursor..<cursor + count])
    cursor += count
    return result
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
}
