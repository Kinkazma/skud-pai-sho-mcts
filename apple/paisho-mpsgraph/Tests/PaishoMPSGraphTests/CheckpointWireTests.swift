import Foundation
import XCTest

@testable import PaishoMPSGraph

final class CheckpointWireTests: XCTestCase {
  func testRequestDecodesDurableProgressInRustOrder() throws {
    var writer = CheckpointTestWriter()
    writer.bytes(PaishoCheckpointWireV1.requestMagic)
    writer.uint64(41)
    writer.uint64(12)
    writer.bytes([UInt8](repeating: 0xab, count: 32))
    writer.uint64(9_876)
    writer.uint64(7)
    writer.float(1.0e-4)
    writer.uint32(2)
    writer.string("actor-seed-cursor")
    writer.uint64(91)
    writer.string("replay-sampler")
    writer.uint64(92)
    writer.string("/tmp/generation-0007-step-0012.psckpt")

    let request = try PaishoCheckpointWireV1.decodeRequest(writer.data)
    XCTAssertEqual(request.requestID, 41)
    XCTAssertEqual(request.expectedTrainingStep, 12)
    XCTAssertEqual(request.replaySnapshotSHA256, String(repeating: "ab", count: 32))
    XCTAssertEqual(request.replayIndex, 9_876)
    XCTAssertEqual(request.generation, 7)
    XCTAssertEqual(request.learningRate, 1.0e-4)
    XCTAssertEqual(request.randomStates.map(\.name), ["actor-seed-cursor", "replay-sampler"])
    XCTAssertEqual(request.randomStates.map(\.state), [91, 92])
    XCTAssertEqual(request.destinationPath, "/tmp/generation-0007-step-0012.psckpt")
  }

  func testResponseUsesTheRustProtocolOrder() throws {
    let response = try PaishoCheckpointWireV1.encodeResponse(
      requestID: 51,
      completedTrainingStep: 13,
      completedReplayIndex: 9_884,
      contentSHA256: [UInt8](repeating: 0xcd, count: 32)
    )
    var reader = CheckpointTestReader(response)
    XCTAssertEqual(try reader.bytes(count: 8), PaishoCheckpointWireV1.responseMagic)
    XCTAssertEqual(try reader.uint64(), 51)
    XCTAssertEqual(try reader.uint64(), 13)
    XCTAssertEqual(try reader.uint64(), 9_884)
    XCTAssertEqual(try reader.bytes(count: 32), [UInt8](repeating: 0xcd, count: 32))
    XCTAssertTrue(reader.isAtEnd)
  }

  func testMalformedRequestsRetainTheirRequestID() throws {
    var writer = CheckpointTestWriter()
    writer.bytes(PaishoCheckpointWireV1.requestMagic)
    writer.uint64(61)
    writer.uint64(0)
    writer.bytes([UInt8](repeating: 0, count: 32))
    writer.uint64(0)
    writer.uint64(0)
    writer.float(1.0e-4)
    writer.uint32(2)
    writer.string("same")
    writer.uint64(1)
    writer.string("same")
    writer.uint64(2)
    writer.string("/tmp/a.psckpt")

    XCTAssertEqual(PaishoCheckpointWireV1.requestIDIfPresent(in: writer.data), 61)
    XCTAssertThrowsError(try PaishoCheckpointWireV1.decodeRequest(writer.data)) {
      XCTAssertEqual($0 as? PaishoCheckpointWireError, .duplicateRandomState("same"))
    }
    let error = PaishoCheckpointWireV1.encodeError(
      requestID: 61,
      error: PaishoCheckpointWireError.duplicateRandomState("same")
    )
    var reader = CheckpointTestReader(error)
    XCTAssertEqual(try reader.bytes(count: 8), PaishoCheckpointWireV1.errorMagic)
    XCTAssertEqual(try reader.uint64(), 61)
  }
}

private struct CheckpointTestWriter {
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

  mutating func string(_ value: String) {
    let bytes = Array(value.utf8)
    uint32(UInt32(bytes.count))
    self.bytes(bytes)
  }
}

private struct CheckpointTestReader {
  let bytes: [UInt8]
  var cursor = 0

  init(_ data: Data) {
    bytes = Array(data)
  }

  var isAtEnd: Bool { cursor == bytes.count }

  mutating func bytes(count: Int) throws -> [UInt8] {
    guard cursor <= bytes.count, count <= bytes.count - cursor else {
      throw PaishoCheckpointWireError.truncatedPayload
    }
    defer { cursor += count }
    return Array(bytes[cursor..<(cursor + count)])
  }

  mutating func uint64() throws -> UInt64 {
    try bytes(count: 8).enumerated().reduce(0) { result, item in
      result | UInt64(item.element) << UInt64(item.offset * 8)
    }
  }
}
