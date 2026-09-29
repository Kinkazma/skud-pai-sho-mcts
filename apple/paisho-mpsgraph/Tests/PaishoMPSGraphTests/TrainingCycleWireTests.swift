import Foundation
import XCTest

@testable import PaishoMPSGraph

final class TrainingCycleWireTests: XCTestCase {
  private let old = String(repeating: "ab", count: 32)
  private let next = String(repeating: "cd", count: 32)

  private func payload(
    id: String = "18446744073709551615", previous: String? = nil,
    index: UInt64 = 0, step: UInt64 = 256, generation: UInt64 = 2,
    states: String = "[{\"name\":\"sampler\",\"state\":18446744073709551615}]"
  ) -> Data {
    Data(
      ("PSG1" + """
        {"request_id":\(id),"expected_training_step":256,
         "previous_snapshot_sha256":\(previous ?? "\"\(old)\""),
         "next_progress":{"generation":\(generation),"replayIndex":\(index),
         "replaySnapshotSHA256":"\(next)",
         "scheduler":{"learningRate":0.0001,"completedSteps":\(step)},
         "randomStates":\(states)}}
        """).utf8)
  }

  func testExactSchemaAndUInt64RoundTripWithoutMetal() throws {
    let bytes = payload()
    XCTAssertTrue(PaishoTrainingCycleWireV1.isRequest(bytes))
    let request = try PaishoTrainingCycleWireV1.decodeRequest(bytes)
    XCTAssertEqual(request.requestID, UInt64.max)
    XCTAssertEqual(request.nextProgress.randomStates.first?.state, UInt64.max)
    XCTAssertEqual(PaishoTrainingCycleWireV1.requestIDIfPresent(in: bytes), UInt64.max)
    try request.validateTransition(
      trainingStep: 256, previousSnapshotSHA256: old, previousGeneration: 1)
    let response = try PaishoTrainingCycleWireV1.encodeResponse(
      requestID: request.requestID, progress: request.nextProgress)
    XCTAssertEqual(response.prefix(4), Data("PSGR".utf8))
    struct Response: Decodable {
      let requestID: UInt64
      let trainingStep: UInt64
      let progress: PaishoTrainingProgress
      enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case trainingStep = "training_step"
        case progress
      }
    }
    let decoded = try JSONDecoder().decode(Response.self, from: Data(response.dropFirst(4)))
    XCTAssertEqual(decoded.requestID, UInt64.max)
    XCTAssertEqual(decoded.trainingStep, 256)
    XCTAssertEqual(decoded.progress, request.nextProgress)
    let error = PaishoTrainingCycleWireV1.encodeError(
      requestID: UInt64.max, error: PaishoTrainingCycleWireError.previousSnapshotMismatch)
    XCTAssertEqual(error.prefix(4), Data("PSGE".utf8))
    struct Failure: Decodable {
      let requestID: UInt64
      let message: String
      enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case message
      }
    }
    let failure = try JSONDecoder().decode(Failure.self, from: Data(error.dropFirst(4)))
    XCTAssertEqual(failure.requestID, UInt64.max)
    XCTAssertFalse(failure.message.isEmpty)
  }

  func testOldBindingAndStepMustMatch() throws {
    let request = try PaishoTrainingCycleWireV1.decodeRequest(payload())
    XCTAssertThrowsError(
      try request.validateTransition(
        trainingStep: 255, previousSnapshotSHA256: old, previousGeneration: 1))
    XCTAssertThrowsError(
      try request.validateTransition(
        trainingStep: 256, previousSnapshotSHA256: next, previousGeneration: 1))
    XCTAssertThrowsError(
      try request.validateTransition(
        trainingStep: 256, previousSnapshotSHA256: nil, previousGeneration: 1))
    XCTAssertThrowsError(
      try request.validateTransition(
        trainingStep: 256, previousSnapshotSHA256: old, previousGeneration: 2))
    // A failed validation has no binding to mutate, and does not poison a later valid request.
    try request.validateTransition(
      trainingStep: 256, previousSnapshotSHA256: old, previousGeneration: 1)
    let unbound = try PaishoTrainingCycleWireV1.decodeRequest(payload(previous: "null"))
    try unbound.validateTransition(
      trainingStep: 256, previousSnapshotSHA256: nil, previousGeneration: 1)
    XCTAssertThrowsError(
      try unbound.validateTransition(
        trainingStep: 256, previousSnapshotSHA256: old, previousGeneration: 1))
  }

  func testInvalidWireAndProgressAreRejected() throws {
    for bytes in [
      Data("PSG1{".utf8), Data("PSI1{}".utf8), payload(index: 1), payload(step: 255),
      payload(previous: "\"BAD\""), payload(states: "[]"), payload(id: "-1"),
      payload(id: "18446744073709551616"),
      payload(states: "[{\"name\":\"s\",\"state\":1},{\"name\":\"s\",\"state\":2}]"),
      Data((String(decoding: payload(), as: UTF8.self) + "garbage").utf8),
      Data("PSG1".utf8) + Data(repeating: 32, count: 65_536),
    ] {
      XCTAssertThrowsError(try PaishoTrainingCycleWireV1.decodeRequest(bytes))
    }
    let missingKey = String(decoding: payload(), as: UTF8.self)
      .replacingOccurrences(of: "\"previous_snapshot_sha256\":\"\(old)\",", with: "")
    XCTAssertThrowsError(try PaishoTrainingCycleWireV1.decodeRequest(Data(missingKey.utf8)))
    XCTAssertEqual(PaishoTrainingCycleWireV1.requestIDIfPresent(in: Data("PSG1{".utf8)), 0)
  }
}
