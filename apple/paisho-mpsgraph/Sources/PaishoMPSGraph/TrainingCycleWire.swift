import Foundation

/// PSG1 + UTF-8 JSON inside the existing UInt64 little-endian length framing.
/// Outer keys are snake_case; next_progress uses the checkpoint's camelCase keys.
public struct PaishoTrainingCycleRequestV1: Decodable, Sendable {
  public let requestID: UInt64
  public let expectedTrainingStep: UInt64
  public let previousSnapshotSHA256: String?
  public let nextProgress: PaishoTrainingProgress

  enum CodingKeys: String, CodingKey {
    case requestID = "request_id"
    case expectedTrainingStep = "expected_training_step"
    case previousSnapshotSHA256 = "previous_snapshot_sha256"
    case nextProgress = "next_progress"
  }

  public init(from decoder: any Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    requestID = try values.decode(UInt64.self, forKey: .requestID)
    expectedTrainingStep = try values.decode(UInt64.self, forKey: .expectedTrainingStep)
    // decode(Optional.self) requires the key, unlike decodeIfPresent.
    previousSnapshotSHA256 = try values.decode(String?.self, forKey: .previousSnapshotSHA256)
    nextProgress = try values.decode(PaishoTrainingProgress.self, forKey: .nextProgress)
  }

  /// Entire transition is checked before the caller replaces its service-owned binding.
  public func validateTransition(
    trainingStep: UInt64, previousSnapshotSHA256: String?, previousGeneration: UInt64?
  ) throws {
    guard expectedTrainingStep == trainingStep else {
      throw PaishoTrainingCycleWireError.trainingStepMismatch(
        expected: expectedTrainingStep, actual: trainingStep)
    }
    guard self.previousSnapshotSHA256 == previousSnapshotSHA256 else {
      throw PaishoTrainingCycleWireError.previousSnapshotMismatch
    }
    try PaishoMPSGraphModel.validateNewGeneration(
      progress: nextProgress, trainingStep: trainingStep, previousGeneration: previousGeneration)
  }
}

public enum PaishoTrainingCycleWireError: Error, Equatable {
  case invalidMagic
  case payloadTooLarge
  case invalidPreviousSnapshot
  case previousSnapshotMismatch
  case trainingStepMismatch(expected: UInt64, actual: UInt64)
  case checkpointGenerationMismatch(expected: UInt64, actual: UInt64)
}

public enum PaishoTrainingCycleWireV1 {
  public static let requestMagic = Data("PSG1".utf8)
  public static let responseMagic = Data("PSGR".utf8)
  public static let errorMagic = Data("PSGE".utf8)
  public static let maximumRequestPayloadSize = 65_536

  public static func isRequest(_ payload: Data) -> Bool { payload.starts(with: requestMagic) }

  public static func decodeRequest(_ payload: Data) throws -> PaishoTrainingCycleRequestV1 {
    guard payload.count <= maximumRequestPayloadSize else {
      throw PaishoTrainingCycleWireError.payloadTooLarge
    }
    guard isRequest(payload) else { throw PaishoTrainingCycleWireError.invalidMagic }
    let request = try JSONDecoder().decode(
      PaishoTrainingCycleRequestV1.self, from: Data(payload.dropFirst(4)))
    if let digest = request.previousSnapshotSHA256 {
      guard digest.utf8.count == 64,
        digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
      else { throw PaishoTrainingCycleWireError.invalidPreviousSnapshot }
    }
    try PaishoMPSGraphModel.validateNewGeneration(
      progress: request.nextProgress, trainingStep: request.expectedTrainingStep,
      previousGeneration: nil)
    return request
  }

  public static func requestIDIfPresent(in payload: Data) -> UInt64 {
    struct Identifier: Decodable {
      let requestID: UInt64
      enum CodingKeys: String, CodingKey { case requestID = "request_id" }
    }
    guard isRequest(payload), payload.count <= maximumRequestPayloadSize else { return 0 }
    return
      (try? JSONDecoder().decode(
        Identifier.self, from: Data(payload.dropFirst(4))
      ).requestID) ?? 0
  }

  public static func encodeResponse(requestID: UInt64, progress: PaishoTrainingProgress) throws
    -> Data
  {
    struct Response: Encodable {
      let requestID: UInt64
      let trainingStep: UInt64
      let progress: PaishoTrainingProgress
      enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case trainingStep = "training_step"
        case progress
      }
    }
    return responseMagic
      + (try JSONEncoder().encode(
        Response(
          requestID: requestID, trainingStep: progress.scheduler.completedSteps, progress: progress)
      ))
  }

  public static func encodeError(requestID: UInt64, error: Error) -> Data {
    // JSONSerialization on these string/number-only values cannot fail.
    let body: [String: Any] = [
      "request_id": NSNumber(value: requestID),
      "message": String(String(describing: error).prefix(16_384)),
    ]
    return errorMagic + (try! JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]))
  }
}
