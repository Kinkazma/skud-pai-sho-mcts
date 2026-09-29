import Foundation

public enum PaishoTerminalPpoWireV1 {
  public static let objectiveID = "ppo-terminal-v1"
  public static let requestMagic = Array("PSTREQ02".utf8)
  public static let durationRequestMagic = Array("PSTREQ03".utf8)
  public static let responseMagic = Array("PSTRSP03".utf8)
  public static let errorMagic = Array("PSTERR02".utf8)

  public static func isRequest(_ payload: Data) -> Bool {
    payload.count >= 8 && [requestMagic, durationRequestMagic].contains(Array(payload.prefix(8)))
  }

  public static func decodeRequest(
    _ payload: Data,
    expectedShape: PaishoExecutionShape
  ) throws -> (
    requestID: UInt64,
    expectedTrainingStep: UInt64,
    learningRate: Float,
    parameters: PaishoTerminalPpoParametersV1,
    replaySnapshotSHA256: String,
    startReplayIndex: UInt64,
    nextReplayIndex: UInt64,
    batch: PaishoTerminalPpoBatch
  ) {
    var reader = TrainingWireReader(payload)
    let magic = try reader.bytes(count: 8)
    guard [requestMagic, durationRequestMagic].contains(magic) else {
      throw PaishoTerminalPpoWireError.invalidRequestMagic
    }
    let requestID = try reader.uint64()
    let expectedTrainingStep = try reader.uint64()
    guard expectedTrainingStep < UInt64.max else {
      throw PaishoTerminalPpoWireError.trainingStepOverflow
    }
    let learningRate = try reader.float()
    guard learningRate.isFinite, learningRate > 0 else {
      throw PaishoTerminalPpoWireError.invalidLearningRate(learningRate)
    }
    let parameters = try PaishoTerminalPpoParametersV1(
      policyTemperature: reader.float(),
      uniformMix: reader.float(),
      clipEpsilon: reader.float(),
      valueLossWeight: reader.float(),
      entropyWeight: reader.float()
    )
    let batchSize = Int(try reader.uint32())
    let legalActionCapacity = Int(try reader.uint32())
    guard batchSize == expectedShape.batchSize,
      legalActionCapacity == expectedShape.legalActionCapacity
    else {
      throw PaishoTerminalPpoWireError.requestShapeMismatch(
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
      throw PaishoTerminalPpoWireError.invalidReplayRange(
        start: startReplayIndex,
        next: nextReplayIndex,
        batch: batchSize
      )
    }

    let spatialCount =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
    var examples: [PaishoTerminalPpoExampleV1] = []
    examples.reserveCapacity(batchSize)
    for row in 0..<batchSize {
      let spatial = try reader.floats(count: spatialCount)
      let global = try reader.floats(count: PaishoTensorSchemaV1.globalFeatures)
      let legalCount = Int(try reader.uint32())
      guard legalCount <= legalActionCapacity else {
        throw PaishoTerminalPpoWireError.tooManyLegalActions(
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
      let playedActionIndex = Int(try reader.uint32())
      let behaviorProbability = try reader.float()
      let terminalCode = try reader.uint32()
      guard let terminalValue = PaishoValueClassV1(rawValue: Int(terminalCode)) else {
        throw PaishoTerminalPpoWireError.invalidTerminalValue(row: row, value: terminalCode)
      }
      let actorValue = try reader.float()
      let policyReturn: Float? = magic == durationRequestMagic ? try reader.float() : nil
      examples.append(
        PaishoTerminalPpoExampleV1(
          inference: PaishoInferenceExampleV1(
            spatial: spatial,
            global: global,
            legalActions: legalActions
          ),
          playedActionIndex: playedActionIndex,
          behaviorProbability: behaviorProbability,
          terminalValue: terminalValue,
          actorValue: actorValue,
          policyReturn: policyReturn
        )
      )
    }
    try reader.finish()
    return (
      requestID,
      expectedTrainingStep,
      learningRate,
      parameters,
      replaySnapshotSHA256,
      startReplayIndex,
      nextReplayIndex,
      try PaishoTerminalPpoBatch.packing(
        examples,
        legalActionCapacity: legalActionCapacity
      )
    )
  }

  public static func encodeResponse(
    requestID: UInt64,
    completedReplayIndex: UInt64,
    result: PaishoTerminalPpoResult
  ) -> Data {
    var writer = TrainingWireWriter()
    writer.bytes(responseMagic)
    writer.uint64(requestID)
    writer.uint64(result.step)
    writer.uint64(completedReplayIndex)
    writer.float(result.policyLoss)
    writer.float(result.valueLoss)
    writer.float(result.entropy)
    writer.float(result.totalLoss)
    writer.float(result.meanAdvantage)
    writer.float(result.meanImportanceRatio)
    writer.float(result.meanSquaredRatioDeviation)
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
      throw PaishoTerminalPpoWireError.shapeExceedsWireFormat(
        batch: shape.batchSize,
        capacity: shape.legalActionCapacity
      )
    }
    let stateFloats =
      PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      + PaishoTensorSchemaV1.globalFeatures
    guard let stateBytes = checkedTrainingProduct(stateFloats, 4),
      let actionBytes = checkedTrainingProduct(shape.legalActionCapacity, 8),
      let rowBytes = checkedTrainingSum(stateBytes, 4, actionBytes, 20),
      let allRows = checkedTrainingProduct(shape.batchSize, rowBytes),
      let total = checkedTrainingSum(104, allRows)
    else {
      throw PaishoTerminalPpoWireError.payloadSizeOverflow
    }
    return total
  }
}

public enum PaishoTerminalPpoWireError: Error, Equatable, CustomStringConvertible {
  case invalidRequestMagic
  case trainingStepOverflow
  case invalidLearningRate(Float)
  case requestShapeMismatch(expected: PaishoExecutionShape, batch: Int, capacity: Int)
  case tooManyLegalActions(row: Int, capacity: Int, actual: Int)
  case invalidTerminalValue(row: Int, value: UInt32)
  case invalidReplayRange(start: UInt64, next: UInt64, batch: Int)
  case trainingStepMismatch(expected: UInt64, actual: UInt64)
  case replaySnapshotMismatch(expected: String, actual: String)
  case replayIndexMismatch(expected: UInt64, actual: UInt64)
  case learningRateMismatch(expected: Float, actual: Float)
  case parametersMismatch(
    expected: PaishoTerminalPpoParametersV1,
    actual: PaishoTerminalPpoParametersV1
  )
  case shapeExceedsWireFormat(batch: Int, capacity: Int)
  case payloadSizeOverflow

  public var description: String {
    switch self {
    case .invalidRequestMagic: "invalid terminal PPO request magic"
    case .trainingStepOverflow: "terminal PPO training step would overflow"
    case .invalidLearningRate(let value):
      "terminal PPO learning rate \(value) is not positive and finite"
    case .requestShapeMismatch(let expected, let batch, let capacity):
      "terminal PPO request shape \(batch)x\(capacity) does not match \(expected)"
    case .tooManyLegalActions(let row, let capacity, let actual):
      "terminal PPO row \(row) has \(actual) legal actions; capacity is \(capacity)"
    case .invalidTerminalValue(let row, let value):
      "terminal PPO row \(row) has unknown terminal value \(value)"
    case .invalidReplayRange(let start, let next, let batch):
      "terminal PPO replay range \(start)..<\(next) does not contain batch \(batch)"
    case .trainingStepMismatch(let expected, let actual):
      "terminal PPO request expected step \(expected); model is at \(actual)"
    case .replaySnapshotMismatch(let expected, let actual):
      "terminal PPO snapshot \(actual) does not match \(expected)"
    case .replayIndexMismatch(let expected, let actual):
      "terminal PPO replay index \(actual) does not match \(expected)"
    case .learningRateMismatch(let expected, let actual):
      "terminal PPO learning rate \(actual) does not match \(expected)"
    case .parametersMismatch(let expected, let actual):
      "terminal PPO parameters \(actual) do not match \(expected)"
    case .shapeExceedsWireFormat(let batch, let capacity):
      "terminal PPO shape \(batch)x\(capacity) exceeds the UInt32 wire format"
    case .payloadSizeOverflow: "terminal PPO payload size overflows Int"
    }
  }
}
