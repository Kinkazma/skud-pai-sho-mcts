import Foundation

public struct PaishoInferenceExampleV1: Sendable {
  public let spatial: [Float]
  public let global: [Float]
  public let legalActions: [PaishoActionAddressV1]

  public init(
    spatial: [Float],
    global: [Float],
    legalActions: [PaishoActionAddressV1]
  ) {
    self.spatial = spatial
    self.global = global
    self.legalActions = legalActions
  }
}

public struct PaishoInferenceBatch: Sendable {
  public let shape: PaishoExecutionShape
  public let spatial: [Float]
  public let global: [Float]
  public let familyIndices: [Int32]
  public let tileIndices: [Int32]
  public let tilePresence: [Float]
  public let destinationIndices: [Int32]
  public let destinationPresence: [Float]
  public let pairIndices: [Int32]
  public let pairPresence: [Float]
  public let legalMask: [Float]

  public init(
    shape: PaishoExecutionShape,
    spatial: [Float],
    global: [Float],
    familyIndices: [Int32],
    tileIndices: [Int32],
    tilePresence: [Float],
    destinationIndices: [Int32],
    destinationPresence: [Float],
    pairIndices: [Int32],
    pairPresence: [Float],
    legalMask: [Float]
  ) throws {
    self.shape = shape
    self.spatial = spatial
    self.global = global
    self.familyIndices = familyIndices
    self.tileIndices = tileIndices
    self.tilePresence = tilePresence
    self.destinationIndices = destinationIndices
    self.destinationPresence = destinationPresence
    self.pairIndices = pairIndices
    self.pairPresence = pairPresence
    self.legalMask = legalMask
    try validate()
  }

  public static func packing(
    _ examples: [PaishoInferenceExampleV1],
    legalActionCapacity: Int
  ) throws -> Self {
    let shape = try PaishoExecutionShape(
      batchSize: examples.count,
      legalActionCapacity: legalActionCapacity
    )
    let actionValueCount = shape.batchSize * shape.legalActionCapacity
    var spatial: [Float] = []
    var global: [Float] = []
    spatial.reserveCapacity(
      shape.batchSize * PaishoTensorSchemaV1.boardCells
        * PaishoTensorSchemaV1.spatialChannels
    )
    global.reserveCapacity(shape.batchSize * PaishoTensorSchemaV1.globalFeatures)
    var familyIndices = [Int32](repeating: 0, count: actionValueCount)
    var tileIndices = [Int32](repeating: 0, count: actionValueCount)
    var tilePresence = [Float](repeating: 0, count: actionValueCount)
    var destinationIndices = [Int32](repeating: 0, count: actionValueCount)
    var destinationPresence = [Float](repeating: 0, count: actionValueCount)
    var pairIndices = [Int32](repeating: 0, count: actionValueCount)
    var pairPresence = [Float](repeating: 0, count: actionValueCount)
    var legalMask = [Float](repeating: 0, count: actionValueCount)

    for (row, example) in examples.enumerated() {
      let expectedSpatialCount =
        PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels
      guard example.spatial.count == expectedSpatialCount else {
        throw BatchError.wrongExampleCount(
          name: "spatial",
          row: row,
          expected: expectedSpatialCount,
          actual: example.spatial.count
        )
      }
      guard example.global.count == PaishoTensorSchemaV1.globalFeatures else {
        throw BatchError.wrongExampleCount(
          name: "global",
          row: row,
          expected: PaishoTensorSchemaV1.globalFeatures,
          actual: example.global.count
        )
      }
      guard !example.legalActions.isEmpty else {
        throw BatchError.noLegalAction(row: row)
      }
      guard example.legalActions.count <= legalActionCapacity else {
        throw BatchError.tooManyLegalActions(
          row: row,
          capacity: legalActionCapacity,
          actual: example.legalActions.count
        )
      }
      spatial.append(contentsOf: example.spatial)
      global.append(contentsOf: example.global)
      let rowStart = row * legalActionCapacity
      for (column, address) in example.legalActions.enumerated() {
        let index = rowStart + column
        let components = address.components
        familyIndices[index] = components.familyIndex
        tileIndices[index] = components.tileIndex
        tilePresence[index] = components.tilePresence
        destinationIndices[index] = components.destinationIndex
        destinationPresence[index] = components.destinationPresence
        pairIndices[index] = components.pairIndex
        pairPresence[index] = components.pairPresence
        legalMask[index] = 1
      }
    }
    return try Self(
      shape: shape,
      spatial: spatial,
      global: global,
      familyIndices: familyIndices,
      tileIndices: tileIndices,
      tilePresence: tilePresence,
      destinationIndices: destinationIndices,
      destinationPresence: destinationPresence,
      pairIndices: pairIndices,
      pairPresence: pairPresence,
      legalMask: legalMask
    )
  }

  public func validate() throws {
    let batch = shape.batchSize
    let capacity = shape.legalActionCapacity
    let actionValues = batch * capacity
    try expectCount(
      spatial,
      batch * PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.spatialChannels,
      "spatial"
    )
    try expectCount(global, batch * PaishoTensorSchemaV1.globalFeatures, "global")
    try expectCount(familyIndices, actionValues, "familyIndices")
    try expectCount(tileIndices, actionValues, "tileIndices")
    try expectCount(tilePresence, actionValues, "tilePresence")
    try expectCount(destinationIndices, actionValues, "destinationIndices")
    try expectCount(destinationPresence, actionValues, "destinationPresence")
    try expectCount(pairIndices, actionValues, "pairIndices")
    try expectCount(pairPresence, actionValues, "pairPresence")
    try expectCount(legalMask, actionValues, "legalMask")
    try requireFinite(spatial, name: "spatial")
    try requireFinite(global, name: "global")

    for row in 0..<batch {
      let actions = row * capacity..<(row + 1) * capacity
      try requireBinary(legalMask[actions], name: "legalMask")
      try requireBinary(tilePresence[actions], name: "tilePresence")
      try requireBinary(destinationPresence[actions], name: "destinationPresence")
      try requireBinary(pairPresence[actions], name: "pairPresence")
      guard legalMask[actions].contains(1) else {
        throw BatchError.noLegalAction(row: row)
      }
    }
    try validateIndices()
    try validateActionComponents()
  }

  private func validateIndices() throws {
    for index in familyIndices where index < 0 || index >= PaishoTensorSchemaV1.actionFamilies {
      throw BatchError.indexOutOfRange(name: "familyIndices", value: index)
    }
    for index in tileIndices where index < 0 || index >= PaishoTensorSchemaV1.tileKinds {
      throw BatchError.indexOutOfRange(name: "tileIndices", value: index)
    }
    for index in destinationIndices where index < 0 || index >= PaishoTensorSchemaV1.boardCells {
      throw BatchError.indexOutOfRange(name: "destinationIndices", value: index)
    }
    let pairCount = PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.boardCells
    for index in pairIndices where index < 0 || index >= pairCount {
      throw BatchError.indexOutOfRange(name: "pairIndices", value: index)
    }
  }

  private func validateActionComponents() throws {
    let capacity = shape.legalActionCapacity
    let boardCells = Int32(PaishoTensorSchemaV1.boardCells)
    for row in 0..<shape.batchSize {
      for column in 0..<capacity {
        let index = row * capacity + column
        guard legalMask[index] == 1 else { continue }

        let tile =
          tilePresence[index] == 1
          ? UInt16(tileIndices[index]) : PaishoActionAddressV1.noTile
        let destination =
          destinationPresence[index] == 1
          ? UInt16(destinationIndices[index]) : PaishoActionAddressV1.noCoordinate
        let source =
          pairPresence[index] == 1
          ? UInt16(pairIndices[index] / boardCells) : PaishoActionAddressV1.noCoordinate
        let address: PaishoActionAddressV1
        do {
          address = try PaishoActionAddressV1(
            slots: [UInt16(familyIndices[index]), tile, source, destination]
          )
        } catch {
          throw BatchError.invalidActionComponents(row: row, index: column)
        }
        let expected = address.components
        guard expected.familyIndex == familyIndices[index],
          expected.tileIndex == tileIndices[index],
          expected.tilePresence == tilePresence[index],
          expected.destinationIndex == destinationIndices[index],
          expected.destinationPresence == destinationPresence[index],
          expected.pairIndex == pairIndices[index],
          expected.pairPresence == pairPresence[index]
        else {
          throw BatchError.invalidActionComponents(row: row, index: column)
        }
      }
    }
  }
}
