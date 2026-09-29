import Foundation

public struct PaishoActionAddressV1: Equatable, Sendable {
  public static let noTile: UInt16 = 12
  public static let noCoordinate: UInt16 = 289

  public let slots: [UInt16]

  public init(slots: [UInt16]) throws {
    guard slots.count == 4 else {
      throw PaishoActionAddressError.wrongSlotCount(slots.count)
    }
    guard slots[0] < UInt16(PaishoTensorSchemaV1.actionFamilies) else {
      throw PaishoActionAddressError.invalidFamily(slots[0])
    }
    guard slots[1] <= Self.noTile else {
      throw PaishoActionAddressError.invalidTile(slots[1])
    }
    guard slots[2] <= Self.noCoordinate else {
      throw PaishoActionAddressError.invalidCoordinate(slots[2])
    }
    guard slots[3] <= Self.noCoordinate else {
      throw PaishoActionAddressError.invalidCoordinate(slots[3])
    }
    for coordinate in slots[2...3]
    where coordinate != Self.noCoordinate && !Self.isPlayable(coordinate) {
      throw PaishoActionAddressError.nonPlayableCoordinate(coordinate)
    }
    guard Self.hasValidShape(slots) else {
      throw PaishoActionAddressError.invalidShape(family: slots[0])
    }
    self.slots = slots
  }

  var components: PaishoPolicyComponentsV1 {
    let tilePresent = slots[1] != Self.noTile
    let sourcePresent = slots[2] != Self.noCoordinate
    let destinationPresent = slots[3] != Self.noCoordinate
    return PaishoPolicyComponentsV1(
      familyIndex: Int32(slots[0]),
      tileIndex: tilePresent ? Int32(slots[1]) : 0,
      tilePresence: tilePresent ? 1 : 0,
      destinationIndex: destinationPresent ? Int32(slots[3]) : 0,
      destinationPresence: destinationPresent ? 1 : 0,
      pairIndex: sourcePresent && destinationPresent
        ? Int32(slots[2]) * Int32(PaishoTensorSchemaV1.boardCells) + Int32(slots[3])
        : 0,
      pairPresence: sourcePresent && destinationPresent ? 1 : 0
    )
  }

  private static func hasValidShape(_ slots: [UInt16]) -> Bool {
    let family = slots[0]
    let tile = slots[1]
    let source = slots[2]
    let destination = slots[3]
    let hasTile = tile != noTile
    let hasSource = source != noCoordinate
    let hasDestination = destination != noCoordinate

    switch family {
    case 0, 6:
      return tile < 6 && !hasSource && hasDestination && isGate(destination)
    case 1:
      return !hasTile && hasSource && hasDestination && source != destination
    case 2:
      return !hasTile && !hasSource && !hasDestination
    case 3:
      return (8..<12).contains(tile) && !hasSource && hasDestination && !isGate(destination)
    case 4:
      return tile == 11 && hasSource && hasDestination && source != destination
    case 5:
      return (6..<8).contains(tile) && !hasSource && hasDestination && isGate(destination)
    default:
      return false
    }
  }

  private static func isPlayable(_ slot: UInt16) -> Bool {
    let row = Int(slot) / PaishoTensorSchemaV1.boardSize
    let column = Int(slot) % PaishoTensorSchemaV1.boardSize
    let x = column - 8
    let y = 8 - row
    return abs(x) + abs(y) <= 12
  }

  private static func isGate(_ slot: UInt16) -> Bool {
    let row = Int(slot) / PaishoTensorSchemaV1.boardSize
    let column = Int(slot) % PaishoTensorSchemaV1.boardSize
    let x = column - 8
    let y = 8 - row
    return (x == 0 && abs(y) == 8) || (y == 0 && abs(x) == 8)
  }
}

struct PaishoPolicyComponentsV1 {
  let familyIndex: Int32
  let tileIndex: Int32
  let tilePresence: Float
  let destinationIndex: Int32
  let destinationPresence: Float
  let pairIndex: Int32
  let pairPresence: Float
}

public enum PaishoActionAddressError: Error, Equatable, CustomStringConvertible {
  case wrongSlotCount(Int)
  case invalidFamily(UInt16)
  case invalidTile(UInt16)
  case invalidCoordinate(UInt16)
  case nonPlayableCoordinate(UInt16)
  case invalidShape(family: UInt16)

  public var description: String {
    switch self {
    case .wrongSlotCount(let count): "action address has \(count) slots; expected 4"
    case .invalidFamily(let value): "invalid V1 action family \(value)"
    case .invalidTile(let value): "invalid V1 tile slot \(value)"
    case .invalidCoordinate(let value): "invalid V1 coordinate slot \(value)"
    case .nonPlayableCoordinate(let value): "V1 coordinate slot \(value) is not playable"
    case .invalidShape(let family): "invalid V1 component shape for family \(family)"
    }
  }
}
