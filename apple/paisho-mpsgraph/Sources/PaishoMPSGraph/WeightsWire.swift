import CryptoKit
import Foundation

public struct PaishoWeightArray: Sendable {
  public let name: String
  public let shape: [Int]
  public let values: [Float]
}

public struct PaishoWeightSnapshot: Sendable {
  public let configuration: PaishoNetworkConfiguration
  public let trainingStep: UInt64
  public let parameters: [PaishoWeightArray]

  public func encode() throws -> Data {
    var writer = WeightWriter()
    writer.data.append(Data("PWGT0001".utf8))
    for dimension in [
      configuration.trunkChannels, configuration.residualBlocks,
      configuration.policyEmbeddingChannels, configuration.valueHiddenChannels,
    ] {
      guard let value = UInt32(exactly: dimension) else { throw WeightsWireError.invalidPacket }
      writer.u32(value)
    }
    writer.u32(configuration.normalizationEpsilon.bitPattern)
    writer.u64(trainingStep)
    writer.u32(UInt32(parameters.count))
    for parameter in parameters {
      let name = Data(parameter.name.utf8)
      writer.u32(UInt32(name.count))
      writer.data.append(name)
      writer.u32(UInt32(parameter.shape.count))
      for dimension in parameter.shape { writer.u64(UInt64(dimension)) }
      writer.u64(UInt64(parameter.values.count))
      parameter.values.withUnsafeBytes { writer.data.append(contentsOf: $0) }
    }
    writer.data.append(contentsOf: SHA256.hash(data: writer.data))
    guard writer.data.count <= PaishoWeightsWire.maximumPayloadSize - 12 else {
      throw WeightsWireError.invalidPacket
    }
    return writer.data
  }

  public static func decode(_ packet: Data) throws -> Self {
    guard packet.count >= 72, packet.count <= PaishoWeightsWire.maximumPayloadSize - 12,
      Data(SHA256.hash(data: packet.dropLast(32))) == packet.suffix(32)
    else { throw WeightsWireError.invalidDigest }
    var reader = WeightReader(data: Data(packet.dropLast(32)))
    guard try reader.bytes(8) == Data("PWGT0001".utf8) else { throw WeightsWireError.invalidPacket }
    let configuration = try PaishoNetworkConfiguration(
      trunkChannels: Int(reader.u32()), residualBlocks: Int(reader.u32()),
      policyEmbeddingChannels: Int(reader.u32()), valueHiddenChannels: Int(reader.u32()),
      normalizationEpsilon: Float(bitPattern: reader.u32()))
    let step = try reader.u64()
    let count = Int(try reader.u32())
    guard count > 0, count <= reader.remaining / 24 else { throw WeightsWireError.invalidPacket }
    var parameters: [PaishoWeightArray] = []
    var names = Set<String>()
    var total = 0
    for _ in 0..<count {
      let nameBytes = try reader.bytes(Int(reader.u32()))
      guard let name = String(data: nameBytes, encoding: .utf8), !name.isEmpty,
        names.insert(name).inserted
      else { throw WeightsWireError.invalidPacket }
      let rank = Int(try reader.u32())
      guard rank > 0, rank <= reader.remaining / 8 else { throw WeightsWireError.invalidPacket }
      var shape: [Int] = []
      var elements = 1
      for _ in 0..<rank {
        guard let dimension = Int(exactly: try reader.u64()), dimension > 0 else {
          throw WeightsWireError.invalidPacket
        }
        let (product, overflow) = elements.multipliedReportingOverflow(by: dimension)
        guard !overflow else { throw WeightsWireError.invalidPacket }
        elements = product
        shape.append(dimension)
      }
      guard try reader.u64() == UInt64(elements), elements <= reader.remaining / 4 else {
        throw WeightsWireError.invalidPacket
      }
      let bytes = try reader.bytes(elements * 4)
      var values = [Float](repeating: 0, count: elements)
      _ = values.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
      guard values.allSatisfy(\.isFinite) else { throw WeightsWireError.invalidPacket }
      total += elements
      parameters.append(PaishoWeightArray(name: name, shape: shape, values: values))
    }
    guard reader.remaining == 0, total == configuration.parameterCount else {
      throw WeightsWireError.invalidPacket
    }
    return Self(configuration: configuration, trainingStep: step, parameters: parameters)
  }
}

public enum WeightsWireError: Error {
  case invalidPacket, invalidDigest, configurationMismatch, parameterMismatch
}

public enum PaishoWeightsWire {
  public static let maximumPayloadSize = 256 * 1024 * 1024
  public static func isRequest(_ data: Data) -> Bool {
    data.starts(with: Data("PSW1".utf8)) || data.starts(with: Data("PSW2".utf8))
  }
  public static func requestID(_ data: Data) -> UInt64 {
    guard data.count >= 12 else { return 0 }
    var reader = WeightReader(data: Data(data.dropFirst(4)))
    return (try? reader.u64()) ?? 0
  }
  public static func decodeRequest(_ data: Data) throws -> (
    id: UInt64, snapshot: PaishoWeightSnapshot?, packet: Data?
  ) {
    guard isRequest(data), data.count >= 12, data.count <= maximumPayloadSize else {
      throw WeightsWireError.invalidPacket
    }
    if data.starts(with: Data("PSW1".utf8)) {
      guard data.count == 12 else { throw WeightsWireError.invalidPacket }
      return (requestID(data), nil, nil)
    }
    let packet = Data(data.dropFirst(12))
    return (requestID(data), try PaishoWeightSnapshot.decode(packet), packet)
  }
  public static func exportResponse(id: UInt64, packet: Data) -> Data {
    var writer = WeightWriter()
    writer.data.append(Data("PSWR".utf8))
    writer.u64(id)
    writer.data.append(packet)
    return writer.data
  }
  public static func importResponse(id: UInt64, step: UInt64, packet: Data) -> Data {
    var writer = WeightWriter()
    writer.data.append(Data("PSWA".utf8))
    writer.u64(id)
    writer.u64(step)
    writer.data.append(packet.suffix(32))
    return writer.data
  }
  public static func encodeError(id: UInt64, error: Error) -> Data {
    let message = Data(String(describing: error).prefix(16_384).utf8)
    var writer = WeightWriter()
    writer.data.append(Data("PSWE".utf8))
    writer.u64(id)
    writer.u32(UInt32(message.count))
    writer.data.append(message)
    return writer.data
  }
}

private struct WeightWriter {
  var data = Data()
  mutating func u32(_ value: UInt32) {
    var v = value.littleEndian
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
  }
  mutating func u64(_ value: UInt64) {
    var v = value.littleEndian
    withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
  }
}

private struct WeightReader {
  let data: Data
  var offset = 0
  var remaining: Int { data.count - offset }
  mutating func bytes(_ count: Int) throws -> Data {
    guard count >= 0, count <= remaining else { throw WeightsWireError.invalidPacket }
    defer { offset += count }
    return data.subdata(in: offset..<(offset + count))
  }
  mutating func u32() throws -> UInt32 {
    try bytes(4).enumerated().reduce(0) { $0 | UInt32($1.element) << ($1.offset * 8) }
  }
  mutating func u64() throws -> UInt64 {
    try bytes(8).enumerated().reduce(0) { $0 | UInt64($1.element) << ($1.offset * 8) }
  }
}
