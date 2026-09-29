import Foundation
import MetalPerformanceShadersGraph

func data(from values: [Float]) -> Data {
  values.withUnsafeBytes { Data($0) }
}

func data(from values: [Int32]) -> Data {
  values.withUnsafeBytes { Data($0) }
}

func tensorData(
  _ values: [Float],
  shape: [Int],
  device: MPSGraphDevice
) -> MPSGraphTensorData {
  MPSGraphTensorData(
    device: device,
    data: data(from: values),
    shape: shape.map(NSNumber.init(value:)),
    dataType: .float32
  )
}

func tensorData(
  _ values: [Int32],
  shape: [Int],
  device: MPSGraphDevice
) -> MPSGraphTensorData {
  MPSGraphTensorData(
    device: device,
    data: data(from: values),
    shape: shape.map(NSNumber.init(value:)),
    dataType: .int32
  )
}

enum PaishoTensorReadError: Error, CustomStringConvertible {
  case unexpectedDataType(String)
  case invalidShape([Int])
  case unexpectedElementCount(expected: Int, actual: Int, shape: [Int])

  var description: String {
    switch self {
    case .unexpectedDataType(let actual):
      "expected float32 tensor data, found \(actual)"
    case .invalidShape(let shape):
      "tensor data has an invalid shape: \(shape)"
    case .unexpectedElementCount(let expected, let actual, let shape):
      "expected \(expected) tensor elements, found \(actual) for shape \(shape)"
    }
  }
}

func readFloats(_ tensorData: MPSGraphTensorData, count: Int) throws -> [Float] {
  guard tensorData.dataType == .float32 else {
    throw PaishoTensorReadError.unexpectedDataType(String(describing: tensorData.dataType))
  }
  let shape = tensorData.shape.map(\.intValue)
  var actualCount = 1
  for dimension in shape {
    let multiplication = actualCount.multipliedReportingOverflow(by: dimension)
    guard dimension >= 0, !multiplication.overflow else {
      throw PaishoTensorReadError.invalidShape(shape)
    }
    actualCount = multiplication.partialValue
  }
  guard actualCount == count else {
    throw PaishoTensorReadError.unexpectedElementCount(
      expected: count,
      actual: actualCount,
      shape: shape
    )
  }
  var values = [Float](repeating: 0, count: count)
  values.withUnsafeMutableBytes { buffer in
    guard let destination = buffer.baseAddress else { return }
    tensorData.mpsndarray().readBytes(destination, strideBytes: nil)
  }
  return values
}
