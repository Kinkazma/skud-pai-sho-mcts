import CoreML
import Foundation

// Trace driver only. Run prediction only in a parent-coordinated campaign pause.
@main
struct PredictFixture {
  struct Fixture: Decodable {
    struct Batch: Decodable { let inputs: [String: [Double]] }
    let batch: Int
    let capacity: Int
    let batches: [Batch]
  }
  static func require(_ valid: Bool, _ message: String) throws {
    if !valid { throw NSError(domain: "CoreMLStudy", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: message]) }
  }
  static func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    try require((3...4).contains(args.count),
                "Usage: predict-fixture MODEL.mlmodelc FIXTURE.json cpu-ane|all [REPEATS=100]")
    let repeats = args.count == 4 ? Int(args[3]) ?? 0 : 100
    try require(repeats > 0, "REPEATS must be positive")
    try require(["cpu-ane", "all"].contains(args[2]), "Expected cpu-ane or all")
    let url = URL(fileURLWithPath: args[0])
    try require(url.pathExtension == "mlmodelc", "Expected compiled .mlmodelc")
    let fixture = try JSONDecoder().decode(Fixture.self,
      from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    try require(fixture.batch > 0 && fixture.capacity > 0 && !fixture.batches.isEmpty,
                "Expected positive batch/capacity and nonempty batches")
    let indices = ["family_indices": 7, "tile_indices": 12,
                   "destination_indices": 289, "pair_indices": 83521]
    var shapes = ["spatial": [fixture.batch, 17, 17, 29],
                  "global_features": [fixture.batch, 26]]
    for name in Array(indices.keys) + ["tile_presence", "destination_presence", "pair_presence", "legal_mask"] {
      shapes[name] = [fixture.batch, fixture.capacity]
    }
    let providers = try fixture.batches.map { batch -> MLDictionaryFeatureProvider in
      try require(Set(batch.inputs.keys) == Set(shapes.keys), "Incorrect input names")
      var features: [String: Any] = [:]
      for (name, shape) in shapes {
        let values = batch.inputs[name]!
        let count = try shape.reduce(1) { result, dimension in
          let product = result.multipliedReportingOverflow(by: dimension)
          try require(!product.overflow, "Shape overflow")
          return product.partialValue
        }
        try require(values.count == count, "Incorrect count for \(name)")
        let tensor = try MLMultiArray(shape: shape.map { NSNumber(value: $0) },
                                      dataType: indices[name] == nil ? .float32 : .int32)
        for (i, value) in values.enumerated() {
          try require(value.isFinite && Float(value).isFinite, "Nonfinite input: \(name)")
          if let limit = indices[name] {
            try require(value >= 0 && value < Double(limit) && value.rounded() == value,
                        "Invalid index: \(name)")
          } else if name.hasSuffix("presence") || name == "legal_mask" {
            try require(value == 0 || value == 1, "Expected binary mask: \(name)")
          }
          tensor[i] = NSNumber(value: value)
        }
        features[name] = tensor
      }
      let mask = batch.inputs["legal_mask"]!
      for row in 0..<fixture.batch {
        try require(mask[(row * fixture.capacity)..<((row + 1) * fixture.capacity)].contains(1),
                    "Each row must have a legal action")
      }
      return try MLDictionaryFeatureProvider(dictionary: features)
    }
    let config = MLModelConfiguration()
    config.computeUnits = args[2] == "cpu-ane" ? .cpuAndNeuralEngine : .all
    let model = try MLModel(contentsOf: url, configuration: config)
    func sweep() throws -> Double {
      var checksum = 0.0
      for provider in providers {
        try autoreleasepool {
          let result = try model.prediction(from: provider)
          for (name, width) in [("policy_probabilities", fixture.capacity), ("value_probabilities", 3)] {
            guard let output = result.featureValue(for: name)?.multiArrayValue else {
              try require(false, "Missing output: \(name)"); return
            }
            try require(output.count == fixture.batch * width, "Incorrect output count")
            // A weighted checksum consumes outputs; it is not a parity comparison.
            for i in 0..<output.count {
              let value = output[i].doubleValue
              try require(value.isFinite, "Nonfinite prediction: \(name)")
              checksum += value * Double(i % 97 + 1)
            }
          }
        }
      }
      return checksum
    }
    for _ in 0..<2 { _ = try sweep() }
    let start = ProcessInfo.processInfo.systemUptime
    var checksum = 0.0
    for _ in 0..<repeats { checksum += try sweep() }
    let seconds = ProcessInfo.processInfo.systemUptime - start
    print("compute_units=\(args[2]) batches=\(providers.count) warmup_calls=\(2 * providers.count) ncalls=\(repeats * providers.count) seconds=\(seconds) checksum=\(checksum)")
    print("Scope: synchronous predictions + output checksum; excludes load/warmup. ANE execution requires trace inspection.")
  }
}
