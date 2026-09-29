import Foundation
import PaishoMPSGraph

enum Preset: String {
  case micro
  case pure

  var configuration: PaishoNetworkConfiguration {
    switch self {
    case .micro: .microV1
    case .pure: .pureV1
    }
  }
}

enum BenchmarkMode: String {
  case inference
  case servingInference = "serving-inference"
  case training
  case terminalPpo = "terminal-ppo"
  case terminalLoss = "terminal-loss"
  case terminalGradients = "terminal-gradients"
  case both
}

struct Options {
  var preset = Preset.micro
  var batchSize = 1
  var legalActions = 64
  var optimizations = [PaishoGraphOptimization.level1]
  var mode = BenchmarkMode.both
  var warmup = 2
  var iterations = 5
  var seed: UInt64 = 1
  var compiledPpo = false
  var profilePpo = false

  static func parse(_ arguments: [String]) throws -> Self {
    var options = Self()
    var index = 0
    while index < arguments.count {
      let flag = arguments[index]
      if flag == "--help" || flag == "-h" {
        printUsage()
        exit(0)
      }
      guard index + 1 < arguments.count else {
        throw ArgumentError.missingValue(flag)
      }
      let value = arguments[index + 1]
      switch flag {
      case "--profile-ppo":
        guard value == "true" || value == "false" else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.profilePpo = value == "true"
      case "--compiled-ppo":
        guard value == "true" || value == "false" else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.compiledPpo = value == "true"
      case "--preset":
        guard let parsed = Preset(rawValue: value) else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.preset = parsed
      case "--batch":
        options.batchSize = try positiveInt(value, flag: flag)
      case "--actions":
        options.legalActions = try positiveInt(value, flag: flag)
      case "--level":
        switch value {
        case "0": options.optimizations = [.level0]
        case "1": options.optimizations = [.level1]
        case "both": options.optimizations = [.level0, .level1]
        default: throw ArgumentError.invalidValue(flag, value)
        }
      case "--mode":
        guard let parsed = BenchmarkMode(rawValue: value) else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.mode = parsed
      case "--warmup":
        options.warmup = try nonNegativeInt(value, flag: flag)
      case "--iterations":
        options.iterations = try positiveInt(value, flag: flag)
      case "--seed":
        guard let parsed = UInt64(value) else {
          throw ArgumentError.invalidValue(flag, value)
        }
        options.seed = parsed
      default:
        throw ArgumentError.unknownFlag(flag)
      }
      index += 2
    }
    return options
  }
}

enum ArgumentError: Error, CustomStringConvertible {
  case missingValue(String)
  case invalidValue(String, String)
  case unknownFlag(String)

  var description: String {
    switch self {
    case .missingValue(let flag): "missing value for \(flag)"
    case .invalidValue(let flag, let value): "invalid value \(value) for \(flag)"
    case .unknownFlag(let flag): "unknown option \(flag)"
    }
  }
}

func positiveInt(_ value: String, flag: String) throws -> Int {
  guard let parsed = Int(value), parsed > 0 else {
    throw ArgumentError.invalidValue(flag, value)
  }
  return parsed
}

func nonNegativeInt(_ value: String, flag: String) throws -> Int {
  guard let parsed = Int(value), parsed >= 0 else {
    throw ArgumentError.invalidValue(flag, value)
  }
  return parsed
}

func printUsage() {
  print(
    """
    usage: paisho-mpsgraph-bench [options]
      --preset micro|pure
      --batch N
      --actions N
      --level 0|1|both
      --mode inference|serving-inference|training|terminal-ppo|terminal-loss|terminal-gradients|both
      --warmup N
      --iterations N
      --seed N
      --compiled-ppo true|false
      --profile-ppo true|false
    """
  )
}

func elapsedMilliseconds(_ body: () throws -> Void) rethrows -> Double {
  let start = DispatchTime.now().uptimeNanoseconds
  try body()
  let elapsed = DispatchTime.now().uptimeNanoseconds - start
  return Double(elapsed) / 1_000_000
}

func percentile(_ values: [Double], fraction: Double) -> Double {
  let sorted = values.sorted()
  let index = min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)
  return sorted[max(0, index)]
}

func printMeasurements(
  mode: String,
  values: [Double],
  batchSize: Int
) {
  let median = percentile(values, fraction: 0.5)
  let p95 = percentile(values, fraction: 0.95)
  let examplesPerSecond = Double(batchSize) * 1_000 / median
  print(
    "mode=\(mode) median_ms=\(String(format: "%.3f", median)) "
      + "p95_ms=\(String(format: "%.3f", p95)) "
      + "examples_per_second=\(String(format: "%.1f", examplesPerSecond))"
  )
}

func makeTerminalPpoBatch(_ source: PaishoTrainingBatch) throws -> PaishoTerminalPpoBatch {
  let shape = source.inference.shape
  let playedActionIndices = [Int](repeating: 0, count: shape.batchSize)
  let behaviorProbabilities = (0..<shape.batchSize).map { row in
    let start = row * shape.legalActionCapacity
    let end = start + shape.legalActionCapacity
    let legalCount = source.inference.legalMask[start..<end].reduce(0, +)
    return 1 / legalCount
  }
  let terminalValues = (0..<shape.batchSize).map {
    $0.isMultiple(of: 2) ? PaishoValueClassV1.win : .loss
  }
  return try PaishoTerminalPpoBatch(
    inference: source.inference,
    playedActionIndices: playedActionIndices,
    behaviorProbabilities: behaviorProbabilities,
    terminalValues: terminalValues,
    actorValues: [Float](repeating: 0, count: shape.batchSize)
  )
}

do {
  let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
  let shape = try PaishoExecutionShape(
    batchSize: options.batchSize,
    legalActionCapacity: options.legalActions
  )
  let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: options.seed)
  let terminalPpoBatch = try makeTerminalPpoBatch(batch)
  let terminalPpoParameters = try PaishoTerminalPpoParametersV1(
    policyTemperature: 1,
    uniformMix: 0.05,
    clipEpsilon: 0.2,
    valueLossWeight: 0.5,
    entropyWeight: 0.01
  )

  for optimization in options.optimizations {
    var model: PaishoMPSGraphModel!
    let constructionMs = try elapsedMilliseconds {
      model = try PaishoMPSGraphModel(
        configuration: options.preset.configuration,
        executionShape: shape,
        optimization: optimization,
        seed: options.seed
      )
    }
    model.useCompiledTerminalPpo = options.compiledPpo
    model.profileTerminalPpo = options.profilePpo
    print(
      "compiled_ppo=\(options.compiledPpo) preset=\(options.preset.rawValue) level=\(optimization.rawValue) "
        + "device=\(model.metalDeviceName) parameters=\(model.configuration.parameterCount) "
        + "batch=\(shape.batchSize) actions=\(shape.legalActionCapacity) "
        + "graph_build_ms=\(String(format: "%.3f", constructionMs))"
    )

    if options.mode == .inference || options.mode == .both {
      for _ in 0..<options.warmup {
        _ = try model.inference(batch)
      }
      let times = try (0..<options.iterations).map { _ in
        try elapsedMilliseconds { _ = try model.inference(batch) }
      }
      printMeasurements(mode: "inference", values: times, batchSize: shape.batchSize)
    }

    if options.mode == .servingInference {
      for _ in 0..<options.warmup {
        _ = try model.servingInference(batch.inference)
      }
      let times = try (0..<options.iterations).map { _ in
        try elapsedMilliseconds { _ = try model.servingInference(batch.inference) }
      }
      printMeasurements(
        mode: "serving_inference",
        values: times,
        batchSize: shape.batchSize
      )
    }

    if options.mode == .training || options.mode == .both {
      for _ in 0..<options.warmup {
        _ = try model.train(batch, learningRate: 1.0e-4)
      }
      var latestLoss: Float = .nan
      let times = try (0..<options.iterations).map { _ in
        try elapsedMilliseconds {
          latestLoss = try model.train(batch, learningRate: 1.0e-4).totalLoss
        }
      }
      printMeasurements(mode: "training", values: times, batchSize: shape.batchSize)
      print("training_step=\(model.trainingStep) total_loss=\(latestLoss)")
    }

    if [.terminalPpo, .terminalLoss, .terminalGradients].contains(options.mode) {
      var components: [PaishoTerminalPpoTiming] = []
      func runTerminal() throws -> PaishoTerminalPpoResult {
        if options.mode != .terminalPpo {
          return try model.probeTerminalPpo(terminalPpoBatch, parameters: terminalPpoParameters,
            probe: options.mode == .terminalLoss ? .loss : .gradients)
        }
        return try model.trainTerminalPpo(terminalPpoBatch, learningRate: 1e-4,
                                         parameters: terminalPpoParameters)
      }
      for _ in 0..<options.warmup {
        _ = try runTerminal()
      }
      var latestLoss: Float = .nan
      let times = try (0..<options.iterations).map { _ in
        try elapsedMilliseconds {
          latestLoss = try runTerminal().totalLoss
          if let timing = model.lastTerminalPpoTiming { components.append(timing) }
        }
      }
      printMeasurements(mode: options.mode.rawValue, values: times, batchSize: shape.batchSize)
      print("training_step=\(model.trainingStep) total_loss=\(latestLoss)")
      if !components.isEmpty {
        for (name, key) in [
          ("validation", \PaishoTerminalPpoTiming.validationSeconds),
          ("feeds", \PaishoTerminalPpoTiming.feedsSeconds),
          ("execution_wall", \PaishoTerminalPpoTiming.executionSeconds),
          ("readback", \PaishoTerminalPpoTiming.readbackSeconds),
        ] {
          let average = components.reduce(0) { $0 + $1[keyPath: key] } / Double(components.count)
          print("ppo_component=\(name) mean_ms=\(average * 1000) samples=\(components.count)")
        }
      }
    }
  }
} catch {
  FileHandle.standardError.write(Data("error: \(error)\n".utf8))
  printUsage()
  exit(2)
}
