import CoreML
import Foundation

// Loading a compute plan can compile/specialize a model. Run only when the
// parent has coordinated a campaign pause. This tool never calls prediction.
@main
struct InspectComputePlan {
  static func main() async throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count == 2, ["cpu", "cpu-ane", "all"].contains(args[1]) else {
      print("Usage: inspect-compute-plan MODEL.mlmodelc cpu|cpu-ane|all")
      return
    }
    guard #available(macOS 14.4, *) else {
      throw NSError(domain: "CoreMLStudy", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "MLComputePlan requires macOS 14.4+"])
    }
    let url = URL(fileURLWithPath: args[0])
    guard url.pathExtension == "mlmodelc" else {
      throw NSError(domain: "CoreMLStudy", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Compile the package first with coremlcompiler"])
    }
    let configuration = MLModelConfiguration()
    switch args[1] {
    case "cpu": configuration.computeUnits = .cpuOnly
    case "cpu-ane": configuration.computeUnits = .cpuAndNeuralEngine
    default: configuration.computeUnits = .all
    }
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
    guard case .program(let program) = plan.modelStructure else {
      throw NSError(domain: "CoreMLStudy", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Expected ML Program"])
    }
    var rows: [[String: Any]] = []
    func visit(_ block: MLModelStructure.Program.Block, path: String) {
      for (index, operation) in block.operations.enumerated() {
        let location = "\(path)/\(index)"
        let usage = plan.deviceUsage(for: operation)
        rows.append([
          "path": location, "operator": operation.operatorName,
          "preferred": usage.map { String(describing: $0.preferred) } as Any? ?? NSNull(),
          "supported": usage.map { $0.supported.map { String(describing: $0) } } as Any? ?? NSNull(),
          "estimated_relative_cost": plan.estimatedCost(of: operation).map { $0.weight } as Any? ?? NSNull(),
        ])
        for (child, nested) in operation.blocks.enumerated() {
          visit(nested, path: "\(location)/block_\(child)")
        }
      }
    }
    for name in program.functions.keys.sorted() {
      visit(program.functions[name]!.block, path: name)
    }
    let report: [String: Any] = [
      "model": url.path, "compute_units": args[1],
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
      "interpretation": "Planned placement only; no prediction or observed ANE activity. Costs are estimates, not timings.",
      "operations": rows,
    ]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
  }
}
