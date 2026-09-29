import Foundation
import MetalPerformanceShadersGraph
import XCTest

@testable import PaishoMPSGraph

final class GraphSmokeTests: XCTestCase {
  func testTrainingObjectivesAreScalarTensors() throws {
    let graph = MPSGraph()
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let artifacts = try PaishoGraphBuilder(
      graph: graph,
      configuration: .microV1,
      executionShape: shape,
      seed: 1,
      restored: []
    ).build()

    for (name, tensor) in [
      ("policy loss", artifacts.outputs.policyLoss),
      ("value loss", artifacts.outputs.valueLoss),
      ("total loss", artifacts.outputs.totalLoss),
      ("terminal policy loss", artifacts.outputs.terminalPolicyLoss),
      ("policy entropy", artifacts.outputs.policyEntropy),
      ("terminal total loss", artifacts.outputs.terminalTotalLoss),
      ("mean advantage", artifacts.outputs.meanAdvantage),
    ] {
      XCTAssertEqual(tensor.shape?.map(\.intValue).reduce(1, *), 1, name)
    }
  }

  func testMicroGraphInferenceMasksPaddingAndNormalizesOutputs() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 7)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 99
    )
    let result = try model.inference(batch)

    XCTAssertEqual(result.policyProbabilities.count, 32)
    XCTAssertEqual(result.valueProbabilities.count, 6)
    XCTAssertTrue(result.legalLogits.allSatisfy(\.isFinite))
    XCTAssertTrue(result.valueLogits.allSatisfy(\.isFinite))
    for row in 0..<shape.batchSize {
      let actionRange = row * shape.legalActionCapacity..<(row + 1) * shape.legalActionCapacity
      XCTAssertEqual(
        result.policyProbabilities[actionRange].reduce(0, +),
        1,
        accuracy: 1.0e-5
      )
      for index in actionRange where batch.inference.legalMask[index] == 0 {
        XCTAssertEqual(result.policyProbabilities[index], 0, accuracy: 1.0e-7)
      }
      let valueStart = row * PaishoTensorSchemaV1.valueClasses
      let valueRange = valueStart..<(valueStart + PaishoTensorSchemaV1.valueClasses)
      XCTAssertEqual(
        result.valueProbabilities[valueRange].reduce(0, +),
        1,
        accuracy: 1.0e-5
      )
    }
  }

  func testCompiledServingInferenceMatchesFullInference() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 70)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 71
    )

    let full = try model.inference(batch)
    let first = try model.servingInference(batch.inference)
    let second = try model.servingInference(batch.inference)

    assertArraysEqual(full.policyProbabilities, first.policyProbabilities, accuracy: 1.0e-6)
    assertArraysEqual(full.valueProbabilities, first.valueProbabilities, accuracy: 1.0e-6)
    XCTAssertEqual(first.policyProbabilities, second.policyProbabilities)
    XCTAssertEqual(first.valueProbabilities, second.valueProbabilities)
  }

  func testTwoSlotServingPipelineMatchesSerialInference() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let batches = try (80..<84).map {
      try PaishoTrainingBatch.synthetic(shape: shape, seed: UInt64($0)).inference
    }
    let serialModel = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 85
    )
    let expected = try batches.map { try serialModel.servingInference($0) }
    let pipelineModel = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 85
    )
    let pipeline = try pipelineModel.makeServingInferencePipeline(slotCount: 2)
    let group = DispatchGroup()
    let collected = InferenceResultCollector(count: batches.count)

    for (index, batch) in batches.enumerated() {
      group.enter()
      try pipeline.submit(batch) { result in
        collected.store(result, at: index)
        group.leave()
      }
    }

    XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
    pipeline.waitUntilIdle()
    let observed = try collected.values()
    for (expected, observed) in zip(expected, observed) {
      assertArraysEqual(
        expected.policyProbabilities,
        observed.policyProbabilities,
        accuracy: 1.0e-6
      )
      assertArraysEqual(
        expected.valueProbabilities,
        observed.valueProbabilities,
        accuracy: 1.0e-6
      )
    }
  }

  func testServingInferenceIsRecompiledAfterTraining() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 72)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 73
    )

    let before = try model.servingInference(batch.inference)
    _ = try model.train(batch, learningRate: 1.0e-3)
    let fullAfter = try model.inference(batch)
    let servingAfter = try model.servingInference(batch.inference)

    XCTAssertNotEqual(before.policyProbabilities, servingAfter.policyProbabilities)
    assertArraysEqual(
      fullAfter.policyProbabilities,
      servingAfter.policyProbabilities,
      accuracy: 1.0e-6
    )
    assertArraysEqual(
      fullAfter.valueProbabilities,
      servingAfter.valueProbabilities,
      accuracy: 1.0e-6
    )
  }

  func testFreshPolicyStartsNearUniformOverLegalChoices() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 8)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 100
    )
    let probabilities = try model.inference(batch).policyProbabilities

    for row in 0..<shape.batchSize {
      let start = row * shape.legalActionCapacity
      let legal = (0..<shape.legalActionCapacity).compactMap { column -> Float? in
        let index = start + column
        return batch.inference.legalMask[index] == 1 ? probabilities[index] : nil
      }
      guard legal.count > 1 else { continue }
      let entropy = -legal.reduce(0.0) { partial, probability in
        partial + Double(probability) * log(Double(probability))
      }
      let normalizedEntropy = entropy / log(Double(legal.count))
      XCTAssertGreaterThan(normalizedEntropy, 0.995)
    }
  }

  func testFreshValueStartsNearUniform() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 9)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 101
    )
    let probabilities = try model.inference(batch).valueProbabilities

    for row in 0..<shape.batchSize {
      let start = row * PaishoTensorSchemaV1.valueClasses
      let distribution = probabilities[start..<(start + PaishoTensorSchemaV1.valueClasses)]
      XCTAssertLessThan(
        Double(distribution.max()! - distribution.min()!),
        0.02
      )
    }
  }

  func testLevel1AdamStepIsFiniteAndUpdatesParameters() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 11)
    let model = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 13
    )
    let before = try model.snapshotParameters()
    let trained = try model.train(batch, learningRate: 1.0e-4)
    let after = try model.snapshotParameters()

    XCTAssertEqual(trained.step, 1)
    XCTAssertTrue(trained.policyLoss.isFinite)
    XCTAssertTrue(trained.valueLoss.isFinite)
    XCTAssertEqual(trained.totalLoss, trained.policyLoss + trained.valueLoss, accuracy: 1.0e-5)
    XCTAssertEqual(before.map(\.name), after.map(\.name))
    XCTAssertTrue(
      zip(before, after).contains { old, new in old.values != new.values },
      "Adam did not modify any trainable value"
    )
    XCTAssertTrue(after.flatMap(\.values).allSatisfy(\.isFinite))
    XCTAssertTrue(after.flatMap(\.momentum).allSatisfy(\.isFinite))
    XCTAssertTrue(after.flatMap(\.velocity).allSatisfy(\.isFinite))
  }

  func testOptimizationLevelsAgreeFromIdenticalState() throws {
    let shape = try PaishoExecutionShape(batchSize: 2, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 19)
    let level0 = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level0,
      seed: 23
    )
    let level1 = try PaishoMPSGraphModel(
      configuration: .microV1,
      executionShape: shape,
      optimization: .level1,
      seed: 23
    )
    let inference0 = try level0.inference(batch)
    let inference1 = try level1.inference(batch)
    let training0 = try level0.train(batch, learningRate: 1.0e-4)
    let training1 = try level1.train(batch, learningRate: 1.0e-4)

    assertArraysEqual(inference0.legalLogits, inference1.legalLogits, accuracy: 1.0e-5)
    assertArraysEqual(
      inference0.valueProbabilities,
      inference1.valueProbabilities,
      accuracy: 1.0e-5
    )
    XCTAssertEqual(training0.totalLoss, training1.totalLoss, accuracy: 1.0e-5)
  }

  func testPureTargetGraphBuildsAndCompletesAdamStep() throws {
    let shape = try PaishoExecutionShape(batchSize: 1, legalActionCapacity: 16)
    let batch = try PaishoTrainingBatch.synthetic(shape: shape, seed: 29)
    let model = try PaishoMPSGraphModel(
      configuration: .pureV1,
      executionShape: shape,
      optimization: .level1,
      seed: 31
    )
    let result = try model.train(batch, learningRate: 1.0e-4)

    XCTAssertEqual(model.configuration.parameterCount, 4_698_679)
    XCTAssertTrue(result.totalLoss.isFinite)
  }

  private func assertArraysEqual(
    _ left: [Float],
    _ right: [Float],
    accuracy: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(left.count, right.count, file: file, line: line)
    guard left.count == right.count else { return }
    let maximumError = zip(left, right).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(maximumError, accuracy, file: file, line: line)
  }
}

private final class InferenceResultCollector: @unchecked Sendable {
  private let lock = NSLock()
  private var results: [Result<PaishoServingInferenceResult, any Error>?]

  init(count: Int) {
    results = Array(repeating: nil, count: count)
  }

  func store(
    _ result: Result<PaishoServingInferenceResult, any Error>,
    at index: Int
  ) {
    lock.lock()
    results[index] = result
    lock.unlock()
  }

  func values() throws -> [PaishoServingInferenceResult] {
    lock.lock()
    let snapshot = results
    lock.unlock()
    return try snapshot.enumerated().map { index, result in
      guard let result else {
        throw InferenceResultCollectorError.missing(index)
      }
      return try result.get()
    }
  }
}

private enum InferenceResultCollectorError: Error {
  case missing(Int)
}
