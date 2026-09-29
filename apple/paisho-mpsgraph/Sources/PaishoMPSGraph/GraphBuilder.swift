import Foundation
import MetalPerformanceShadersGraph

struct GraphInputs {
  let spatial: MPSGraphTensor
  let global: MPSGraphTensor
  let familyIndices: MPSGraphTensor
  let tileIndices: MPSGraphTensor
  let tilePresence: MPSGraphTensor
  let destinationIndices: MPSGraphTensor
  let destinationPresence: MPSGraphTensor
  let pairIndices: MPSGraphTensor
  let pairPresence: MPSGraphTensor
  let legalMask: MPSGraphTensor
  let policyTargets: MPSGraphTensor
  let valueTargets: MPSGraphTensor
  let playedActionMask: MPSGraphTensor
  let behaviorProbabilities: MPSGraphTensor
  let advantages: MPSGraphTensor
  let policyTemperature: MPSGraphTensor
  let uniformMix: MPSGraphTensor
  let clipEpsilon: MPSGraphTensor
  let valueLossWeight: MPSGraphTensor
  let entropyWeight: MPSGraphTensor
  let learningRate: MPSGraphTensor
  let beta1Power: MPSGraphTensor
  let beta2Power: MPSGraphTensor
}

struct GraphOutputs {
  let legalLogits: MPSGraphTensor
  let policyProbabilities: MPSGraphTensor
  let valueLogits: MPSGraphTensor
  let valueProbabilities: MPSGraphTensor
  let policyLoss: MPSGraphTensor
  let valueLoss: MPSGraphTensor
  let totalLoss: MPSGraphTensor
  let terminalPolicyLoss: MPSGraphTensor
  let policyEntropy: MPSGraphTensor
  let terminalTotalLoss: MPSGraphTensor
  let meanAdvantage: MPSGraphTensor
  let meanImportanceRatio: MPSGraphTensor
  let meanSquaredRatioDeviation: MPSGraphTensor
}

struct GraphArtifacts {
  let inputs: GraphInputs
  let outputs: GraphOutputs
  let parameters: [GraphParameter]
  let updateOperations: [MPSGraphOperation]
  let terminalUpdateOperations: [MPSGraphOperation]
}

final class PaishoGraphBuilder {
  private let graph: MPSGraph
  private let configuration: PaishoNetworkConfiguration
  private let executionShape: PaishoExecutionShape
  private let store: ParameterStore
  private let convolutionDescriptor: MPSGraphConvolution2DOpDescriptor

  init(
    graph: MPSGraph,
    configuration: PaishoNetworkConfiguration,
    executionShape: PaishoExecutionShape,
    seed: UInt64,
    restored: [PaishoParameterSnapshot]
  ) throws {
    self.graph = graph
    self.configuration = configuration
    self.executionShape = executionShape
    store = try ParameterStore(graph: graph, seed: seed, restored: restored)
    convolutionDescriptor = MPSGraphConvolution2DOpDescriptor(
      strideInX: 1,
      strideInY: 1,
      dilationRateInX: 1,
      dilationRateInY: 1,
      groups: 1,
      paddingStyle: .TF_SAME,
      dataLayout: .NHWC,
      weightsLayout: .HWIO
    )!
  }

  func build() throws -> GraphArtifacts {
    let inputs = makeInputs()
    let trunk = try makeTrunk(spatial: inputs.spatial, global: inputs.global)
    let pooled = graph.reshape(
      graph.mean(of: trunk, axes: [1, 2], name: "trunk/pool"),
      shape: [executionShape.batchSize as NSNumber, configuration.trunkChannels as NSNumber],
      name: "trunk/pooled"
    )
    let outputs = try makeOutputs(trunk: trunk, pooled: pooled, inputs: inputs)
    let updateOperations = makeAdamUpdates(
      loss: outputs.totalLoss, inputs: inputs, objective: "supervised"
    )
    let terminalUpdateOperations = makeAdamUpdates(
      loss: outputs.terminalTotalLoss, inputs: inputs, objective: "terminal_ppo"
    )
    try store.validateRestorationComplete()
    let actualCount = store.parameters.reduce(0) { $0 + $1.count }
    guard actualCount == configuration.parameterCount else {
      throw PaishoMPSGraphError.parameterCountMismatch(
        expected: configuration.parameterCount,
        actual: actualCount
      )
    }
    return GraphArtifacts(
      inputs: inputs,
      outputs: outputs,
      parameters: store.parameters,
      updateOperations: updateOperations,
      terminalUpdateOperations: terminalUpdateOperations
    )
  }

  private func makeInputs() -> GraphInputs {
    let batch = executionShape.batchSize as NSNumber
    let capacity = executionShape.legalActionCapacity as NSNumber
    let actionShape = [batch, capacity]
    return GraphInputs(
      spatial: graph.placeholder(
        shape: [
          batch,
          PaishoTensorSchemaV1.boardSize as NSNumber,
          PaishoTensorSchemaV1.boardSize as NSNumber,
          PaishoTensorSchemaV1.spatialChannels as NSNumber,
        ],
        dataType: .float32,
        name: "input/spatial"
      ),
      global: graph.placeholder(
        shape: [batch, PaishoTensorSchemaV1.globalFeatures as NSNumber],
        dataType: .float32,
        name: "input/global"
      ),
      familyIndices: graph.placeholder(
        shape: actionShape,
        dataType: .int32,
        name: "input/family_indices"
      ),
      tileIndices: graph.placeholder(
        shape: actionShape,
        dataType: .int32,
        name: "input/tile_indices"
      ),
      tilePresence: graph.placeholder(
        shape: actionShape,
        dataType: .float32,
        name: "input/tile_presence"
      ),
      destinationIndices: graph.placeholder(
        shape: actionShape,
        dataType: .int32,
        name: "input/destination_indices"
      ),
      destinationPresence: graph.placeholder(
        shape: actionShape,
        dataType: .float32,
        name: "input/destination_presence"
      ),
      pairIndices: graph.placeholder(
        shape: actionShape,
        dataType: .int32,
        name: "input/pair_indices"
      ),
      pairPresence: graph.placeholder(
        shape: actionShape,
        dataType: .float32,
        name: "input/pair_presence"
      ),
      legalMask: graph.placeholder(
        shape: actionShape,
        dataType: .float32,
        name: "input/legal_mask"
      ),
      policyTargets: graph.placeholder(
        shape: actionShape,
        dataType: .float32,
        name: "target/policy"
      ),
      valueTargets: graph.placeholder(
        shape: [batch, PaishoTensorSchemaV1.valueClasses as NSNumber],
        dataType: .float32,
        name: "target/value"
      ),
      playedActionMask: graph.placeholder(
        shape: actionShape,
        dataType: .float32,
        name: "terminal_ppo/played_action_mask"
      ),
      behaviorProbabilities: graph.placeholder(
        shape: [batch],
        dataType: .float32,
        name: "terminal_ppo/behavior_probabilities"
      ),
      advantages: graph.placeholder(
        shape: [batch],
        dataType: .float32,
        name: "terminal_ppo/advantages"
      ),
      policyTemperature: graph.placeholder(
        shape: [1],
        dataType: .float32,
        name: "terminal_ppo/policy_temperature"
      ),
      uniformMix: graph.placeholder(
        shape: [1],
        dataType: .float32,
        name: "terminal_ppo/uniform_mix"
      ),
      clipEpsilon: graph.placeholder(
        shape: [1],
        dataType: .float32,
        name: "terminal_ppo/clip_epsilon"
      ),
      valueLossWeight: graph.placeholder(
        shape: [1],
        dataType: .float32,
        name: "terminal_ppo/value_loss_weight"
      ),
      entropyWeight: graph.placeholder(
        shape: [1],
        dataType: .float32,
        name: "terminal_ppo/entropy_weight"
      ),
      learningRate: graph.placeholder(shape: [1], dataType: .float32, name: "adam/lr"),
      beta1Power: graph.placeholder(
        shape: [1], dataType: .float32, name: "adam/beta1_power"
      ),
      beta2Power: graph.placeholder(
        shape: [1], dataType: .float32, name: "adam/beta2_power"
      )
    )
  }

  private func makeTrunk(
    spatial: MPSGraphTensor,
    global: MPSGraphTensor
  ) throws -> MPSGraphTensor {
    let channels = configuration.trunkChannels
    var trunk = try convolution(
      spatial,
      kernel: 3,
      inputChannels: PaishoTensorSchemaV1.spatialChannels,
      outputChannels: channels,
      name: "stem/conv"
    )
    let projectedGlobal = try dense(
      global,
      input: PaishoTensorSchemaV1.globalFeatures,
      output: channels,
      name: "stem/global"
    )
    let broadcastGlobal = graph.reshape(
      projectedGlobal,
      shape: [executionShape.batchSize as NSNumber, 1, 1, channels as NSNumber],
      name: "stem/global_broadcast"
    )
    trunk = graph.addition(trunk, broadcastGlobal, name: "stem/add_global")
    trunk = try channelNormalization(trunk, channels: channels, name: "stem/norm")
    trunk = graph.reLU(with: trunk, name: "stem/relu")

    for block in 0..<configuration.residualBlocks {
      let prefix = "trunk/block_\(block)"
      var residual = try convolution(
        trunk,
        kernel: 3,
        inputChannels: channels,
        outputChannels: channels,
        name: "\(prefix)/conv_1"
      )
      residual = try channelNormalization(
        residual,
        channels: channels,
        name: "\(prefix)/norm_1"
      )
      residual = graph.reLU(with: residual, name: "\(prefix)/relu_1")
      residual = try convolution(
        residual,
        kernel: 3,
        inputChannels: channels,
        outputChannels: channels,
        name: "\(prefix)/conv_2"
      )
      residual = try channelNormalization(
        residual,
        channels: channels,
        name: "\(prefix)/norm_2"
      )
      trunk = graph.reLU(
        with: graph.addition(trunk, residual, name: "\(prefix)/add"),
        name: "\(prefix)/relu_2"
      )
    }
    return trunk
  }

  private func makeOutputs(
    trunk: MPSGraphTensor,
    pooled: MPSGraphTensor,
    inputs: GraphInputs
  ) throws -> GraphOutputs {
    let channels = configuration.trunkChannels
    let batch = executionShape.batchSize as NSNumber
    let familyLogits = try dense(
      pooled,
      input: channels,
      output: PaishoTensorSchemaV1.actionFamilies,
      name: "policy/family",
      weightInitialization: .scaledUniform(fanIn: channels, outputScale: 0.01)
    )
    let tileLogits = try dense(
      pooled,
      input: channels,
      output: PaishoTensorSchemaV1.tileKinds,
      name: "policy/tile",
      weightInitialization: .scaledUniform(fanIn: channels, outputScale: 0.01)
    )
    let destinationLogits = graph.reshape(
      try convolution(
        trunk,
        kernel: 1,
        inputChannels: channels,
        outputChannels: 1,
        name: "policy/destination",
        weightInitialization: .scaledUniform(fanIn: channels, outputScale: 0.01)
      ),
      shape: [batch, PaishoTensorSchemaV1.boardCells as NSNumber],
      name: "policy/destination_flat"
    )
    let sourceEmbedding = graph.reshape(
      try convolution(
        trunk,
        kernel: 1,
        inputChannels: channels,
        outputChannels: configuration.policyEmbeddingChannels,
        name: "policy/source_embedding",
        weightInitialization: .scaledUniform(fanIn: channels, outputScale: 0.1)
      ),
      shape: [
        batch,
        PaishoTensorSchemaV1.boardCells as NSNumber,
        configuration.policyEmbeddingChannels as NSNumber,
      ],
      name: "policy/source_embedding_flat"
    )
    let destinationEmbedding = graph.reshape(
      try convolution(
        trunk,
        kernel: 1,
        inputChannels: channels,
        outputChannels: configuration.policyEmbeddingChannels,
        name: "policy/destination_embedding",
        weightInitialization: .scaledUniform(fanIn: channels, outputScale: 0.1)
      ),
      shape: [
        batch,
        PaishoTensorSchemaV1.boardCells as NSNumber,
        configuration.policyEmbeddingChannels as NSNumber,
      ],
      name: "policy/destination_embedding_flat"
    )
    let transposedDestination = graph.transpose(
      destinationEmbedding,
      permutation: [0, 2, 1],
      name: "policy/destination_embedding_transposed"
    )
    let pairScale = graph.constant(
      1 / sqrt(Double(configuration.policyEmbeddingChannels)),
      dataType: .float32
    )
    let pairLogits = graph.reshape(
      graph.multiplication(
        graph.matrixMultiplication(
          primary: sourceEmbedding,
          secondary: transposedDestination,
          name: "policy/pair_product"
        ),
        pairScale,
        name: "policy/pair_scaled"
      ),
      shape: [
        batch,
        (PaishoTensorSchemaV1.boardCells * PaishoTensorSchemaV1.boardCells) as NSNumber,
      ],
      name: "policy/pair_flat"
    )

    let familyScores = graph.gatherAlongAxis(
      1,
      updates: familyLogits,
      indices: inputs.familyIndices,
      name: "policy/legal_family"
    )
    let tileScores = graph.multiplication(
      graph.gatherAlongAxis(
        1,
        updates: tileLogits,
        indices: inputs.tileIndices,
        name: "policy/legal_tile_raw"
      ),
      inputs.tilePresence,
      name: "policy/legal_tile"
    )
    let destinationScores = graph.multiplication(
      graph.gatherAlongAxis(
        1,
        updates: destinationLogits,
        indices: inputs.destinationIndices,
        name: "policy/legal_destination_raw"
      ),
      inputs.destinationPresence,
      name: "policy/legal_destination"
    )
    let pairScores = graph.multiplication(
      graph.gatherAlongAxis(
        1,
        updates: pairLogits,
        indices: inputs.pairIndices,
        name: "policy/legal_pair_raw"
      ),
      inputs.pairPresence,
      name: "policy/legal_pair"
    )
    var legalLogits = graph.addition(familyScores, tileScores, name: "policy/legal_sum_1")
    legalLogits = graph.addition(legalLogits, destinationScores, name: "policy/legal_sum_2")
    legalLogits = graph.addition(legalLogits, pairScores, name: "policy/legal_sum_3")
    let invalidMask = graph.subtraction(
      graph.constant(1, dataType: .float32),
      inputs.legalMask,
      name: "policy/invalid_mask"
    )
    legalLogits = graph.addition(
      legalLogits,
      graph.multiplication(
        invalidMask,
        graph.constant(-1.0e9, dataType: .float32),
        name: "policy/padding_penalty"
      ),
      name: "policy/masked_logits"
    )

    var valueHidden = try dense(
      pooled,
      input: channels,
      output: configuration.valueHiddenChannels,
      name: "value/hidden"
    )
    valueHidden = graph.reLU(with: valueHidden, name: "value/hidden_relu")
    let valueLogits = try dense(
      valueHidden,
      input: configuration.valueHiddenChannels,
      output: PaishoTensorSchemaV1.valueClasses,
      name: "value/logits",
      weightInitialization: .scaledUniform(
        fanIn: configuration.valueHiddenChannels,
        outputScale: 0.01
      )
    )

    let policyProbabilities = graph.softMax(
      with: legalLogits,
      axis: -1,
      name: "policy/probabilities"
    )
    let policyLoss = graph.softMaxCrossEntropy(
      legalLogits,
      labels: inputs.policyTargets,
      axis: -1,
      reuctionType: .mean,
      name: "loss/policy"
    )
    let valueLoss = graph.softMaxCrossEntropy(
      valueLogits,
      labels: inputs.valueTargets,
      axis: -1,
      reuctionType: .mean,
      name: "loss/value"
    )
    let terminalObjective = makeTerminalPpoObjective(
      legalLogits: legalLogits,
      valueLoss: valueLoss,
      inputs: inputs
    )
    return GraphOutputs(
      legalLogits: legalLogits,
      policyProbabilities: policyProbabilities,
      valueLogits: valueLogits,
      valueProbabilities: graph.softMax(with: valueLogits, axis: -1, name: "value/probabilities"),
      policyLoss: policyLoss,
      valueLoss: valueLoss,
      totalLoss: graph.addition(policyLoss, valueLoss, name: "loss/total"),
      terminalPolicyLoss: terminalObjective.policyLoss,
      policyEntropy: terminalObjective.entropy,
      terminalTotalLoss: terminalObjective.totalLoss,
      meanAdvantage: terminalObjective.meanAdvantage,
      meanImportanceRatio: terminalObjective.meanImportanceRatio,
      meanSquaredRatioDeviation: terminalObjective.meanSquaredRatioDeviation
    )
  }

  private func makeTerminalPpoObjective(
    legalLogits: MPSGraphTensor,
    valueLoss: MPSGraphTensor,
    inputs: GraphInputs
  ) -> (
    policyLoss: MPSGraphTensor,
    entropy: MPSGraphTensor,
    totalLoss: MPSGraphTensor,
    meanAdvantage: MPSGraphTensor,
    meanImportanceRatio: MPSGraphTensor,
    meanSquaredRatioDeviation: MPSGraphTensor
  ) {
    let paddedTemperedProbabilities = graph.softMax(
      with: graph.division(
        legalLogits,
        inputs.policyTemperature,
        name: "terminal_ppo/temperature_scaled_logits"
      ),
      axis: -1,
      name: "terminal_ppo/padded_tempered_probabilities"
    )
    let legalTemperedProbabilities = graph.multiplication(
      paddedTemperedProbabilities,
      inputs.legalMask,
      name: "terminal_ppo/legal_tempered_probabilities"
    )
    let legalTemperedMass = graph.reductionSum(
      with: legalTemperedProbabilities,
      axis: 1,
      name: "terminal_ppo/legal_tempered_mass"
    )
    let temperedProbabilities = graph.division(
      legalTemperedProbabilities,
      graph.reshape(
        legalTemperedMass,
        shape: [executionShape.batchSize as NSNumber, 1],
        name: "terminal_ppo/legal_tempered_mass_column"
      ),
      name: "terminal_ppo/tempered_probabilities"
    )
    let legalActionCounts = graph.reductionSum(
      with: inputs.legalMask,
      axis: 1,
      name: "terminal_ppo/legal_action_counts"
    )
    let uniformProbabilities = graph.division(
      inputs.legalMask,
      graph.reshape(
        legalActionCounts,
        shape: [executionShape.batchSize as NSNumber, 1],
        name: "terminal_ppo/legal_action_counts_column"
      ),
      name: "terminal_ppo/uniform_probabilities"
    )
    let policyProbabilities = graph.addition(
      graph.multiplication(
        temperedProbabilities,
        graph.subtraction(
          graph.constant(1, dataType: .float32),
          inputs.uniformMix,
          name: "terminal_ppo/retained_policy_weight"
        ),
        name: "terminal_ppo/retained_policy"
      ),
      graph.multiplication(
        uniformProbabilities,
        inputs.uniformMix,
        name: "terminal_ppo/uniform_policy"
      ),
      name: "terminal_ppo/candidate_behavior_policy"
    )
    let playedProbabilities = graph.reshape(
      graph.reductionSum(
        with: graph.multiplication(
          policyProbabilities,
          inputs.playedActionMask,
          name: "terminal_ppo/played_probabilities_masked"
        ),
        axis: 1,
        name: "terminal_ppo/played_probabilities_reduced"
      ),
      shape: [executionShape.batchSize as NSNumber],
      name: "terminal_ppo/played_probabilities"
    )
    let ratio = graph.division(
      playedProbabilities,
      inputs.behaviorProbabilities,
      name: "terminal_ppo/ratio"
    )
    let one = graph.constant(1, dataType: .float32)
    let ratioDeviation = graph.subtraction(
      ratio,
      one,
      name: "terminal_ppo/ratio_deviation"
    )
    let meanImportanceRatio = graph.mean(
      of: ratio,
      axes: [0],
      name: "terminal_ppo/mean_importance_ratio"
    )
    let meanSquaredRatioDeviation = graph.mean(
      of: graph.multiplication(
        ratioDeviation,
        ratioDeviation,
        name: "terminal_ppo/squared_ratio_deviation"
      ),
      axes: [0],
      name: "terminal_ppo/mean_squared_ratio_deviation"
    )
    let clipMinimum = graph.subtraction(
      one,
      inputs.clipEpsilon,
      name: "terminal_ppo/clip_min"
    )
    let clipMaximum = graph.addition(
      one,
      inputs.clipEpsilon,
      name: "terminal_ppo/clip_max"
    )
    let positiveAdvantages = graph.reLU(
      with: inputs.advantages,
      name: "terminal_ppo/positive_advantages"
    )
    let negativeAdvantages = graph.negative(
      with: graph.reLU(
        with: graph.negative(
          with: inputs.advantages,
          name: "terminal_ppo/negated_advantages"
        ),
        name: "terminal_ppo/negative_advantage_magnitudes"
      ),
      name: "terminal_ppo/negative_advantages"
    )
    let positiveRatio = graph.subtraction(
      clipMaximum,
      graph.reLU(
        with: graph.subtraction(
          clipMaximum,
          ratio,
          name: "terminal_ppo/positive_ratio_distance"
        ),
        name: "terminal_ppo/positive_ratio_below_clip"
      ),
      name: "terminal_ppo/positive_clipped_ratio"
    )
    let finiteRatio = graph.minimum(
      ratio,
      graph.constant(Double(Float.greatestFiniteMagnitude) / 4, dataType: .float32),
      name: "terminal_ppo/finite_ratio"
    )
    let negativeRatio = graph.addition(
      clipMinimum,
      graph.reLU(
        with: graph.subtraction(
          finiteRatio,
          clipMinimum,
          name: "terminal_ppo/negative_ratio_distance"
        ),
        name: "terminal_ppo/negative_ratio_above_clip"
      ),
      name: "terminal_ppo/negative_clipped_ratio"
    )
    // The sign-specific form is the usual min(r*A, clip(r)*A), but avoids
    // MPSGraph's zero gradient when the two surrogate operands are equal and
    // avoids infinity-minus-infinity for a tiny stored behavior probability.
    let minimumSurrogate = graph.addition(
      graph.multiplication(
        positiveRatio,
        positiveAdvantages,
        name: "terminal_ppo/positive_surrogate"
      ),
      graph.multiplication(
        negativeRatio,
        negativeAdvantages,
        name: "terminal_ppo/negative_surrogate"
      ),
      name: "terminal_ppo/minimum_surrogate"
    )
    let policyLoss = graph.negative(
      with: graph.mean(
        of: minimumSurrogate,
        axes: [0],
        name: "terminal_ppo/mean_surrogate"
      ),
      name: "terminal_ppo/policy_loss"
    )

    let safeProbabilities = graph.maximum(
      policyProbabilities,
      graph.constant(1.0e-12, dataType: .float32),
      name: "terminal_ppo/safe_probabilities"
    )
    let entropyTerms = graph.multiplication(
      graph.multiplication(
        policyProbabilities,
        graph.logarithm(with: safeProbabilities, name: "terminal_ppo/log_probabilities"),
        name: "terminal_ppo/p_log_p"
      ),
      inputs.legalMask,
      name: "terminal_ppo/legal_p_log_p"
    )
    let entropy = graph.negative(
      with: graph.mean(
        of: graph.reductionSum(
          with: entropyTerms,
          axis: 1,
          name: "terminal_ppo/row_negative_entropy"
        ),
        axes: [0],
        name: "terminal_ppo/mean_negative_entropy"
      ),
      name: "terminal_ppo/entropy"
    )
    let weightedValueLoss = graph.multiplication(
      valueLoss,
      inputs.valueLossWeight,
      name: "terminal_ppo/weighted_value_loss"
    )
    let weightedEntropy = graph.multiplication(
      entropy,
      inputs.entropyWeight,
      name: "terminal_ppo/weighted_entropy"
    )
    let totalLoss = graph.subtraction(
      graph.addition(
        policyLoss,
        weightedValueLoss,
        name: "terminal_ppo/policy_plus_value"
      ),
      weightedEntropy,
      name: "terminal_ppo/total_loss"
    )
    return (
      policyLoss,
      entropy,
      totalLoss,
      graph.mean(of: inputs.advantages, axes: [0], name: "terminal_ppo/mean_advantage"),
      meanImportanceRatio,
      meanSquaredRatioDeviation
    )
  }

  private func makeAdamUpdates(
    loss: MPSGraphTensor,
    inputs: GraphInputs,
    objective: String
  ) -> [MPSGraphOperation] {
    let trainable = store.parameters.map(\.values)
    let gradients = graph.gradients(
      of: loss,
      with: trainable,
      name: "adam/\(objective)/gradients"
    )
    let beta1 = graph.constant(0.9, dataType: .float32)
    let beta2 = graph.constant(0.999, dataType: .float32)
    let epsilon = graph.constant(1.0e-8, dataType: .float32)
    return store.parameters.flatMap { parameter in
      guard let gradient = gradients[parameter.values] else {
        preconditionFailure("missing gradient for \(parameter.name)")
      }
      let updates = graph.adam(
        learningRate: inputs.learningRate,
        beta1: beta1,
        beta2: beta2,
        epsilon: epsilon,
        beta1Power: inputs.beta1Power,
        beta2Power: inputs.beta2Power,
        values: parameter.values,
        momentum: parameter.momentum,
        velocity: parameter.velocity,
        maximumVelocity: nil,
        gradient: gradient,
        name: "\(parameter.name)/adam/\(objective)"
      )
      return [
        graph.assign(
          parameter.values,
          tensor: updates[0],
          name: "\(parameter.name)/assign/\(objective)"
        ),
        graph.assign(
          parameter.momentum,
          tensor: updates[1],
          name: "\(parameter.name)/assign_m/\(objective)"
        ),
        graph.assign(
          parameter.velocity,
          tensor: updates[2],
          name: "\(parameter.name)/assign_v/\(objective)"
        ),
      ]
    }
  }

  private func convolution(
    _ input: MPSGraphTensor,
    kernel: Int,
    inputChannels: Int,
    outputChannels: Int,
    name: String,
    weightInitialization: ParameterInitialization? = nil
  ) throws -> MPSGraphTensor {
    let weights = try store.make(
      name: "\(name)/weights",
      shape: [kernel, kernel, inputChannels, outputChannels],
      initialization: weightInitialization
        ?? .heUniform(fanIn: kernel * kernel * inputChannels)
    )
    let bias = try store.make(
      name: "\(name)/bias",
      shape: [outputChannels],
      initialization: .zeros
    )
    return graph.addition(
      graph.convolution2D(
        input,
        weights: weights,
        descriptor: convolutionDescriptor,
        name: name
      ),
      bias,
      name: "\(name)/biased"
    )
  }

  private func dense(
    _ inputTensor: MPSGraphTensor,
    input: Int,
    output: Int,
    name: String,
    weightInitialization: ParameterInitialization? = nil
  ) throws -> MPSGraphTensor {
    let weights = try store.make(
      name: "\(name)/weights",
      shape: [input, output],
      initialization: weightInitialization ?? .heUniform(fanIn: input)
    )
    let bias = try store.make(
      name: "\(name)/bias",
      shape: [output],
      initialization: .zeros
    )
    return graph.addition(
      graph.matrixMultiplication(primary: inputTensor, secondary: weights, name: name),
      bias,
      name: "\(name)/biased"
    )
  }

  private func channelNormalization(
    _ input: MPSGraphTensor,
    channels: Int,
    name: String
  ) throws -> MPSGraphTensor {
    let gamma = try store.make(
      name: "\(name)/gamma",
      shape: [1, 1, 1, channels],
      initialization: .ones
    )
    let beta = try store.make(
      name: "\(name)/beta",
      shape: [1, 1, 1, channels],
      initialization: .zeros
    )
    let mean = graph.mean(of: input, axes: [3], name: "\(name)/mean")
    let variance = graph.variance(of: input, mean: mean, axes: [3], name: "\(name)/variance")
    return graph.normalize(
      input,
      mean: mean,
      variance: variance,
      gamma: gamma,
      beta: beta,
      epsilon: configuration.normalizationEpsilon,
      name: name
    )
  }
}
