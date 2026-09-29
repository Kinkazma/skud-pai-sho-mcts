use core::fmt;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use minilp::{ComparisonOp, OptimizationDirection, Problem};
use nalgebra::{DMatrix, DVector};

use crate::{AgentId, RatedGame, RatedOutcome};

const ELO_PER_LOG_STRENGTH: f64 = 400.0 / core::f64::consts::LN_10;
const NORMAL_975: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DavidsonOptions {
    /// Conventional origin only; changing it cannot change estimated gaps.
    pub center_elo: f64,
    pub estimate_host_advantage: bool,
    pub maximum_iterations: usize,
    pub gradient_tolerance: f64,
    /// Numerical magnitude guard in natural-log strength units. Reaching it is
    /// reported separately from mathematically certified separation.
    pub separation_limit: f64,
}

impl Default for DavidsonOptions {
    fn default() -> Self {
        Self {
            center_elo: 1_500.0,
            estimate_host_advantage: true,
            maximum_iterations: 200,
            gradient_tolerance: 1e-6,
            separation_limit: 40.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uncertainty {
    pub standard_error: f64,
    pub interval_95: ConfidenceInterval,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterEstimate {
    pub estimate: f64,
    /// Inverse observed-information interval under the fitted model.
    pub model: Uncertainty,
    /// Pair-clustered CR1 sandwich interval, using a Student critical value.
    pub paired_cluster: Option<Uncertainty>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentRating {
    pub agent: AgentId,
    pub elo: ParameterEstimate,
    pub rated_games: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseEloDifference {
    pub first: AgentId,
    pub second: AgentId,
    /// Elo of `first` minus Elo of `second`, with covariance-aware uncertainty.
    pub first_minus_second: ParameterEstimate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TieModel {
    Davidson,
    /// No draw was observed, so the Davidson weight has its MLE at zero and
    /// the finite parameters are fitted with the Bradley–Terry boundary model.
    BradleyTerryBoundary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DavidsonFit {
    pub ratings: Vec<AgentRating>,
    pub pairwise_elo_differences: Vec<PairwiseEloDifference>,
    pub host_advantage_elo: Option<ParameterEstimate>,
    /// Natural logarithm of the Davidson draw weight. `None` means a zero
    /// boundary weight because no draw was observed.
    pub draw_log_weight: Option<ParameterEstimate>,
    pub tie_model: TieModel,
    pub games: usize,
    pub pairs: usize,
    pub iterations: usize,
    pub log_likelihood: f64,
    pub information_condition_number: f64,
}

impl DavidsonFit {
    pub fn draw_weight(&self) -> f64 {
        self.draw_log_weight
            .map(|parameter| parameter.estimate.exp())
            .unwrap_or(0.0)
    }

    pub fn elo_difference(&self, first: &AgentId, second: &AgentId) -> Option<ParameterEstimate> {
        if first == second {
            return Some(ParameterEstimate {
                estimate: 0.0,
                model: Uncertainty {
                    standard_error: 0.0,
                    interval_95: ConfidenceInterval {
                        lower: 0.0,
                        upper: 0.0,
                    },
                },
                paired_cluster: Some(Uncertainty {
                    standard_error: 0.0,
                    interval_95: ConfidenceInterval {
                        lower: 0.0,
                        upper: 0.0,
                    },
                }),
            });
        }
        self.pairwise_elo_differences.iter().find_map(|gap| {
            if &gap.first == first && &gap.second == second {
                Some(gap.first_minus_second)
            } else if &gap.first == second && &gap.second == first {
                Some(negate_estimate(gap.first_minus_second))
            } else {
                None
            }
        })
    }
}

/// Fits an unpenalized Bradley–Terry–Davidson maximum-likelihood model.
/// Agent strengths use an orthonormal zero-sum Helmert basis, while reversed
/// games are clustered by `pair_id` for robust uncertainty.
pub fn fit_davidson(
    games: &[RatedGame],
    options: DavidsonOptions,
) -> Result<DavidsonFit, DavidsonError> {
    let agents: Vec<_> = games
        .iter()
        .flat_map(|game| [game.host().clone(), game.guest().clone()])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    fit_davidson_for_agents(&agents, games, options)
}

/// Fits the model over an explicit agent universe. Unlike [`fit_davidson`],
/// this entry point rejects an expected agent that has no rating-eligible
/// game instead of silently dropping it from the result.
pub fn fit_davidson_for_agents(
    agents: &[AgentId],
    games: &[RatedGame],
    options: DavidsonOptions,
) -> Result<DavidsonFit, DavidsonError> {
    validate_options(options)?;
    let validated = ValidatedData::new(agents, games)?;
    let includes_draw_parameter = validated.draws > 0;
    if validated.decisions == 0 {
        return Err(DavidsonError::OnlyDraws);
    }
    let design = Design::new(
        validated.agents.len(),
        options.estimate_host_advantage,
        includes_draw_parameter,
    );
    reject_separated_likelihood(&validated, &design)?;
    let mut parameters = DVector::zeros(design.parameter_count);
    if let Some(draw_index) = design.draw_index {
        let draw_fraction = validated.draws as f64 / games.len() as f64;
        parameters[draw_index] = (2.0 * draw_fraction / (1.0 - draw_fraction)).ln();
    }

    let mut iterations = 0;
    loop {
        let evaluation = evaluate(&validated, &design, &parameters);
        if maximum_absolute(&parameters) >= options.separation_limit {
            return Err(DavidsonError::ParameterLimitExceeded);
        }
        let gradient_norm = maximum_absolute(&evaluation.gradient);
        if gradient_norm <= options.gradient_tolerance {
            break;
        }
        if iterations >= options.maximum_iterations {
            return Err(DavidsonError::NoConvergence);
        }
        let step = solve_newton_step(&evaluation.information, &evaluation.gradient)
            .ok_or(DavidsonError::NonIdentifiableDesign)?;
        let descent = -step;
        let slope = evaluation.gradient.dot(&descent);
        if !slope.is_finite() || slope >= 0.0 {
            return Err(DavidsonError::NumericalFailure);
        }
        let mut multiplier = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let candidate = &parameters + multiplier * &descent;
            let candidate_objective = objective(&validated, &design, &candidate);
            if candidate_objective.is_finite()
                && candidate_objective
                    <= evaluation.negative_log_likelihood + 1e-4 * multiplier * slope
            {
                accepted = Some(candidate);
                break;
            }
            multiplier *= 0.5;
        }
        parameters = accepted.ok_or(DavidsonError::NumericalFailure)?;
        iterations += 1;
        if maximum_absolute(&parameters) >= options.separation_limit {
            return Err(DavidsonError::ParameterLimitExceeded);
        }
    }

    let final_evaluation = evaluate(&validated, &design, &parameters);
    let eigenvalues = final_evaluation.information.clone().symmetric_eigenvalues();
    let minimum_eigenvalue = eigenvalues.min();
    let maximum_eigenvalue = eigenvalues.max();
    if !minimum_eigenvalue.is_finite()
        || minimum_eigenvalue <= f64::EPSILON * maximum_eigenvalue.max(1.0)
    {
        return Err(DavidsonError::NonIdentifiableDesign);
    }
    let condition_number = maximum_eigenvalue / minimum_eigenvalue;
    if !condition_number.is_finite() || condition_number > 1e14 {
        return Err(DavidsonError::NonIdentifiableDesign);
    }
    let model_covariance = final_evaluation
        .information
        .clone()
        .cholesky()
        .ok_or(DavidsonError::NonIdentifiableDesign)?
        .inverse();
    let cluster_covariance =
        paired_cluster_covariance(&validated, &design, &parameters, &model_covariance);
    let cluster_quantile = student_t_975(validated.pairs.len().saturating_sub(1));

    let skills = &design.helmert * parameters.rows(0, design.skill_parameters);
    let mut ratings = Vec::with_capacity(validated.agents.len());
    for (agent_index, agent) in validated.agents.iter().enumerate() {
        let mut contrast = DVector::zeros(design.parameter_count);
        for column in 0..design.skill_parameters {
            contrast[column] = design.helmert[(agent_index, column)] * ELO_PER_LOG_STRENGTH;
        }
        ratings.push(AgentRating {
            agent: agent.clone(),
            elo: parameter_estimate(
                options.center_elo + skills[agent_index] * ELO_PER_LOG_STRENGTH,
                &contrast,
                &model_covariance,
                cluster_covariance.as_ref(),
                cluster_quantile,
            )?,
            rated_games: validated.games_per_agent[agent_index],
        });
    }

    let mut pairwise_elo_differences = Vec::new();
    for first in 0..validated.agents.len() {
        for second in (first + 1)..validated.agents.len() {
            let mut contrast = DVector::zeros(design.parameter_count);
            for column in 0..design.skill_parameters {
                contrast[column] = (design.helmert[(first, column)]
                    - design.helmert[(second, column)])
                    * ELO_PER_LOG_STRENGTH;
            }
            pairwise_elo_differences.push(PairwiseEloDifference {
                first: validated.agents[first].clone(),
                second: validated.agents[second].clone(),
                first_minus_second: parameter_estimate(
                    (skills[first] - skills[second]) * ELO_PER_LOG_STRENGTH,
                    &contrast,
                    &model_covariance,
                    cluster_covariance.as_ref(),
                    cluster_quantile,
                )?,
            });
        }
    }

    let host_advantage_elo = design
        .host_index
        .map(|index| {
            let mut contrast = DVector::zeros(design.parameter_count);
            contrast[index] = ELO_PER_LOG_STRENGTH;
            parameter_estimate(
                parameters[index] * ELO_PER_LOG_STRENGTH,
                &contrast,
                &model_covariance,
                cluster_covariance.as_ref(),
                cluster_quantile,
            )
        })
        .transpose()?;
    let draw_log_weight = design
        .draw_index
        .map(|index| {
            let mut contrast = DVector::zeros(design.parameter_count);
            contrast[index] = 1.0;
            parameter_estimate(
                parameters[index],
                &contrast,
                &model_covariance,
                cluster_covariance.as_ref(),
                cluster_quantile,
            )
        })
        .transpose()?;

    Ok(DavidsonFit {
        ratings,
        pairwise_elo_differences,
        host_advantage_elo,
        draw_log_weight,
        tie_model: if includes_draw_parameter {
            TieModel::Davidson
        } else {
            TieModel::BradleyTerryBoundary
        },
        games: games.len(),
        pairs: validated.pairs.len(),
        iterations,
        log_likelihood: -final_evaluation.negative_log_likelihood,
        information_condition_number: condition_number,
    })
}

fn negate_estimate(value: ParameterEstimate) -> ParameterEstimate {
    let negate_uncertainty = |uncertainty: Uncertainty| Uncertainty {
        standard_error: uncertainty.standard_error,
        interval_95: ConfidenceInterval {
            lower: -uncertainty.interval_95.upper,
            upper: -uncertainty.interval_95.lower,
        },
    };
    ParameterEstimate {
        estimate: -value.estimate,
        model: negate_uncertainty(value.model),
        paired_cluster: value.paired_cluster.map(negate_uncertainty),
    }
}

fn validate_options(options: DavidsonOptions) -> Result<(), DavidsonError> {
    if !options.center_elo.is_finite()
        || options.maximum_iterations == 0
        || !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.0
        || !options.separation_limit.is_finite()
        || options.separation_limit <= 0.0
    {
        return Err(DavidsonError::InvalidOptions);
    }
    Ok(())
}

struct ValidatedData<'a> {
    games: &'a [RatedGame],
    agents: Vec<AgentId>,
    agent_indices: BTreeMap<AgentId, usize>,
    games_per_agent: Vec<usize>,
    pairs: BTreeMap<u64, Vec<usize>>,
    draws: usize,
    decisions: usize,
}

impl<'a> ValidatedData<'a> {
    fn new(agents: &[AgentId], games: &'a [RatedGame]) -> Result<Self, DavidsonError> {
        if games.is_empty() {
            return Err(DavidsonError::NoGames);
        }
        if agents.len() < 2 {
            return Err(DavidsonError::TooFewAgents);
        }
        let mut agent_indices = BTreeMap::new();
        for (index, agent) in agents.iter().enumerate() {
            if agent_indices.insert(agent.clone(), index).is_some() {
                return Err(DavidsonError::DuplicateAgent(agent.clone()));
            }
        }
        let mut sequences = BTreeSet::new();
        let mut pairs: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut draws = 0;
        let mut games_per_agent = vec![0; agents.len()];
        for (index, game) in games.iter().enumerate() {
            if !sequences.insert(game.sequence()) {
                return Err(DavidsonError::DuplicateSequence(game.sequence()));
            }
            let host = agent_indices
                .get(game.host())
                .copied()
                .ok_or_else(|| DavidsonError::UnknownAgent(game.host().clone()))?;
            let guest = agent_indices
                .get(game.guest())
                .copied()
                .ok_or_else(|| DavidsonError::UnknownAgent(game.guest().clone()))?;
            games_per_agent[host] += 1;
            games_per_agent[guest] += 1;
            pairs.entry(game.pair_id()).or_default().push(index);
            draws += usize::from(game.outcome() == RatedOutcome::Draw);
        }
        for (pair_id, indices) in &pairs {
            if indices.len() != 2 {
                return Err(DavidsonError::MalformedPair { pair_id: *pair_id });
            }
            let first = &games[indices[0]];
            let second = &games[indices[1]];
            if first.host() != second.guest() || first.guest() != second.host() {
                return Err(DavidsonError::MalformedPair { pair_id: *pair_id });
            }
        }
        if let Some((index, _)) = games_per_agent
            .iter()
            .enumerate()
            .find(|(_, count)| **count == 0)
        {
            return Err(DavidsonError::UnratedAgent(agents[index].clone()));
        }
        let components = comparison_components(games, agents, &agent_indices);
        if components.len() != 1 {
            return Err(DavidsonError::DisconnectedComparisonGraph { components });
        }
        Ok(Self {
            games,
            agents: agents.to_vec(),
            agent_indices,
            games_per_agent,
            pairs,
            draws,
            decisions: games.len() - draws,
        })
    }
}

struct Design {
    helmert: DMatrix<f64>,
    skill_parameters: usize,
    host_index: Option<usize>,
    draw_index: Option<usize>,
    parameter_count: usize,
}

impl Design {
    fn new(agent_count: usize, host: bool, draw: bool) -> Self {
        let skill_parameters = agent_count - 1;
        let host_index = host.then_some(skill_parameters);
        let draw_index = draw.then_some(skill_parameters + usize::from(host));
        Self {
            helmert: helmert_basis(agent_count),
            skill_parameters,
            host_index,
            draw_index,
            parameter_count: skill_parameters + usize::from(host) + usize::from(draw),
        }
    }

    fn game_contrast(&self, host: usize, guest: usize) -> DVector<f64> {
        let mut contrast = DVector::zeros(self.parameter_count);
        for column in 0..self.skill_parameters {
            contrast[column] = self.helmert[(host, column)] - self.helmert[(guest, column)];
        }
        if let Some(index) = self.host_index {
            contrast[index] = 1.0;
        }
        contrast
    }
}

struct Evaluation {
    negative_log_likelihood: f64,
    gradient: DVector<f64>,
    information: DMatrix<f64>,
}

fn evaluate(data: &ValidatedData<'_>, design: &Design, parameters: &DVector<f64>) -> Evaluation {
    let mut negative_log_likelihood = 0.0;
    let mut gradient = DVector::zeros(design.parameter_count);
    let mut information = DMatrix::zeros(design.parameter_count, design.parameter_count);
    for game in data.games {
        let contrast = design.game_contrast(
            data.agent_indices[game.host()],
            data.agent_indices[game.guest()],
        );
        let contribution = game_contribution(game.outcome(), design, parameters, &contrast);
        negative_log_likelihood += contribution.negative_log_likelihood;
        gradient += contribution.gradient;
        information += contribution.information;
    }
    Evaluation {
        negative_log_likelihood,
        gradient,
        information,
    }
}

struct GameContribution {
    negative_log_likelihood: f64,
    gradient: DVector<f64>,
    information: DMatrix<f64>,
}

fn game_contribution(
    outcome: RatedOutcome,
    design: &Design,
    parameters: &DVector<f64>,
    contrast: &DVector<f64>,
) -> GameContribution {
    let difference = contrast.dot(parameters);
    let host_logit = 0.5 * difference;
    let guest_logit = -host_logit;
    let draw_logit = design.draw_index.map(|index| parameters[index]);
    let maximum = draw_logit
        .map(|draw| host_logit.max(guest_logit).max(draw))
        .unwrap_or_else(|| host_logit.max(guest_logit));
    let host_weight = (host_logit - maximum).exp();
    let guest_weight = (guest_logit - maximum).exp();
    let draw_weight = draw_logit.map(|draw| (draw - maximum).exp()).unwrap_or(0.0);
    let denominator = host_weight + guest_weight + draw_weight;
    let host_probability = host_weight / denominator;
    let guest_probability = guest_weight / denominator;
    let draw_probability = draw_weight / denominator;
    let log_denominator = maximum + denominator.ln();
    let observed_logit = match outcome {
        RatedOutcome::HostWin => host_logit,
        RatedOutcome::GuestWin => guest_logit,
        RatedOutcome::Draw => draw_logit.expect("draw data always includes the draw parameter"),
    };

    let expected_contrast_coefficient = 0.5 * (host_probability - guest_probability);
    let observed_contrast_coefficient = match outcome {
        RatedOutcome::HostWin => 0.5,
        RatedOutcome::GuestWin => -0.5,
        RatedOutcome::Draw => 0.0,
    };
    let mut gradient = (expected_contrast_coefficient - observed_contrast_coefficient) * contrast;
    if let Some(draw_index) = design.draw_index {
        gradient[draw_index] += draw_probability - f64::from(outcome == RatedOutcome::Draw);
    }

    let contrast_variance =
        0.25 * (host_probability + guest_probability) - expected_contrast_coefficient.powi(2);
    let mut information = contrast_variance * (contrast * contrast.transpose());
    if let Some(draw_index) = design.draw_index {
        let covariance = -expected_contrast_coefficient * draw_probability;
        for index in 0..design.parameter_count {
            information[(index, draw_index)] += covariance * contrast[index];
            information[(draw_index, index)] += covariance * contrast[index];
        }
        information[(draw_index, draw_index)] += draw_probability * (1.0 - draw_probability);
    }

    GameContribution {
        negative_log_likelihood: log_denominator - observed_logit,
        gradient,
        information,
    }
}

fn objective(data: &ValidatedData<'_>, design: &Design, parameters: &DVector<f64>) -> f64 {
    data.games
        .iter()
        .map(|game| {
            let contrast = design.game_contrast(
                data.agent_indices[game.host()],
                data.agent_indices[game.guest()],
            );
            game_contribution(game.outcome(), design, parameters, &contrast).negative_log_likelihood
        })
        .sum()
}

fn solve_newton_step(information: &DMatrix<f64>, gradient: &DVector<f64>) -> Option<DVector<f64>> {
    let scale = information.diagonal().max().max(1.0);
    for attempt in 0..9 {
        let mut candidate = information.clone();
        if attempt > 0 {
            let damping = scale * 1e-12 * 10.0_f64.powi(attempt - 1);
            for index in 0..candidate.nrows() {
                candidate[(index, index)] += damping;
            }
        }
        if let Some(cholesky) = candidate.cholesky() {
            return Some(cholesky.solve(gradient));
        }
    }
    None
}

fn paired_cluster_covariance(
    data: &ValidatedData<'_>,
    design: &Design,
    parameters: &DVector<f64>,
    model_covariance: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    let cluster_count = data.pairs.len();
    let observation_count = data.games.len();
    let parameter_count = design.parameter_count;
    if cluster_count < 2 || observation_count <= parameter_count {
        return None;
    }
    let mut meat = DMatrix::zeros(parameter_count, parameter_count);
    for indices in data.pairs.values() {
        let mut score = DVector::zeros(parameter_count);
        for &index in indices {
            let game = &data.games[index];
            let contrast = design.game_contrast(
                data.agent_indices[game.host()],
                data.agent_indices[game.guest()],
            );
            score += game_contribution(game.outcome(), design, parameters, &contrast).gradient;
        }
        meat += &score * score.transpose();
    }
    let correction = cluster_count as f64 / (cluster_count - 1) as f64
        * (observation_count - 1) as f64
        / (observation_count - parameter_count) as f64;
    let covariance = correction * model_covariance * meat * model_covariance;
    let eigenvalues = covariance.clone().symmetric_eigenvalues();
    let minimum = eigenvalues.min();
    let maximum = eigenvalues.max();
    if !minimum.is_finite()
        || !maximum.is_finite()
        || minimum <= f64::EPSILON * maximum.max(1.0)
        || maximum / minimum > 1e14
    {
        return None;
    }
    Some(covariance)
}

fn parameter_estimate(
    estimate: f64,
    contrast: &DVector<f64>,
    model_covariance: &DMatrix<f64>,
    cluster_covariance: Option<&DMatrix<f64>>,
    cluster_quantile: f64,
) -> Result<ParameterEstimate, DavidsonError> {
    let model_variance = quadratic_form(contrast, model_covariance)?;
    let model = uncertainty(estimate, model_variance, NORMAL_975);
    let paired_cluster = cluster_covariance
        .map(|covariance| {
            quadratic_form(contrast, covariance)
                .map(|variance| uncertainty(estimate, variance, cluster_quantile))
        })
        .transpose()?;
    Ok(ParameterEstimate {
        estimate,
        model,
        paired_cluster,
    })
}

fn quadratic_form(
    contrast: &DVector<f64>,
    covariance: &DMatrix<f64>,
) -> Result<f64, DavidsonError> {
    let variance = (contrast.transpose() * covariance * contrast)[(0, 0)];
    if !variance.is_finite() || variance < -1e-9 {
        return Err(DavidsonError::NumericalFailure);
    }
    Ok(variance.max(0.0))
}

fn uncertainty(estimate: f64, variance: f64, quantile: f64) -> Uncertainty {
    let standard_error = variance.sqrt();
    Uncertainty {
        standard_error,
        interval_95: ConfidenceInterval {
            lower: estimate - quantile * standard_error,
            upper: estimate + quantile * standard_error,
        },
    }
}

fn helmert_basis(size: usize) -> DMatrix<f64> {
    DMatrix::from_fn(size, size - 1, |row, column| {
        let leading = column + 1;
        let denominator = ((leading * (leading + 1)) as f64).sqrt();
        match row.cmp(&leading) {
            core::cmp::Ordering::Less => 1.0 / denominator,
            core::cmp::Ordering::Equal => -(leading as f64) / denominator,
            core::cmp::Ordering::Greater => 0.0,
        }
    })
}

fn comparison_components(
    games: &[RatedGame],
    agents: &[AgentId],
    indices: &BTreeMap<AgentId, usize>,
) -> Vec<Vec<AgentId>> {
    let mut neighbors = vec![Vec::new(); agents.len()];
    for game in games {
        let host = indices[game.host()];
        let guest = indices[game.guest()];
        neighbors[host].push(guest);
        neighbors[guest].push(host);
    }
    let mut visited = vec![false; agents.len()];
    let mut components = Vec::new();
    for start in 0..agents.len() {
        if visited[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        visited[start] = true;
        let mut component = Vec::new();
        while let Some(index) = queue.pop_front() {
            component.push(agents[index].clone());
            for &neighbor in &neighbors[index] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort();
        components.push(component);
    }
    components.sort();
    components
}

/// Detects complete and quasi-complete separation before numerical fitting.
/// For each observation and every alternative class, a recession direction
/// must make the observed class at least as likely. A bounded LP maximizes the
/// sum of those margins; a positive optimum proves that the likelihood has no
/// finite maximum. The zero direction keeps this LP feasible for every input.
fn reject_separated_likelihood(
    data: &ValidatedData<'_>,
    design: &Design,
) -> Result<(), DavidsonError> {
    let mut rows = Vec::new();
    for game in data.games {
        let contrast = design.game_contrast(
            data.agent_indices[game.host()],
            data.agent_indices[game.guest()],
        );
        let host = 0.5 * &contrast;
        let guest = -0.5 * &contrast;
        let draw = design.draw_index.map(|index| {
            let mut features = DVector::zeros(design.parameter_count);
            features[index] = 1.0;
            features
        });
        let observed = match game.outcome() {
            RatedOutcome::HostWin => &host,
            RatedOutcome::GuestWin => &guest,
            RatedOutcome::Draw => draw
                .as_ref()
                .expect("draw observations always include the Davidson parameter"),
        };
        for alternative in [&host, &guest].into_iter().chain(draw.as_ref()) {
            if !core::ptr::eq(observed, alternative) {
                rows.push(observed - alternative);
            }
        }
    }

    let mut objective = vec![0.0; design.parameter_count];
    for row in &rows {
        for (index, value) in row.iter().copied().enumerate() {
            objective[index] += value;
        }
    }
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let variables: Vec<_> = objective
        .iter()
        .map(|coefficient| problem.add_var(*coefficient, (-1.0, 1.0)))
        .collect();
    for row in &rows {
        let expression: Vec<_> = row
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, coefficient)| *coefficient != 0.0)
            .map(|(index, coefficient)| (variables[index], coefficient))
            .collect();
        problem.add_constraint(&expression, ComparisonOp::Ge, 0.0);
    }
    let solution = problem
        .solve()
        .map_err(|_| DavidsonError::NumericalFailure)?;
    let tolerance = 1e-9 * rows.len().max(1) as f64;
    if solution.objective() > tolerance {
        return Err(DavidsonError::SeparatedLikelihood);
    }
    if solution.objective() < -tolerance || !solution.objective().is_finite() {
        return Err(DavidsonError::NumericalFailure);
    }
    for variable in variables {
        if !solution[variable].is_finite() {
            return Err(DavidsonError::NumericalFailure);
        }
    }
    Ok(())
}

fn maximum_absolute(vector: &DVector<f64>) -> f64 {
    vector.iter().copied().map(f64::abs).fold(0.0, f64::max)
}

fn student_t_975(degrees_of_freedom: usize) -> f64 {
    const SMALL: [f64; 30] = [
        12.706_204_736_432_095,
        4.302_652_729_696_142,
        3.182_446_305_284_263,
        2.776_445_105_197_799,
        2.570_581_835_636_314,
        2.446_911_851_144_969,
        2.364_624_251_592_784,
        2.306_004_135_204_166,
        2.262_157_162_854_099,
        2.228_138_851_964_939,
        2.200_985_160_082_949,
        2.178_812_829_663_418,
        2.160_368_656_461_013,
        2.144_786_687_916_927,
        2.131_449_545_559_323,
        2.119_905_299_221_011,
        2.109_815_577_833_181,
        2.100_922_040_240_96,
        2.093_024_054_408_263,
        2.085_963_447_265_837,
        2.079_613_844_727_662,
        2.073_873_067_904_015,
        2.068_657_610_419_041,
        2.063_898_561_628_021,
        2.059_538_552_753_294,
        2.055_529_438_642_871,
        2.051_830_516_480_283,
        2.048_407_141_795_244,
        2.045_229_642_132_703,
        2.042_272_456_301_237,
    ];
    if degrees_of_freedom == 0 {
        return f64::INFINITY;
    }
    if degrees_of_freedom <= SMALL.len() {
        return SMALL[degrees_of_freedom - 1];
    }
    let freedom = degrees_of_freedom as f64;
    let z = NORMAL_975;
    z + (z.powi(3) + z) / (4.0 * freedom)
        + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * freedom.powi(2))
        + (3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 15.0 * z)
            / (384.0 * freedom.powi(3))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DavidsonError {
    NoGames,
    TooFewAgents,
    DuplicateAgent(AgentId),
    UnknownAgent(AgentId),
    UnratedAgent(AgentId),
    DuplicateSequence(u64),
    MalformedPair { pair_id: u64 },
    DisconnectedComparisonGraph { components: Vec<Vec<AgentId>> },
    OnlyDraws,
    NonIdentifiableDesign,
    SeparatedLikelihood,
    ParameterLimitExceeded,
    NoConvergence,
    NumericalFailure,
    InvalidOptions,
}

impl fmt::Display for DavidsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGames => formatter.write_str("rating fit needs at least one game"),
            Self::TooFewAgents => formatter.write_str("rating fit needs at least two agents"),
            Self::DuplicateAgent(agent) => write!(formatter, "duplicate rating agent `{agent}`"),
            Self::UnknownAgent(agent) => {
                write!(formatter, "rated game references unknown agent `{agent}`")
            }
            Self::UnratedAgent(agent) => {
                write!(
                    formatter,
                    "expected agent `{agent}` has no rating-eligible game"
                )
            }
            Self::DuplicateSequence(sequence) => {
                write!(formatter, "duplicate game sequence {sequence}")
            }
            Self::MalformedPair { pair_id } => {
                write!(
                    formatter,
                    "pair {pair_id} is not exactly two reversed-seat games"
                )
            }
            Self::DisconnectedComparisonGraph { components } => {
                write!(
                    formatter,
                    "comparison graph has {} components",
                    components.len()
                )
            }
            Self::OnlyDraws => {
                formatter.write_str("an all-draw history has no finite Davidson fit")
            }
            Self::NonIdentifiableDesign => {
                formatter.write_str("rating design or information matrix is not identifiable")
            }
            Self::SeparatedLikelihood => {
                formatter.write_str("rating likelihood is separated and has no finite maximum")
            }
            Self::ParameterLimitExceeded => formatter
                .write_str("rating optimizer reached the configured parameter magnitude limit"),
            Self::NoConvergence => formatter.write_str("rating optimizer did not converge"),
            Self::NumericalFailure => {
                formatter.write_str("numerical failure while fitting ratings")
            }
            Self::InvalidOptions => formatter.write_str("invalid Davidson fitting options"),
        }
    }
}

impl std::error::Error for DavidsonError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> AgentId {
        AgentId::new(value).unwrap()
    }

    fn pair(pair_id: u64, first: RatedOutcome, second: RatedOutcome) -> [RatedGame; 2] {
        [
            RatedGame::new(pair_id * 2, pair_id, id("a"), id("b"), first).unwrap(),
            RatedGame::new(pair_id * 2 + 1, pair_id, id("b"), id("a"), second).unwrap(),
        ]
    }

    fn flatten(pairs: &[[RatedGame; 2]]) -> Vec<RatedGame> {
        pairs
            .iter()
            .flat_map(|games| games.iter().cloned())
            .collect()
    }

    fn rating<'a>(fit: &'a DavidsonFit, agent: &str) -> &'a AgentRating {
        fit.ratings
            .iter()
            .find(|rating| rating.agent.as_str() == agent)
            .unwrap()
    }

    #[test]
    fn decisive_two_agent_gap_matches_analytic_elo() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(2, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(3, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(4, RatedOutcome::GuestWin, RatedOutcome::HostWin),
        ]);
        let fit = fit_davidson(
            &games,
            DavidsonOptions {
                estimate_host_advantage: false,
                ..DavidsonOptions::default()
            },
        )
        .unwrap();
        let gap = rating(&fit, "a").elo.estimate - rating(&fit, "b").elo.estimate;
        assert!((gap - 400.0 * 4.0_f64.log10()).abs() < 1e-7);
        let direct = fit.elo_difference(&id("a"), &id("b")).unwrap();
        let reversed = fit.elo_difference(&id("b"), &id("a")).unwrap();
        assert!((direct.estimate - gap).abs() < 1e-7);
        assert!((reversed.estimate + gap).abs() < 1e-7);
        assert_eq!(direct.model.standard_error, reversed.model.standard_error);
        assert_eq!(
            direct.model.interval_95.lower,
            -reversed.model.interval_95.upper
        );
        assert_eq!(fit.tie_model, TieModel::BradleyTerryBoundary);
        assert_eq!(fit.draw_weight(), 0.0);
    }

    #[test]
    fn three_agent_synthetic_strengths_are_recovered() {
        let mut games = Vec::new();
        let mut pair_id = 0_u64;
        let mut add_pair = |first: &str, second: &str, first_wins: bool| {
            let first_outcome = if first_wins {
                RatedOutcome::HostWin
            } else {
                RatedOutcome::GuestWin
            };
            let reversed_outcome = if first_wins {
                RatedOutcome::GuestWin
            } else {
                RatedOutcome::HostWin
            };
            games.push(
                RatedGame::new(pair_id * 2, pair_id, id(first), id(second), first_outcome).unwrap(),
            );
            games.push(
                RatedGame::new(
                    pair_id * 2 + 1,
                    pair_id,
                    id(second),
                    id(first),
                    reversed_outcome,
                )
                .unwrap(),
            );
            pair_id += 1;
        };
        for first_wins in [true, true, false] {
            add_pair("a", "b", first_wins);
            add_pair("b", "c", first_wins);
        }
        for first_wins in [true, true, true, true, false] {
            add_pair("a", "c", first_wins);
        }

        let fit = fit_davidson(
            &games,
            DavidsonOptions {
                estimate_host_advantage: false,
                ..DavidsonOptions::default()
            },
        )
        .unwrap();
        let expected_adjacent_gap = 400.0 * 2.0_f64.log10();
        let a = rating(&fit, "a").elo.estimate;
        let b = rating(&fit, "b").elo.estimate;
        let c = rating(&fit, "c").elo.estimate;
        assert!((a - b - expected_adjacent_gap).abs() < 1e-7);
        assert!((b - c - expected_adjacent_gap).abs() < 1e-7);
    }

    #[test]
    fn host_advantage_matches_four_to_one_odds() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(2, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(3, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(4, RatedOutcome::GuestWin, RatedOutcome::GuestWin),
        ]);
        let fit = fit_davidson(&games, DavidsonOptions::default()).unwrap();
        let host = fit.host_advantage_elo.unwrap().estimate;
        assert!((host - 400.0 * 4.0_f64.log10()).abs() < 1e-7);
        assert!((rating(&fit, "a").elo.estimate - rating(&fit, "b").elo.estimate).abs() < 1e-8);
    }

    #[test]
    fn balanced_draw_fixture_recovers_unit_davidson_weight() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(2, RatedOutcome::GuestWin, RatedOutcome::GuestWin),
            pair(3, RatedOutcome::GuestWin, RatedOutcome::GuestWin),
            pair(4, RatedOutcome::Draw, RatedOutcome::Draw),
            pair(5, RatedOutcome::Draw, RatedOutcome::Draw),
        ]);
        let fit = fit_davidson(&games, DavidsonOptions::default()).unwrap();
        assert!(fit.host_advantage_elo.unwrap().estimate.abs() < 1e-8);
        assert!(fit.draw_log_weight.unwrap().estimate.abs() < 1e-8);
        assert!((fit.draw_weight() - 1.0).abs() < 1e-8);
        assert!(fit
            .ratings
            .iter()
            .all(|rating| rating.elo.paired_cluster.is_none()));
    }

    #[test]
    fn game_order_and_display_origin_do_not_change_gaps() {
        let mut games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::GuestWin, RatedOutcome::HostWin),
            pair(2, RatedOutcome::Draw, RatedOutcome::Draw),
        ]);
        let first = fit_davidson(&games, DavidsonOptions::default()).unwrap();
        games.reverse();
        let shifted = fit_davidson(
            &games,
            DavidsonOptions {
                center_elo: 1_700.0,
                ..DavidsonOptions::default()
            },
        )
        .unwrap();
        for agent in ["a", "b"] {
            assert!(
                (rating(&shifted, agent).elo.estimate - rating(&first, agent).elo.estimate - 200.0)
                    .abs()
                    < 1e-8
            );
            assert!(
                (rating(&shifted, agent).elo.model.standard_error
                    - rating(&first, agent).elo.model.standard_error)
                    .abs()
                    < 1e-10
            );
        }
    }

    #[test]
    fn malformed_disconnected_and_unbounded_histories_are_explicit() {
        let malformed =
            vec![RatedGame::new(0, 9, id("a"), id("b"), RatedOutcome::HostWin).unwrap()];
        assert_eq!(
            fit_davidson(&malformed, DavidsonOptions::default()),
            Err(DavidsonError::MalformedPair { pair_id: 9 })
        );

        let mut disconnected = flatten(&[pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin)]);
        disconnected.push(RatedGame::new(2, 1, id("c"), id("d"), RatedOutcome::HostWin).unwrap());
        disconnected.push(RatedGame::new(3, 1, id("d"), id("c"), RatedOutcome::GuestWin).unwrap());
        assert!(matches!(
            fit_davidson(&disconnected, DavidsonOptions::default()),
            Err(DavidsonError::DisconnectedComparisonGraph { .. })
        ));

        let dominated = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::GuestWin),
        ]);
        assert_eq!(
            fit_davidson(
                &dominated,
                DavidsonOptions {
                    estimate_host_advantage: false,
                    ..DavidsonOptions::default()
                }
            ),
            Err(DavidsonError::SeparatedLikelihood)
        );
    }

    #[test]
    fn explicit_agent_universe_rejects_an_unrated_agent() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::GuestWin, RatedOutcome::HostWin),
        ]);
        assert_eq!(
            fit_davidson_for_agents(
                &[id("a"), id("b"), id("c")],
                &games,
                DavidsonOptions::default(),
            ),
            Err(DavidsonError::UnratedAgent(id("c")))
        );
    }

    #[test]
    fn a_finite_extreme_fit_is_not_mislabeled_as_separation() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(2, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(3, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(4, RatedOutcome::GuestWin, RatedOutcome::HostWin),
        ]);
        assert_eq!(
            fit_davidson(
                &games,
                DavidsonOptions {
                    estimate_host_advantage: false,
                    separation_limit: 0.1,
                    ..DavidsonOptions::default()
                },
            ),
            Err(DavidsonError::ParameterLimitExceeded)
        );
    }

    #[test]
    fn exhausting_iterations_below_the_limit_is_no_convergence() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(2, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(3, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(4, RatedOutcome::GuestWin, RatedOutcome::HostWin),
        ]);
        assert_eq!(
            fit_davidson(
                &games,
                DavidsonOptions {
                    estimate_host_advantage: false,
                    maximum_iterations: 1,
                    separation_limit: 1.0,
                    ..DavidsonOptions::default()
                },
            ),
            Err(DavidsonError::NoConvergence)
        );
    }

    #[test]
    fn connected_ladder_pattern_converges_at_sub_millielo_accuracy() {
        let agents: Vec<_> = ["site", "m8", "m32", "m128", "m512"]
            .into_iter()
            .map(id)
            .collect();
        let edge_patterns: [&[(usize, &str)]; 5] = [
            &[(2, "GG"), (15, "GH"), (1, "HH")],
            &[(2, "DH"), (3, "GD"), (2, "GG"), (9, "GH"), (1, "HH")],
            &[
                (2, "DD"),
                (1, "DG"),
                (2, "DH"),
                (2, "GD"),
                (2, "GG"),
                (6, "GH"),
                (1, "HH"),
            ],
            &[(1, "DH"), (3, "GD"), (3, "GG"), (7, "GH"), (2, "HG")],
            &[(18, "GH")],
        ];
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)];
        let mut games = Vec::new();
        let mut pair_id = 0;
        for ((first, second), patterns) in edges.into_iter().zip(edge_patterns) {
            for (count, outcomes) in patterns {
                for _ in 0..*count {
                    let mut outcome = outcomes.chars().map(|code| match code {
                        'H' => RatedOutcome::HostWin,
                        'D' => RatedOutcome::Draw,
                        'G' => RatedOutcome::GuestWin,
                        _ => unreachable!(),
                    });
                    games.push(
                        RatedGame::new(
                            pair_id * 2,
                            pair_id,
                            agents[first].clone(),
                            agents[second].clone(),
                            outcome.next().unwrap(),
                        )
                        .unwrap(),
                    );
                    games.push(
                        RatedGame::new(
                            pair_id * 2 + 1,
                            pair_id,
                            agents[second].clone(),
                            agents[first].clone(),
                            outcome.next().unwrap(),
                        )
                        .unwrap(),
                    );
                    pair_id += 1;
                }
            }
        }

        let fit = fit_davidson_for_agents(&agents, &games, DavidsonOptions::default()).unwrap();
        let expected = [
            657.408_267,
            1_242.169_016,
            1_634.081_838,
            1_876.510_854,
            2_089.830_025,
        ];
        assert!(fit.iterations <= 20);
        for (agent, expected) in agents.iter().zip(expected) {
            assert!((rating(&fit, agent.as_str()).elo.estimate - expected).abs() < 0.001);
        }
        assert!((fit.host_advantage_elo.unwrap().estimate + 94.788_126).abs() < 0.001);
        assert!((fit.draw_log_weight.unwrap().estimate + 0.734_161).abs() < 0.000_001);
    }

    #[test]
    fn host_only_wins_are_detected_as_host_effect_separation() {
        let games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(2, RatedOutcome::HostWin, RatedOutcome::HostWin),
            pair(3, RatedOutcome::HostWin, RatedOutcome::HostWin),
        ]);
        assert_eq!(
            fit_davidson(&games, DavidsonOptions::default()),
            Err(DavidsonError::SeparatedLikelihood)
        );
    }

    #[test]
    fn a_draw_elsewhere_does_not_hide_skill_separation() {
        let mut games = flatten(&[
            pair(0, RatedOutcome::HostWin, RatedOutcome::GuestWin),
            pair(1, RatedOutcome::HostWin, RatedOutcome::GuestWin),
        ]);
        let outcomes = [
            (RatedOutcome::HostWin, RatedOutcome::GuestWin),
            (RatedOutcome::GuestWin, RatedOutcome::HostWin),
            (RatedOutcome::Draw, RatedOutcome::Draw),
        ];
        for (offset, (first, second)) in outcomes.into_iter().enumerate() {
            let pair_id = 2 + offset as u64;
            games.push(RatedGame::new(pair_id * 2, pair_id, id("b"), id("c"), first).unwrap());
            games.push(RatedGame::new(pair_id * 2 + 1, pair_id, id("c"), id("b"), second).unwrap());
        }
        assert_eq!(
            fit_davidson(&games, DavidsonOptions::default()),
            Err(DavidsonError::SeparatedLikelihood)
        );
    }

    #[test]
    fn all_draws_are_not_assigned_fabricated_finite_ratings() {
        let games = flatten(&[
            pair(0, RatedOutcome::Draw, RatedOutcome::Draw),
            pair(1, RatedOutcome::Draw, RatedOutcome::Draw),
        ]);
        assert_eq!(
            fit_davidson(&games, DavidsonOptions::default()),
            Err(DavidsonError::OnlyDraws)
        );
    }
}
