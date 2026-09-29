use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use paisho_model::{InferenceExampleV1, InferenceOutputV1};

use crate::{
    InferenceBroker, InferenceBrokerClient, InferenceBrokerConfiguration, InferenceBrokerError,
    InferenceBrokerTelemetry, ServiceConfiguration,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapacityClassConfiguration {
    pub service: ServiceConfiguration,
    pub lanes: usize,
}

struct CapacityClass {
    legal_action_capacity: usize,
    batch_size: usize,
    brokers: Vec<InferenceBroker>,
}

/// Routes each position to the smallest fixed MPSGraph action shape that fits it.
pub struct CapacityInferenceBroker {
    classes: Vec<CapacityClass>,
}

impl CapacityInferenceBroker {
    pub fn launch(
        service_configurations: Vec<ServiceConfiguration>,
        broker_configuration: InferenceBrokerConfiguration,
    ) -> Result<Self, InferenceBrokerError> {
        Self::launch_with_lanes(
            service_configurations
                .into_iter()
                .map(|service| CapacityClassConfiguration { service, lanes: 1 })
                .collect(),
            broker_configuration,
        )
    }

    pub fn launch_with_lanes(
        mut configurations: Vec<CapacityClassConfiguration>,
        broker_configuration: InferenceBrokerConfiguration,
    ) -> Result<Self, InferenceBrokerError> {
        prepare_configurations(&mut configurations)?;
        let mut classes = Vec::with_capacity(configurations.len());
        for configuration in configurations {
            let legal_action_capacity = configuration.service.legal_action_capacity;
            let batch_size = configuration.service.batch_size;
            let brokers = (0..configuration.lanes)
                .map(|_| {
                    InferenceBroker::launch(configuration.service.clone(), broker_configuration)
                })
                .collect::<Result<Vec<_>, _>>()?;
            classes.push(CapacityClass {
                legal_action_capacity,
                batch_size,
                brokers,
            });
        }
        Ok(Self { classes })
    }

    pub fn client(&self) -> Result<CapacityInferenceBrokerClient, InferenceBrokerError> {
        let classes = self
            .classes
            .iter()
            .map(|class| {
                Ok(CapacityClassClient {
                    legal_action_capacity: class.legal_action_capacity,
                    clients: class
                        .brokers
                        .iter()
                        .map(InferenceBroker::client)
                        .collect::<Result<Vec<_>, _>>()?,
                    next_lane: Arc::new(AtomicUsize::new(0)),
                })
            })
            .collect::<Result<Vec<_>, InferenceBrokerError>>()?;
        Ok(CapacityInferenceBrokerClient { classes })
    }

    pub fn capacities(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.classes.iter().map(|class| class.legal_action_capacity)
    }

    /// Call between completed actor rounds, never during a game. Failure aborts
    /// the round setup; callers must not play with a partially updated pool.
    pub fn update_services<F>(&mut self, update: F) -> Result<(), InferenceBrokerError>
    where
        F: Fn(&mut crate::MpsGraphProcess) -> Result<(), crate::MpsGraphClientError>
            + Clone
            + Send
            + 'static,
    {
        for class in &mut self.classes {
            for broker in &mut class.brokers {
                broker.update_service(update.clone())?;
            }
        }
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<CapacityInferenceTelemetry, InferenceBrokerError> {
        let mut classes = Vec::with_capacity(self.classes.len());
        for mut class in self.classes.drain(..) {
            let lanes = class
                .brokers
                .drain(..)
                .map(InferenceBroker::shutdown)
                .collect::<Result<Vec<_>, _>>()?;
            classes.push(CapacityClassTelemetry {
                legal_action_capacity: class.legal_action_capacity,
                batch_size: class.batch_size,
                broker: aggregate_telemetry(&lanes),
                lanes,
            });
        }
        Ok(CapacityInferenceTelemetry { classes })
    }
}

#[derive(Clone)]
struct CapacityClassClient {
    legal_action_capacity: usize,
    clients: Vec<InferenceBrokerClient>,
    next_lane: Arc<AtomicUsize>,
}

#[derive(Clone)]
pub struct CapacityInferenceBrokerClient {
    classes: Vec<CapacityClassClient>,
}

impl CapacityInferenceBrokerClient {
    pub fn infer(
        &self,
        position: &paisho_core::Position,
    ) -> Result<InferenceOutputV1, InferenceBrokerError> {
        let example =
            InferenceExampleV1::from_position(position).map_err(InferenceBrokerError::Encoding)?;
        self.infer_encoded(example)
    }

    pub fn infer_encoded(
        &self,
        example: InferenceExampleV1,
    ) -> Result<InferenceOutputV1, InferenceBrokerError> {
        let actual = example.legal_actions().len();
        let class = self
            .classes
            .iter()
            .find(|class| actual <= class.legal_action_capacity)
            .ok_or_else(|| InferenceBrokerError::TooManyLegalActions {
                capacity: self.maximum_capacity(),
                actual,
            })?;
        let lane = class.next_lane.fetch_add(1, Ordering::Relaxed) % class.clients.len();
        class.clients[lane].infer_encoded(example)
    }

    pub fn capacity_for(&self, legal_action_count: usize) -> Option<usize> {
        routed_capacity(
            self.classes.iter().map(|class| class.legal_action_capacity),
            legal_action_count,
        )
    }

    fn maximum_capacity(&self) -> usize {
        self.classes
            .last()
            .map_or(0, |class| class.legal_action_capacity)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapacityClassTelemetry {
    pub legal_action_capacity: usize,
    pub batch_size: usize,
    pub broker: InferenceBrokerTelemetry,
    pub lanes: Vec<InferenceBrokerTelemetry>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CapacityInferenceTelemetry {
    pub classes: Vec<CapacityClassTelemetry>,
}

impl CapacityInferenceTelemetry {
    pub fn requested_positions(&self) -> u64 {
        self.classes
            .iter()
            .map(|class| class.broker.requested_positions)
            .sum()
    }

    pub fn executed_positions(&self) -> u64 {
        self.classes
            .iter()
            .map(|class| class.broker.executed_positions)
            .sum()
    }
}

fn prepare_configurations(
    configurations: &mut [CapacityClassConfiguration],
) -> Result<(), InferenceBrokerError> {
    let Some(reference) = configurations
        .first()
        .map(|configuration| &configuration.service)
    else {
        return Err(InferenceBrokerError::NoCapacityClasses);
    };
    let reference = reference.clone();
    configurations.sort_by_key(|configuration| configuration.service.legal_action_capacity);
    for pair in configurations.windows(2) {
        if pair[0].service.legal_action_capacity == pair[1].service.legal_action_capacity {
            return Err(InferenceBrokerError::DuplicateCapacityClass(
                pair[0].service.legal_action_capacity,
            ));
        }
    }
    for configuration in configurations.iter() {
        if configuration.lanes == 0 {
            return Err(InferenceBrokerError::ZeroCapacityClassLanes {
                capacity: configuration.service.legal_action_capacity,
            });
        }
        if !configuration.service.runs_same_model_as(&reference) {
            return Err(InferenceBrokerError::InconsistentCapacityModel {
                capacity: configuration.service.legal_action_capacity,
            });
        }
    }
    Ok(())
}

fn aggregate_telemetry(lanes: &[InferenceBrokerTelemetry]) -> InferenceBrokerTelemetry {
    let mut aggregate = InferenceBrokerTelemetry::default();
    for lane in lanes {
        aggregate.batches += lane.batches;
        aggregate.full_batches += lane.full_batches;
        aggregate.requested_positions += lane.requested_positions;
        aggregate.executed_positions += lane.executed_positions;
        aggregate.padded_positions += lane.padded_positions;
        aggregate.maximum_observed_batch = aggregate
            .maximum_observed_batch
            .max(lane.maximum_observed_batch);
        aggregate.maximum_observed_in_flight_batches = aggregate
            .maximum_observed_in_flight_batches
            .max(lane.maximum_observed_in_flight_batches);
    }
    aggregate
}

fn routed_capacity(
    capacities: impl IntoIterator<Item = usize>,
    legal_action_count: usize,
) -> Option<usize> {
    capacities
        .into_iter()
        .find(|capacity| legal_action_count <= *capacity)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{NetworkPreset, OptimizationLevel};

    use super::*;

    fn configuration(capacity: usize, lanes: usize) -> CapacityClassConfiguration {
        CapacityClassConfiguration {
            service: ServiceConfiguration {
                executable: PathBuf::from("service"),
                preset: NetworkPreset::Pure,
                batch_size: 8,
                legal_action_capacity: capacity,
                inference_slots: 1,
                optimization: OptimizationLevel::Level1,
                seed: 17,
                checkpoint: Some(PathBuf::from("weights.psckpt")),
            },
            lanes,
        }
    }

    #[test]
    fn routing_uses_smallest_fitting_capacity() {
        assert_eq!(routed_capacity([128, 256, 1_024], 1), Some(128));
        assert_eq!(routed_capacity([128, 256, 1_024], 128), Some(128));
        assert_eq!(routed_capacity([128, 256, 1_024], 129), Some(256));
        assert_eq!(routed_capacity([128, 256, 1_024], 1_024), Some(1_024));
        assert_eq!(routed_capacity([128, 256, 1_024], 1_025), None);
    }

    #[test]
    fn configurations_are_sorted_and_require_one_model_identity() {
        let mut configurations = vec![
            configuration(1_024, 1),
            configuration(128, 1),
            configuration(256, 1),
        ];
        prepare_configurations(&mut configurations).unwrap();
        assert_eq!(
            configurations
                .iter()
                .map(|configuration| configuration.service.legal_action_capacity)
                .collect::<Vec<_>>(),
            [128, 256, 1_024]
        );

        configurations[1].service.optimization = OptimizationLevel::Level0;
        assert!(matches!(
            prepare_configurations(&mut configurations),
            Err(InferenceBrokerError::InconsistentCapacityModel { capacity: 256 })
        ));

        let mut checkpointed = vec![configuration(128, 1), configuration(256, 1)];
        checkpointed[1].service.seed += 1;
        prepare_configurations(&mut checkpointed).unwrap();
        checkpointed[0].service.checkpoint = None;
        checkpointed[1].service.checkpoint = None;
        assert!(matches!(
            prepare_configurations(&mut checkpointed),
            Err(InferenceBrokerError::InconsistentCapacityModel { capacity: 256 })
        ));
    }

    #[test]
    fn empty_and_duplicate_classes_are_rejected() {
        assert!(matches!(
            prepare_configurations(&mut []),
            Err(InferenceBrokerError::NoCapacityClasses)
        ));
        let mut duplicate = vec![configuration(128, 1), configuration(128, 2)];
        assert!(matches!(
            prepare_configurations(&mut duplicate),
            Err(InferenceBrokerError::DuplicateCapacityClass(128))
        ));

        let mut zero_lanes = vec![configuration(128, 0)];
        assert!(matches!(
            prepare_configurations(&mut zero_lanes),
            Err(InferenceBrokerError::ZeroCapacityClassLanes { capacity: 128 })
        ));
    }

    #[test]
    fn lane_telemetry_is_aggregated_without_losing_individual_counts() {
        let lanes = [
            InferenceBrokerTelemetry {
                batches: 2,
                full_batches: 1,
                requested_positions: 7,
                executed_positions: 8,
                padded_positions: 1,
                maximum_observed_batch: 4,
                maximum_observed_in_flight_batches: 1,
            },
            InferenceBrokerTelemetry {
                batches: 3,
                full_batches: 2,
                requested_positions: 10,
                executed_positions: 12,
                padded_positions: 2,
                maximum_observed_batch: 4,
                maximum_observed_in_flight_batches: 2,
            },
        ];
        assert_eq!(
            aggregate_telemetry(&lanes),
            InferenceBrokerTelemetry {
                batches: 5,
                full_batches: 3,
                requested_positions: 17,
                executed_positions: 20,
                padded_positions: 3,
                maximum_observed_batch: 4,
                maximum_observed_in_flight_batches: 2,
            }
        );
    }
}
