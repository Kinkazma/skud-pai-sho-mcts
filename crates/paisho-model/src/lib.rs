//! Backend-neutral neural inputs and policy addresses for standard Skud Pai Sho.

mod action;
mod checkpoint_wire;
mod inference_wire;
mod perspective;
mod schema;
mod state;
mod terminal_ppo_wire;
mod training_wire;
mod value;

pub use action::{
    encode_action_v1, encode_legal_actions_v1, ActionEncodingError, ActionEncodingV1,
    ActionFamilyV1, ACTION_FAMILY_COUNT_V1, ACTION_SLOT_COUNT_V1, NO_COORDINATE_V1, NO_TILE_V1,
};
pub use checkpoint_wire::{
    CheckpointRandomStateV1, CheckpointRequestV1, CheckpointResponseV1, CheckpointWireError,
    CHECKPOINT_ERROR_MAGIC_V1, CHECKPOINT_REQUEST_MAGIC_V1, CHECKPOINT_RESPONSE_MAGIC_V1,
    MAXIMUM_CHECKPOINT_REQUEST_PAYLOAD_V1,
};
pub use inference_wire::{
    InferenceExampleEncodingError, InferenceExampleV1, InferenceOutputV1, InferenceRequestV1,
    InferenceResponseV1, InferenceWireError, INFERENCE_ERROR_MAGIC_V1, INFERENCE_REQUEST_MAGIC_V1,
    INFERENCE_RESPONSE_MAGIC_V1,
};
pub use perspective::{canonical_coordinate_v1, native_coordinate_v1};
pub use schema::{
    special_slot_v1, tile_kind_v1, tile_slot_v1, BOARD_CELL_COUNT_V1, BOARD_SIZE_V1, TILE_KINDS_V1,
    TILE_KIND_COUNT_V1,
};
pub use state::{
    current_tile_channel_v1, encode_state_v1, opponent_tile_channel_v1, StateEncodingError,
    StateEncodingV1, CURRENT_RESERVE_START_V1, CURRENT_TILE_CHANNEL_START_V1, GATE_CHANNEL_V1,
    GLOBAL_FEATURE_COUNT_V1, HARMONY_BONUS_PHASE_FEATURE_V1, MAIN_PHASE_FEATURE_V1,
    NEUTRAL_REGION_CHANNEL_V1, OPPONENT_RESERVE_START_V1, OPPONENT_TILE_CHANNEL_START_V1,
    PLAYABLE_CHANNEL_V1, RED_REGION_CHANNEL_V1, SPATIAL_CHANNEL_COUNT_V1, SPATIAL_VALUE_COUNT_V1,
    WHITE_REGION_CHANNEL_V1,
};
pub use terminal_ppo_wire::{
    TerminalPpoExampleV1, TerminalPpoParametersV1, TerminalPpoRequestV1, TerminalPpoResponseV1,
    TerminalPpoWireError, TERMINAL_PPO_ERROR_MAGIC_V1, TERMINAL_PPO_OBJECTIVE_V1,
    TERMINAL_PPO_REQUEST_MAGIC_V1, TERMINAL_PPO_RESPONSE_MAGIC_V1,
};
pub use training_wire::{
    TrainingExampleV1, TrainingRequestV1, TrainingResponseV1, TrainingWireError,
    TRAINING_ERROR_MAGIC_V1, TRAINING_REQUEST_MAGIC_V1, TRAINING_RESPONSE_MAGIC_V1,
};
pub use value::{ValueClassV1, ValueTargetError, VALUE_CLASS_COUNT_V1};
