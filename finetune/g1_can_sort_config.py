"""
Cấu hình modality của dataset phân loại lon (collect_demos.py) cho fine-tune GR00T N1.7 với Isaac-GR00T.

Layout state/action giống embodiment REAL_G1 của checkpoint base (real_g1_relative_eef_relative_joints) để client
và server của hướng A dùng lại nguyên vẹn, nhưng đăng ký dưới tag NEW_EMBODIMENT: fine-tune dưới tag REAL_G1 sẽ
giữ thống kê chuẩn hoá của dữ liệu robot thật trong checkpoint base và bỏ qua thống kê của dữ liệu sim
(Gr00tN1d7Processor.set_statistics với override=False).

Dùng với:
    python gr00t/data/stats.py --dataset-path <dataset> --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path finetune/g1_can_sort_config.py
    bash examples/finetune.sh ... --embodiment-tag NEW_EMBODIMENT --modality-config-path finetune/g1_can_sort_config.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


def _relative_eef(state_key):
    return ActionConfig(
        rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROT6D, state_key=state_key
    )


def _joints(rep, state_key=None):
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT, state_key=state_key)


g1_can_sort_config = {
    # Khung hiện tại và khung 20 bước trước (0.8 s ở 25 Hz), như REAL_G1
    "video": ModalityConfig(delta_indices=[-20, 0], modality_keys=["ego_view"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_wrist_eef_9d", "right_wrist_eef_9d", "left_hand", "right_hand", "left_arm", "right_arm", "waist",
        ],
    ),
    # Chunk 40 bước (1.6 s ở 25 Hz)
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "left_wrist_eef_9d", "right_wrist_eef_9d", "left_hand", "right_hand", "left_arm", "right_arm", "waist",
            "base_height_command", "navigate_command",
        ],
        action_configs=[
            _relative_eef("left_wrist_eef_9d"),
            _relative_eef("right_wrist_eef_9d"),
            _joints(ActionRepresentation.ABSOLUTE, "left_hand"),
            _joints(ActionRepresentation.ABSOLUTE, "right_hand"),
            _joints(ActionRepresentation.RELATIVE, "left_arm"),
            _joints(ActionRepresentation.RELATIVE, "right_arm"),
            _joints(ActionRepresentation.ABSOLUTE, "waist"),
            _joints(ActionRepresentation.ABSOLUTE),
            _joints(ActionRepresentation.ABSOLUTE),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(g1_can_sort_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
