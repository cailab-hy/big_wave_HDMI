# from https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# from whole_body_tracking.assets import ASSET_DIR
ASSET_PATH = os.path.dirname(__file__)

# ============================================================================
# TienkungPro Training Config (Lightweight - decimated meshes, no cameras)
# Mesh size: ~10MB (vs original ~120MB), supports 4096 parallel envs
# ============================================================================
TIENKUNG_PRO_TRAINING_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSET_PATH}/tienkung_pro_collision_like_full/tienkung_pro/urdf/tiangong2.0_pro_training_hand-feet_box-eef_box-body_capsule_like.urdf",
        # asset_path=f"{ASSET_PATH}/tienkung_pro/urdf/tiangong2.0_pro_training_hand.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None
            )
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={  # Isaac Idx: real robot's joint name
            "body_yaw_joint": 0.0,  #        00: waist_yaw
            "hip_roll_l_joint": 0.0,  #        01: l_hip_roll
            "hip_roll_r_joint": 0.0,  #        02: r_hip_roll
            "head_yaw_joint": 0.0,  #        03: head_yaw
            "shoulder_pitch_l_joint": 0.0,  #        04: l_shoulder_pitch
            "shoulder_pitch_r_joint": 0.0,  #        05: r_shoulder_pitch
            "hip_pitch_l_joint": -0.5,  #        06: l_hip_pitch
            "hip_pitch_r_joint": -0.5,  #        07: r_hip_pitch
            "head_pitch_joint": 0.0,  #        08: head_pitch
            "shoulder_roll_l_joint": 0.1,  #        09: l_shoulder_roll
            "shoulder_roll_r_joint": -0.1,  #        10: r_shoulder_roll
            "hip_yaw_l_joint": 0.0,  #        11: l_hip_yaw
            "hip_yaw_r_joint": 0.0,  #        12: r_hip_yaw
            "head_roll_joint": 0.0,  #        13: head_roll
            "shoulder_yaw_l_joint": -0.3,  #        14: l_shoulder_yaw
            "shoulder_yaw_r_joint": -0.3,  #        15: r_shoulder_yaw
            "knee_pitch_l_joint": 1.0,  #        16: l_knee
            "knee_pitch_r_joint": 1.0,  #        17: r_knee
            "elbow_pitch_l_joint": 0.0,  #        18: l_elbow
            "elbow_pitch_r_joint": 0.0,  #        19: r_elbow
            "ankle_pitch_l_joint": -0.5,  #        20: l_ankle_pitch
            "ankle_pitch_r_joint": -0.5,  #        21: r_ankle_pitch
            "elbow_yaw_l_joint": 0.0,  #        22: l_wrist_yaw
            "elbow_yaw_r_joint": 0.0,  #        23: r_wrist_yaw
            "ankle_roll_l_joint": 0.0,  #        24: l_ankle_roll
            "ankle_roll_r_joint": 0.0,  #        25: r_ankle_roll
            "wrist_pitch_l_joint": 0.0,  #        26: l_wrist_pitch
            "wrist_pitch_r_joint": 0.0,  #        27: r_wrist_pitch
            "wrist_roll_l_joint": 0.0,  #        28: l_wrist_roll
            "wrist_roll_r_joint": 0.0,  #        29: r_wrist_roll
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "hip_yaw_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 124.0,
                "hip_pitch_.*_joint": 300.0,
                "hip_yaw_.*_joint": 124.0,
                "knee_pitch_.*_joint": 300.0,
                },
            velocity_limit_sim={
                "hip_roll_.*_joint": 13.1,
                "hip_pitch_.*_joint": 13.1,
                "hip_yaw_.*_joint": 13.1,
                "knee_pitch_.*_joint": 13.1,           
                },
            stiffness={
                "hip_roll_.*_joint": 700.0,
                "hip_pitch_.*_joint": 700.0,
                "hip_yaw_.*_joint": 500.0,
                "knee_pitch_.*_joint": 700.0,
            },
            damping={
                "hip_roll_.*_joint": 10.0,
                "hip_pitch_.*_joint": 10.0,
                "hip_yaw_.*_joint": 6.0,
                "knee_pitch_.*_joint": 10.0,
            },
),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_pitch_.*_joint",
                "ankle_roll_.*_joint",
            ],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 36.0,
                "ankle_roll_.*_joint": 36.0,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 13.4,
                "ankle_roll_.*_joint": 13.4,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30.0,
                "ankle_roll_.*_joint": 22.0,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
                "ankle_roll_.*_joint": 2.0,
            },
        ),        
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
                "wrist_pitch_.*_joint",
                "wrist_roll_.*_joint",
                "elbow_yaw_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 91.0,
                "shoulder_roll_.*_joint": 60.0,
                "shoulder_yaw_.*_joint": 30.0,
                "elbow_pitch_.*_joint": 30.0,
                "wrist_pitch_.*_joint": 9.0,
                "wrist_roll_.*_joint": 9.0,
                "elbow_yaw_.*_joint": 15.0,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 9.2,
                "shoulder_roll_.*_joint": 12.6,
                "shoulder_yaw_.*_joint": 7.6,
                "elbow_pitch_.*_joint": 7.6,
                "wrist_pitch_.*_joint": 7.5,
                "wrist_roll_.*_joint": 7.5,
                "elbow_yaw_.*_joint": 15.3,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60.0,
                "shoulder_roll_.*_joint": 50.0,
                "shoulder_yaw_.*_joint": 30.0,
                "elbow_pitch_.*_joint": 25.0,
                "wrist_pitch_.*_joint": 8.0,
                "wrist_roll_.*_joint": 8.0,
                "elbow_yaw_.*_joint": 12.0,
            },
            damping={
                "shoulder_pitch_.*_joint": 5.0,
                "shoulder_roll_.*_joint": 4.0,
                "shoulder_yaw_.*_joint": 3.0,
                "elbow_pitch_.*_joint": 2.5,
                "wrist_pitch_.*_joint": 1.0,
                "wrist_roll_.*_joint": 1.0,
                "elbow_yaw_.*_joint": 1.5,
            },
        ),
        "torso_head": ImplicitActuatorCfg(
            joint_names_expr=[
                "body_yaw_joint",
                "head_yaw_joint",
                "head_pitch_joint",
                "head_roll_joint",
            ],
        effort_limit_sim={
            "body_yaw_joint": 91.0,

            "head_yaw_joint": 9.0,
            "head_pitch_joint": 9.0,
            "head_roll_joint": 9.0,
        },
        velocity_limit_sim={
            "body_yaw_joint": 9.2,

            "head_yaw_joint": 6.7,
            "head_pitch_joint": 6.7,
            "head_roll_joint": 6.7,
        },
        stiffness={
            "body_yaw_joint": 40.0,
            "head_yaw_joint": 8.0,
            "head_pitch_joint": 8.0,
            "head_roll_joint": 8.0,
        },
        damping={
            "body_yaw_joint": 3.0,
            "head_yaw_joint": 0.5,
            "head_pitch_joint": 0.5,
            "head_roll_joint": 0.5,
        },
    ),
        }
)

TIENKUNG_PRO_TRAINING_ACTION_SCALE = {}
for a in TIENKUNG_PRO_TRAINING_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            TIENKUNG_PRO_TRAINING_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
