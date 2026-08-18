Source URDF: tiangong2.0_pro_training_hand.urdf
Output URDF: tiangong2.0_pro_training_hand-feet_box-eef_box-body_capsule_like.urdf
Removed original mesh collision blocks: 37
Links: 37
Joints: 36
New primitive collision blocks: 19
Collision geometry counts:
  - box: 9
  - cylinder: 10
New collision blocks:
  - pelvis: pelvis_contour_box (box)
  - hip_roll_l_link: hip_roll_l_capsule_like (cylinder)
  - hip_yaw_l_link: hip_yaw_l_capsule_like (cylinder)
  - knee_pitch_l_link: knee_pitch_l_capsule_like (cylinder)
  - ankle_roll_l_link: foot_l_box (box)
  - hip_roll_r_link: hip_roll_r_capsule_like (cylinder)
  - hip_yaw_r_link: hip_yaw_r_capsule_like (cylinder)
  - knee_pitch_r_link: knee_pitch_r_capsule_like (cylinder)
  - ankle_roll_r_link: foot_r_box (box)
  - body_yaw_link: torso_lower_capsule_like (cylinder)
  - body_yaw_link: torso_upper_capsule_like (cylinder)
  - elbow_yaw_l_link: elbow_yaw_l_capsule_like (cylinder)
  - wrist_roll_l_link: wrist_l_box (box)
  - left_tcp_link: left_tcp_box (box)
  - left_hand_link: left_hand_box (box)
  - elbow_yaw_r_link: elbow_yaw_r_capsule_like (cylinder)
  - wrist_roll_r_link: wrist_r_box (box)
  - right_tcp_link: right_tcp_box (box)
  - right_hand_link: right_hand_box (box)
Important note: URDF does not define a standard capsule primitive; body_capsule is approximated with cylinder primitives for Isaac/URDF compatibility.
