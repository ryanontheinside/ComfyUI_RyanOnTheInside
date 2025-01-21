import numpy as np
from ... import RyanOnTheInside
from comfy.utils import ProgressBar
from ...tooltips import apply_tooltips
from ... import ProgressMixin
@apply_tooltips
class PoseInterpolator(ProgressMixin):
    @classmethod

    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_1": ("POSE_KEYPOINT",),
                "pose_2": ("POSE_KEYPOINT",),
                "feature": ("FEATURE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interpolation_mode": (["Linear", "Spherical"], {"default": "Linear"}),
                "omit_missing_points": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "interpolate_poses"
    CATEGORY = "RyanOnTheInside/ExperimentalWIP"

    def interpolate_poses(self, pose_1, pose_2, feature, strength, interpolation_mode, omit_missing_points):
        print("Debug: Starting interpolate_poses method")
        num_frames = feature.frame_count
        self.start_progress(num_frames)


        result_poses = []

        for i in range(num_frames):
            pose_data_1 = pose_1[i]
            pose_data_2 = pose_2[i]
            t = feature.get_value_at_frame(i)

            # Match poses before interpolation
            matched_poses = self.match_poses(pose_data_1['people'], pose_data_2['people'])
            print(f"Debug: matched_poses: {matched_poses}")

            interpolated_people = []
            for start_idx, end_idx in matched_poses:
                person1 = pose_data_1['people'][start_idx]
                person2 = pose_data_2['people'][end_idx]
                interpolated_person = self.interpolate_person_keypoints(
                    person1, person2, t, interpolation_mode, omit_missing_points
                )
                interpolated_people.append(interpolated_person)

            result_poses.append({'people': interpolated_people, 'canvas_height': pose_data_1['canvas_height'], 'canvas_width': pose_data_1['canvas_width']})
            self.update_progress(1)


        self.end_progress()
        return (result_poses,)


    def match_poses(self, start_poses, end_poses):
        pose_count = min(len(start_poses), len(end_poses))
        pose_matches_final = []
        pose_match_candidates = []

        for start_idx, start_pose in enumerate(start_poses):
            for end_idx, end_pose in enumerate(end_poses):
                start_keypoints = np.array(start_pose['pose_keypoints_2d']).reshape(-1, 3)
                end_keypoints = np.array(end_pose['pose_keypoints_2d']).reshape(-1, 3)
                
                # Calculate distance between poses
                valid_points = (start_keypoints[:, 2] > 0) & (end_keypoints[:, 2] > 0)
                if np.sum(valid_points) > 0:
                    distance = np.mean(np.linalg.norm(start_keypoints[valid_points, :2] - end_keypoints[valid_points, :2], axis=1))
                    pose_match_candidates.append((distance, start_idx, end_idx))

        pose_match_candidates.sort(key=lambda x: x[0])
        for _, start_idx, end_idx in pose_match_candidates:
            if start_idx not in [x[0] for x in pose_matches_final] and end_idx not in [x[1] for x in pose_matches_final]:
                pose_matches_final.append((start_idx, end_idx))
                if len(pose_matches_final) == pose_count:
                    break

        return pose_matches_final

    def interpolate_person_keypoints(self, person1, person2, t, interpolation_mode, omit_missing_points):
        """Interpolate keypoints for a single person."""
        interpolated_person = {}  # Initialize the dictionary here

        keypoints1 = np.array(person1['pose_keypoints_2d']).reshape(-1, 3)
        keypoints2 = np.array(person2['pose_keypoints_2d']).reshape(-1, 3)

        interpolated_keypoints = []
        for start_keyp, end_keyp in zip(keypoints1, keypoints2):
            if np.all(start_keyp == 0) and np.all(end_keyp == 0):
                interpolated_keypoints.extend([0, 0, 0])
            elif np.all(start_keyp == 0) or np.all(end_keyp == 0):
                if omit_missing_points:
                    interpolated_keypoints.extend([0, 0, 0])
                else:
                    interpolated_keypoints.extend(end_keyp if np.all(start_keyp == 0) else start_keyp)
            else:
                if interpolation_mode == 'Linear':
                    interp_keyp = (1 - t) * start_keyp + t * end_keyp
                else:
                    interp_keyp = self.slerp_keypoints(start_keyp[:2], end_keyp[:2], t)
                    interp_keyp = np.append(interp_keyp, (1 - t) * start_keyp[2] + t * end_keyp[2])
                interpolated_keypoints.extend(interp_keyp)

        interpolated_person['pose_keypoints_2d'] = interpolated_keypoints

        # Add interpolation for face and hand keypoints
        for keypoint_type in ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            keypoints1 = np.array(person1.get(keypoint_type, [])).reshape(-1, 3)
            keypoints2 = np.array(person2.get(keypoint_type, [])).reshape(-1, 3)
            
            if len(keypoints1) == 0 or len(keypoints2) == 0:
                interpolated_person[keypoint_type] = person1.get(keypoint_type, [])
                continue

            interpolated_keypoints = []
            for start_keyp, end_keyp in zip(keypoints1, keypoints2):
                if np.all(start_keyp == 0) and np.all(end_keyp == 0):
                    interpolated_keypoints.extend([0, 0, 0])
                elif np.all(start_keyp == 0) or np.all(end_keyp == 0):
                    if omit_missing_points:
                        interpolated_keypoints.extend([0, 0, 0])
                    else:
                        interpolated_keypoints.extend(end_keyp if np.all(start_keyp == 0) else start_keyp)
                else:
                    if interpolation_mode == 'Linear':
                        interp_keyp = (1 - t) * start_keyp + t * end_keyp
                    else:
                        interp_keyp = self.slerp_keypoints(start_keyp[:2], end_keyp[:2], t)
                        interp_keyp = np.append(interp_keyp, (1 - t) * start_keyp[2] + t * end_keyp[2])
                    interpolated_keypoints.extend(interp_keyp)
            
            interpolated_person[keypoint_type] = interpolated_keypoints

        return interpolated_person

    def slerp_keypoints(self, kp1, kp2, t):
        """Spherical interpolation for keypoints."""
        # Avoid division by zero
        norm1 = np.linalg.norm(kp1) + 1e-8
        norm2 = np.linalg.norm(kp2) + 1e-8

        # Compute the cosine of the angle between the vectors
        cos_omega = np.clip(np.dot(kp1, kp2) / (norm1 * norm2), -1.0, 1.0)
        omega = np.arccos(cos_omega)

        if omega == 0:
            return kp1
        else:
            sin_omega = np.sin(omega)
            coef1 = np.sin((1 - t) * omega) / sin_omega
            coef2 = np.sin(t * omega) / sin_omega
            return coef1 * kp1 + coef2 * kp2
