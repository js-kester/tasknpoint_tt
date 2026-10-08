import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# TaskNPoint expected header order for G1 (29-DOF):
# root_tx, root_ty, root_tz, root_qw, root_qx, root_qy, root_qz, joint_0 ... joint_28
def pkl_to_csv(pkl_path: Path, output_csv_path: Path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    root_pos = data['root_pos']  # Shape: (T, 3)
    root_rot = data['root_rot']  # Shape: (T, 4) - Quaternion [w, x, y, z]
    dof_pos = data['dof_pos']  # Shape: (T, 29)

    # Combine into a single matrix: [root_pos (3), root_rot (4), dof_pos (29)]
    motion_matrix = np.hstack([root_pos, root_rot, dof_pos])

    # Save as CSV without header rows (csv_to_npz reads raw matrix values)
    np.savetxt(output_csv_path, motion_matrix, delimiter=',')


# Example batch run:
input_dir = Path("retargeted/trial_05")
output_dir = Path("retarget/retarget_outputs/trial_05")
output_dir.mkdir(parents=True, exist_ok=True)

for pkl_file in input_dir.glob("*.pkl"):
    csv_file = output_dir / f"{pkl_file.stem}.csv"
    pkl_to_csv(pkl_file, csv_file)