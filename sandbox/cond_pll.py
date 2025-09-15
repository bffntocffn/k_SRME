import os, sys, warnings

import multiprocessing
import datetime
# import traceback

from copy import deepcopy
import json, pickle, h5py

import pandas as pd

from ase.io import read
from phono3py import load

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

FREQUENCY_THRESHOLD = -1e-2
MODE_KAPPA_THRESHOLD = 1e-6

cond_keys = ['frequency', 'gamma', 'gamma_isotope', 'grid_point_count', 'grid_points', 'grid_weights', 'gv_by_gv_operator', 'kappa_C', 'kappa_P_RTA', 'kappa_TOT_RTA', 'mode_heat_capacities', 'mode_kappa_C', 'mode_kappa_P_RTA', 'number_of_ignored_phonon_modes', 'qpoints', 'temperatures', 'velocity_operator']

def check_imaginary_freqs(frequencies: np.ndarray):
    try:
        if np.all(pd.isna(frequencies)):
            return True

        if np.any(frequencies[0, 3:] < 0):
            return True

        if np.any(frequencies[0, :3] < FREQUENCY_THRESHOLD):
            return True

        if np.any(frequencies[1:] < 0):
            return True
    except Exception as e:
        warnings.warn(f"Failed to check imaginary frequencies: {e!r}")
    return False

def please_work_please(atoms):
    idx = atoms.info['index']
    if atoms.info['fc2_error']:
        return 'FC2_ERROR'
    mesh = [19, 19, 15] if atoms.info['spg_num'] == 186 else [19, 19, 19]
    ph3.mesh_numbers = mesh
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_phonon_solver()
    freqs, _, _ = ph3.get_phonon_data()
     
    has_img = check_imaginary_freqs(freqs)
    cond_kwargs = {'temperatures': [300,],
                   'conductivity_type': 'wigner',
                   'is_isotope': False,
                   # 'boundary_mfp': None
                   }
    if has_img:
        return f'{idx}-IMG_ERROR'

    try:
        ph3.run_thermal_conductivity(**cond_kwargs)

    except Exception as exc:
        warnings.warn(f"Failed to calculate conductivity {atoms}: {exc!r}")
        return f'{idx}-KAPPA_ERROR'

    ph3.save(f'{head}/phonon/phono3py.yaml')
    cond = ph3.thermal_conductivity
    kappa = deepcopy(cond)

    with open(f'{head}/cond/kappa_{idx}.pkl', 'wb') as f:
        pickle.dump(cond, f)

    with h5py.File(f'{head}/cond/kappa_{idx}.hdf5', 'w') as f:
            g = f.create_group(f'{idx}')
            for key in cond_keys:
                try:
                    g.create_dataset(key, data = kappa.key)
                except:
                    continue
            g.attrs['spg_num'] = atoms.info['spg_num']
            g.attrs['formula'] = atoms.info['formula']
            g.attrs['idx'] = idx
            g.attrs['type'] = 'wigner'

    return f'{idx}-O'

if __name__ == "__main__":
    ############################ inintial setting ############################
    head = sys.argv[1]

    slurm_array_task_count = int(
        os.getenv(
            "K_SRME_RESTART_ARRAY_TASK_COUNT", os.getenv("SLURM_ARRAY_TASK_COUNT", "1")
        )
    )
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", os.getenv("SLURM_JOB_ID", "debug"))
    slurm_array_task_min = int(
        os.getenv("K_SRME_RESTART_ARRAY_TASK_MIN", os.getenv("SLURM_ARRAY_TASK_MIN", "0"))
    )

    task_type = "LTC"  # lattice thermal conductivity
    module_dir = os.path.dirname(__file__)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nJob LTC calc from {head}  started {timestamp}")

    atoms_list = read(f'{head}/relax/phonondb_relaxed.extxyz', format='extxyz', index=':')

    run_params = {
        "timestamp": timestamp,
        "slurm_array_task_count": slurm_array_task_count,
        "slurm_array_job_id": slurm_array_job_id,
        "task_type": task_type,
        "job_name": head,
        "struct_data_path": os.path.join(head, 'relax'),
        "n_structures": len(atoms_list),
    }

    if slurm_array_task_id == slurm_array_task_min:
        with open(f"{head}/run_params.json", "w") as f:
            json.dump(run_params, f, indent=4)
    
    if slurm_array_job_id == "debug":
        atoms_list = atoms_list[:5]
        print("Running in DEBUG mode.")

    elif slurm_array_task_count > 1:
        # Split the atoms_list into slurm_array_task_count parts trying to make even runtime
        atoms_list = atoms_list[
            slurm_array_task_id - slurm_array_task_min :: slurm_array_task_count
        ]

    ############################# main process #############################

    num_cores = int(sys.argv[2])
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(pool.imap(please_work_please, atoms_list))
    print(results)

