"""Rejestr eksperymentow diagnostycznych.

Nazwa klucza = nazwa wpisu w diagnostics_report.json ORAZ argument --exp.
Nie zmieniac istniejacych kluczy: raport scala sie po nich miedzy sesjami.
"""

from .exp_averaging import (run_e3_averaging_domain, run_e4_ir_length)
from .exp_determinism import (run_e1, run_e1_checkpoint_boundary, run_e1_checkpoint_boundary_batch_a, run_e1_checkpoint_boundary_batch_b, run_e1_checkpoint_boundary_merge, run_e1_extended, run_p0)
from .exp_gpu import (run_gpu_memory_scale)
from .exp_materials import (run_materials_verify)
from .exp_noise_floor import (run_listener_height, run_noise_floor_orientation, run_noise_floor_remaining, run_noise_floor_scenes, run_signal_noise_recheck)
from .exp_rays import (run_e2_bias_orientation, run_e2_ray_bias, run_e2_rays_vs_renders, run_e2_thread_budget_confirm, run_e2_thread_effective_rays, run_e2_thread_estimator)

EXPERIMENTS = {
    "p0": run_p0,
    "e1": run_e1,
    "e1_extended": run_e1_extended,
    "e1_checkpoint_boundary": run_e1_checkpoint_boundary,
    "e1_checkpoint_boundary_batch_a": run_e1_checkpoint_boundary_batch_a,
    "e1_checkpoint_boundary_batch_b": run_e1_checkpoint_boundary_batch_b,
    "e1_checkpoint_boundary_merge": run_e1_checkpoint_boundary_merge,
    "e2_rays_vs_renders": run_e2_rays_vs_renders,
    "e2_ray_bias": run_e2_ray_bias,
    "e2_bias_orientation": run_e2_bias_orientation,
    "e2_thread_estimator": run_e2_thread_estimator,
    "e2_thread_effective_rays": run_e2_thread_effective_rays,
    "e2_thread_budget_confirm": run_e2_thread_budget_confirm,
    "e3_averaging_domain": run_e3_averaging_domain,
    "e4_ir_length": run_e4_ir_length,
    "listener_height": run_listener_height,
    "materials_verify": run_materials_verify,
    "signal_noise_recheck": run_signal_noise_recheck,
    "gpu_memory_scale": run_gpu_memory_scale,
    "noise_floor_scenes": run_noise_floor_scenes,
    "noise_floor_orientation": run_noise_floor_orientation,
    "noise_floor_remaining": run_noise_floor_remaining,
}
