# V8 failure archive

Every rejected candidate remains a negative or diagnostic result. None is
relabelled as a pass, and confirmation data was not accessed for any candidate
listed here.

- `V8.2_seed300000`: valid Qiskit result for a direct-addressed parallel delay
  architecture; rejected from final consideration because it lacked physical
  shift transport.
- `V8.4_seed300000`: valid full-length physical-FIFO HH result; rejected before
  corner completion because the held-out nonlinear-combination feature family
  was incomplete.
- `V8.5_seed300000`: invalid-as-exact HH diagnostic. An entangled control was
  reset in the `past*` route, causing MPS reset trajectories to vary with the
  simulator seed (maximum reproduced feature difference 0.0390625). The three
  uncheckpointed comparison corners were terminated and the candidate rejected.
