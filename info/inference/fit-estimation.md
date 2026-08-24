# ICN model fit assessment

ICN estimates whether a model can be served on this host. It is an estimate, not a measurement:
inference happens inside an EIM container, and a container cannot be asked from outside how much
memory it would need before it is running.

The estimate comes from six numbers per model — total and active parameters, layer count, key-value
head count, head dimension, and trained context — plus the model's measured on-disk weight size.
Geometry is fetched from each model's Hugging Face `config.json` and the size from its safetensors
index; neither is guessed, because a wrong layer count produces an estimate that is confidently
wrong. Weights are charged at their measured size rather than derived from the parameter count,
since a published checkpoint is not always bf16.

A result is `Fits` or `DoesNotFit`, and either way it reports the four buckets separately: model
weights, key-value cache, activations, and runtime overhead. For a mixture of experts every expert
counts toward weights, because all of them stay resident, while only the active parameters bear on
throughput.

The estimate is not what limits the engine. That is `--gpu-memory-utilization`, misleadingly named
on the CPU backend, where it is a fraction of the memory the *container* may use rather than of a
NUMA node — and each tensor-parallel worker claims that fraction independently. vLLM also compares
the request against memory currently available rather than against the ceiling, so the ceiling has
to exceed the workers' share by whatever is already resident.

ICN therefore derives the container's memory ceiling and that fraction together from one estimate,
because they multiply. Setting either from the estimate alone produces a container that refuses to
start or runs out of memory part-way through loading; both mistakes were made before they were
measured.

On a host with plenty of memory the assessment is informational far more often than it is a gate. Its
value is in what it shows the user before a multi-gigabyte download begins, so the buckets are
reported honestly rather than collapsed into a single total.

Load is still the final boundary. The container is given a hard memory ceiling derived from the
estimate, so an underestimate produces a fast, legible out-of-memory failure with real byte figures
instead of degrading the whole host.
