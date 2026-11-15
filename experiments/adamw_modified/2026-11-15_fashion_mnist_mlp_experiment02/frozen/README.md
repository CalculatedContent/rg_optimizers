# Exact reference, not a new training launch

`optimizer_reference.py` is a byte-identical copy of Experiment 01's frozen training source. SHA256: `1bfd7032e00ad87d729670197925fb40e19bc802a8eb716db2571e56b538d3d8`. `Parent_Protocol.json` preserves its original MNIST protocol and is provenance, not the Fashion-MNIST configuration.

Use the exact MLP, optimizer, schedule, ordering, magnitude-graft and fit functions when implementing the isolated Fashion-MNIST execution adapter. Keep Adam direction/moments/bias updates and each run's own SGDM magnitude history. Preserve the denominator floor and matched decay.

Do not execute this reference directly: its historical protocol/data-freezing and default sweep functions target MNIST. Fashion-MNIST's active plan is `../protocol/Protocol_Plan.json`. Freeze the new adapter and prepared-data manifest after preflight and before starting training.
