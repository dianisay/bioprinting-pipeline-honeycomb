# Robot and printing status

This directory contains historical robot-related code and an offline
computational adapter. The repository integration milestone did **not** run
CoppeliaSim, command a physical robot, validate inverse kinematics or
collisions, or perform printing.

## Offline adapter

The bounded synthetic demonstration is documented in
[`simulation/README.md`](simulation/README.md) and can be generated with:

```powershell
python -m robot.simulation.observable_demo
```

It evaluates five software-only cases and writes records under
`results/coppelia_demo/`. The command does not start a simulator, robot,
deposition controller, or printer. Preview videos are visualization outputs,
not robot-camera recordings.

## Existing hardware-oriented components

Files such as `junto2.m`, `mc_bridge.py`, and related calibration/model assets
are retained as historical project components. Their presence does not mean
that the current wound-reconstruction pipeline has been integrated with them
or that their hardware behavior is verified. Do not treat stored example
parameters, coordinates, or controller logic as validated operating
instructions.

## Safety boundary

- Physical printing is disabled and not authorized.
- No CoppeliaSim scene or robot execution was used for the reported results.
- The final computational architecture is
  `STAGED_FILL_WITH_OPERATOR_FINAL_TARGET`.
- The operator-defined target is a required input, not a clinically validated
  target.
