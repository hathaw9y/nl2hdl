# HDL Module Interface Contract

All HDL sub-agents must implement kernels against this contract unless their task
explicitly says otherwise.

## Clock and Reset

- `input logic aclk`
- `input logic aresetn`
- reset is synchronous, active-low

## Kernel Handshake

Use a simple command/done protocol for kernel milestones:

```systemverilog
input  logic start_i;
output logic done_o;
```

Rules:

- `start_i` is sampled when the module is idle.
- `done_o` remains high until `start_i` is deasserted.
- Inputs must remain stable while the kernel is busy.
- Outputs must be stable when `done_o` is high.

## Data Naming

- Scalar/vector inputs end in `_i`.
- Scalar/vector outputs end in `_o`.
- Packed vectors use little element order: element `idx` is `[idx*WIDTH +: WIDTH]`.

## Required Artifacts Per Module

Each HDL sub-agent must produce:

- SystemVerilog module generator or concrete `.sv`.
- Testbench or generated testbench.
- Python/NumPy golden vector source or report.
- `kernel_report.json` with simulation and synthesis evidence.
- `module_ooc_synthesis_report.json` for real datapath modules, or an explicit
  fixture/control-scaffold waiver inside `kernel_report.json`.
- Vivado `timing_summary.rpt` and `utilization.rpt` when synthesis is requested.

`module_ooc_synthesis_report.json` must include the hardware identity it was
run against: FPGA part, target clock, memory data width, configured `max_*`
budgets, and configured `device_*` resource inventory. If any of these fields
changes, parent orchestration treats the report as stale.

## Pass Criteria

A module is not complete until:

- Python/unit tests pass.
- RTL simulation passes.
- Vivado timing parser reports setup, hold, and pulse-width checks as passing when synthesis is enabled.
- Real datapath modules have module-level out-of-context synthesis evidence with
  LUT, DSP, BRAM, URAM, FF, I/O, selected tuning knobs, and a resource
  assessment before any integration agent consumes them.
- Any failure pattern is captured in a Skill if the HDL sub-agent cannot resolve it.

## Integration Layers

After module kernels pass:

- Module-level synthesis/tuning gates select the child configuration that
  integration agents must consume.
- Layer FSM sub-agent composes kernels for one decoder layer.
- Top FSM sub-agent composes layer calls for token prefill/decode scheduling.
- Parent agent only verifies and coordinates; it does not hand-write HDL.
