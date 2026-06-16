---
name: fpga-vivado-systemverilog
description: Use this skill when writing, reviewing, simulating, synthesizing, or optimizing FPGA RTL in SystemVerilog using Xilinx Vivado 2024.1, including XSIM simulation, synthesis, timing, DSP Macro IP, Block Memory Generator IP, and FPGA resource-aware design.
---

# FPGA Vivado SystemVerilog Skill

## Environment

Assume the target FPGA toolchain is Xilinx Vivado.

Expected Vivado version:

```bash
vivado -version
# Vivado v2024.1 (64-bit)
# Tool Version Limit: 2024.05
```

Vivado may be available in the current terminal environment.

Before claiming Vivado verification, actually run the relevant Vivado command and inspect the generated logs/reports.

---

## Default HDL

Use SystemVerilog by default.

Prefer:

```systemverilog
logic
always_ff
always_comb
parameter
localparam
typedef enum logic
```

Avoid legacy Verilog style unless explicitly required.

Do not use:

```systemverilog
reg
wire
always @(posedge clk)
always @(*)
```

unless compatibility with old Verilog is explicitly requested.

---

## Naming Convention

### Clock and Reset

Use:

```systemverilog
aclk
aresetn
```

Assume:

- `aclk` is the primary clock.
- `aresetn` is active-low reset.
- Reset is synchronous by default.

Sequential logic must use:

```systemverilog
always_ff @(posedge aclk) begin
    if (!aresetn) begin
        ...
    end else begin
        ...
    end
end
```

Do not use asynchronous reset unless explicitly requested.

Do not use:

```systemverilog
always_ff @(posedge aclk or negedge aresetn)
always @(posedge clk)
always @(posedge clk or negedge rst_n)
```

unless specifically required.

---

### Port Naming

All module ports must use direction suffixes.

| Direction | Suffix |
|---|---|
| input | `_i` |
| output | `_o` |

Examples:

```systemverilog
input  logic        valid_i;
input  logic [31:0] data_i;

output logic        ready_o;
output logic [31:0] result_o;
```

Clock and reset are exceptions and should remain:

```systemverilog
input logic aclk;
input logic aresetn;
```

---

### Internal Signal Naming

Use:

| Signal Type | Suffix |
|---|---|
| register | `_r` |
| next-state / next-value | `_n` |
| combinational wire | `_w` |
| pipeline stage | `_s0_r`, `_s1_r`, `_s2_r` |
| valid | `_valid` |
| ready | `_ready` |

Examples:

```systemverilog
logic [31:0] acc_r;
logic [31:0] acc_n;
logic [15:0] data_s0_r;
logic [15:0] data_s1_r;
logic        mul_valid_w;
```

Use lowercase snake case for modules and signals.

Use uppercase snake case for parameters.

```systemverilog
parameter DATA_WIDTH = 32;
parameter DEPTH      = 1024;
```

Avoid vague names:

```systemverilog
tmp
sig
data2
aaa
```

---

## RTL Coding Rules

All sequential logic must use:

```systemverilog
always_ff @(posedge aclk)
```

All combinational logic must use:

```systemverilog
always_comb
```

Inside `always_ff`:

- Use only nonblocking assignment `<=`.
- Do not use blocking assignment `=`.
- Reset all important registers.
- Use synchronous active-low reset with `aresetn`.

Inside `always_comb`:

- Use blocking assignment `=`.
- Assign default values first.
- Avoid inferred latches.
- Cover all `case` branches.
- Use `default`.

Example:

```systemverilog
always_comb begin
    state_n = state_r;

    case (state_r)
        IDLE: begin
            if (start_i) begin
                state_n = RUN;
            end
        end

        RUN: begin
            if (done_w) begin
                state_n = DONE;
            end
        end

        DONE: begin
            state_n = IDLE;
        end

        default: begin
            state_n = IDLE;
        end
    endcase
end
```

---

## Synthesizable RTL Rules

Do not use non-synthesizable constructs in RTL:

```systemverilog
#10
initial
fork/join
randomize
class
queue
dynamic array
$display
$finish
```

These may be used only in testbenches.

Use explicit bit widths.

Avoid unsized constants when width matters.

Bad:

```systemverilog
acc_r <= acc_r + 1;
```

Better:

```systemverilog
acc_r <= acc_r + {{(ACC_WIDTH-1){1'b0}}, 1'b1};
```

or:

```systemverilog
acc_r <= acc_r + ACC_WIDTH'(1);
```

Clearly define signedness.

```systemverilog
logic signed [DATA_WIDTH-1:0] a_i;
logic signed [DATA_WIDTH-1:0] b_i;
```

---

## Interface Rule

For streaming datapaths, prefer valid/ready style.

Use:

```systemverilog
valid_i
ready_o
data_i

valid_o
ready_i
data_o
```

For AXI-Stream-like interfaces, use Xilinx-style names when appropriate:

```systemverilog
s_axis_tvalid_i
s_axis_tready_o
s_axis_tdata_i

m_axis_tvalid_o
m_axis_tready_i
m_axis_tdata_o
```

Clearly state:

- latency
- throughput
- whether backpressure is supported
- whether input can be accepted every cycle

---

## Timing and Pipeline Rules

Assume timing matters.

Avoid large combinational paths.

Pipeline:

- multipliers
- adder trees
- comparators
- reduction logic
- wide mux paths
- memory output paths

When designing arithmetic datapaths, always state:

- pipeline depth
- latency
- initiation interval
- expected throughput

For reduction or accumulation, prefer balanced/pipelined structures over long combinational chains.

---

## Vivado Simulation Rule

Use Vivado XSIM for simulation when Vivado is available.

Preferred XSIM CLI flow:

```bash
xvlog -sv rtl/*.sv tb/*.sv
xelab tb_top -s tb_top_sim
xsim tb_top_sim -runall
```

If project/IP dependencies exist, use Vivado batch mode:

```bash
vivado -mode batch -source scripts/run_sim.tcl
```

Only claim simulation passed if XSIM actually ran and the log shows PASS.

Testbenches should include:

- clock generation using `aclk`
- reset sequence using `aresetn`
- directed tests
- edge cases
- self-checking comparisons
- final PASS/FAIL message

Example testbench clock/reset:

```systemverilog
logic aclk;
logic aresetn;

initial begin
    aclk = 1'b0;
    forever #5 aclk = ~aclk;
end

initial begin
    aresetn = 1'b0;
    repeat (5) @(posedge aclk);
    aresetn = 1'b1;
end
```

---

## Vivado Synthesis Rule

Use Vivado batch mode for synthesis when possible.

```bash
vivado -mode batch -source scripts/run_synth.tcl
```

Do not claim:

- synthesis passed
- timing met
- implementation succeeded
- Vivado verified

unless Vivado actually ran and the reports were checked.

Check:

```text
vivado.log
reports/utilization_synth.rpt
reports/timing_synth.rpt
```

---

## Vivado TCL Template

Generate Vivado TCL scripts when synthesis or implementation is relevant.

```tcl
set project_name "fpga_auto_project"
set project_dir  "./vivado_proj"
set part_name    "<FPGA_PART_NAME>"
set top_name     "<TOP_MODULE_NAME>"

file mkdir reports

create_project $project_name $project_dir -part $part_name -force

add_files [glob ./rtl/*.sv]
add_files -fileset sim_1 [glob ./tb/*.sv]

set_property top $top_name [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1
wait_on_run synth_1

open_run synth_1
report_utilization -file reports/utilization_synth.rpt
report_timing_summary -file reports/timing_synth.rpt

exit
```

If implementation is required:

```tcl
launch_runs impl_1
wait_on_run impl_1

open_run impl_1
report_utilization -file reports/utilization_impl.rpt
report_timing_summary -file reports/timing_impl.rpt
```

---

## DSP48 Rule

For arithmetic-heavy modules, explicitly consider DSP48 usage.

Use DSPs for:

- multiplication
- MAC
- dot product
- matrix multiplication
- convolution
- low-bit accelerator datapaths

Simple RTL multiplication may infer DSPs:

```systemverilog
always_ff @(posedge aclk) begin
    if (!aresetn) begin
        product_r <= '0;
    end else begin
        product_r <= a_i * b_i;
    end
end
```

MAC example:

```systemverilog
always_ff @(posedge aclk) begin
    if (!aresetn) begin
        acc_r <= '0;
    end else if (valid_i) begin
        acc_r <= acc_r + a_i * b_i;
    end
end
```

But do not assume DSP inference succeeded.

Check Vivado utilization report for:

```text
DSP
DSP48E1
DSP48E2
DSP48E
```

Only claim DSP usage if the report confirms it.

---

## DSP Macro IP Rule

Vivado provides DSP Macro IP through IP Catalog.

Use DSP Macro IP when deterministic DSP usage or controlled DSP48 structure is required.

Use DSP Macro IP for:

- explicit multiplier
- MAC
- pipelined DSP datapath
- controlled DSP48 mapping
- timing-sensitive arithmetic

Before using DSP Macro IP, define:

- input width
- signed/unsigned type
- output width
- accumulator width
- pipeline stages
- clock enable behavior
- reset behavior
- latency
- truncation/saturation behavior

Generate TCL when DSP Macro IP is required.

Template:

```tcl
create_ip -name dsp_macro \
    -vendor xilinx.com \
    -library ip \
    -module_name dsp_mac_0

set_property -dict [list \
    CONFIG.Component_Name {dsp_mac_0} \
] [get_ips dsp_mac_0]

generate_target all [get_ips dsp_mac_0]
```

Do not invent exact DSP Macro CONFIG fields unless verified in Vivado 2024.1.

If exact options are unknown, add TODO comments and instruct the user to export the IP TCL from Vivado.

---

## BRAM Rule

Use BRAM for large memories.

Use BRAM for:

- weight buffers
- activation buffers
- codebooks
- lookup tables
- scale/exponent tables
- large FIFOs
- tile buffers

Avoid large FF arrays.

For simple portable memories, RTL inference is acceptable.

For deterministic BRAM behavior, use Block Memory Generator IP.

---

## Block Memory Generator IP Rule

Vivado provides Block Memory Generator IP through IP Catalog.

Use Block Memory Generator IP when exact memory behavior is required:

- single-port RAM
- simple dual-port RAM
- true dual-port RAM
- ROM
- initialized memory
- `.coe` / `.mem` file initialization
- fixed read latency
- controlled write mode

Before generating BRAM IP, define:

- memory type
- data width
- depth
- address width
- read latency
- write mode
- byte enable usage
- initialization file
- single clock or independent clocks
- reset behavior
- enable behavior

Generate TCL when BRAM IP is required.

Template:

```tcl
create_ip -name blk_mem_gen \
    -vendor xilinx.com \
    -library ip \
    -module_name bram_0

set_property -dict [list \
    CONFIG.Component_Name {bram_0} \
] [get_ips bram_0]

generate_target all [get_ips bram_0]
```

Do not invent exact Block Memory Generator CONFIG fields unless verified in Vivado 2024.1.

Check utilization report for:

```text
Block RAM Tile
RAMB18
RAMB36
BRAM
```

Only claim BRAM usage if the Vivado report confirms it.

---

## Accelerator Design Rules

For FPGA accelerator design:

- separate memory interface from compute datapath
- explicitly define operand bit width
- explicitly define signedness
- track accumulator width
- define rounding behavior
- define saturation behavior
- define truncation behavior
- estimate latency and throughput
- estimate DSP usage
- estimate BRAM usage
- estimate memory bandwidth

For BFP or quantized datapaths:

- separate exponent logic from mantissa MAC datapath
- define group size
- define shared exponent width
- define mantissa width
- define accumulator width
- define scale/exponent update timing
- avoid silently truncating results

---

## Verification Output Rule

At the end of every FPGA task, summarize:

```text
RTL files changed:
Testbench files changed:
Vivado simulation command:
Simulation result:
Vivado synthesis command:
Synthesis result:
Timing result:
DSP usage:
BRAM usage:
Important warnings:
Known limitations:
```

If Vivado was not run, explicitly say:

```text
Vivado was not run. The design is code-reviewed only.
```

If Vivado was run but timing was not checked, explicitly say:

```text
Simulation/synthesis was run, but timing was not fully verified.
```

---

## Review Checklist

When reviewing RTL, check:

- SystemVerilog style
- `aclk` / `aresetn` usage
- `_i` / `_o` port suffixes
- `always_ff @(posedge aclk)` usage
- `always_comb` usage
- latch inference risk
- blocking/nonblocking assignment misuse
- width mismatch
- signed/unsigned mismatch
- reset behavior
- valid/ready handshake correctness
- DSP inference
- BRAM inference
- timing-critical paths
- testbench coverage
- Vivado report warnings

Prefer minimal, surgical fixes over rewriting the entire design.
