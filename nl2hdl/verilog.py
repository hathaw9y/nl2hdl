from __future__ import annotations

from pathlib import Path

from .config import AgentConfig
from .quant import QuantizedLayer, QuantizedModel


def _sv_signed(value: int, width: int) -> str:
    value = int(value)
    if value < 0:
        return f"-{width}'sd{abs(value)}"
    return f"{width}'sd{value}"


def _dense_layer_sv(layer: QuantizedLayer, module_name: str, pe_count: int) -> str:
    pe_count = max(1, int(pe_count))
    weight_cases: list[str] = []
    for out_idx in range(layer.output_size):
        inner = [
            f"                {in_idx}: weight_at = {_sv_signed(layer.weights_i8[out_idx, in_idx], 8)};"
            for in_idx in range(layer.input_size)
        ]
        weight_cases.append(
            "\n".join(
                [
                    f"            {out_idx}: begin",
                    "                case (in_idx)",
                    *inner,
                    "                    default: weight_at = 8'sd0;",
                    "                endcase",
                    "            end",
                ]
            )
        )

    bias_cases = [
        f"            {out_idx}: bias_at = {_sv_signed(layer.bias_i32[out_idx], 32)};"
        for out_idx in range(layer.output_size)
    ]
    pe_terms = [
        f"""            if ((in_base_r + {pe_idx}) < IN_SIZE) begin
                input_w = data_i[(in_base_r + {pe_idx})*8 +: 8];
                pe_sum_w = pe_sum_w + (input_w * weight_at(out_idx_r, in_base_r + {pe_idx}));
            end"""
        for pe_idx in range(pe_count)
    ]
    relu_block = (
        "        if (requant_w < 0) begin\n"
        "            requant_w = 0;\n"
        "        end\n"
        if layer.activation == "relu"
        else ""
    )

    return f"""`timescale 1ns/1ps

module {module_name} #(
    parameter int IN_SIZE  = {layer.input_size},
    parameter int OUT_SIZE = {layer.output_size},
    parameter int PE_COUNT = {pe_count}
) (
    input  logic                         aclk,
    input  logic                         aresetn,
    input  logic                         start_i,
    input  logic signed [IN_SIZE*8-1:0]  data_i,
    output logic                         done_o,
    output logic signed [OUT_SIZE*8-1:0] data_o
);
    localparam int REQUANT_MULT  = {layer.requant_mult};
    localparam int REQUANT_SHIFT = {layer.requant_shift};

    typedef enum logic [1:0] {{IDLE, RUN, DONE}} state_t;

    state_t state_r;
    state_t state_n;
    int out_idx_r;
    int in_base_r;
    logic signed [31:0] acc_r;
    logic signed [OUT_SIZE*8-1:0] data_r;
    logic signed [31:0] pe_sum_w;
    logic signed [31:0] requant_w;
    logic signed [7:0] input_w;
    logic signed [7:0] final_output_w;
    logic last_chunk_w;
    logic last_output_w;

    assign data_o = data_r;
    assign done_o = (state_r == DONE);
    assign last_chunk_w = ((in_base_r + PE_COUNT) >= IN_SIZE);
    assign last_output_w = (out_idx_r == (OUT_SIZE - 1));

    function automatic logic signed [7:0] weight_at(input int out_idx, input int in_idx);
        begin
            weight_at = 8'sd0;
            case (out_idx)
{chr(10).join(weight_cases)}
                default: weight_at = 8'sd0;
            endcase
        end
    endfunction

    function automatic logic signed [31:0] bias_at(input int out_idx);
        begin
            bias_at = 32'sd0;
            case (out_idx)
{chr(10).join(bias_cases)}
                default: bias_at = 32'sd0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] clamp_i8(input logic signed [31:0] value_i);
        begin
            if (value_i > 32'sd127) begin
                clamp_i8 = 8'sd127;
            end else if (value_i < -32'sd128) begin
                clamp_i8 = -8'sd128;
            end else begin
                clamp_i8 = value_i[7:0];
            end
        end
    endfunction

    always_comb begin
        pe_sum_w = 32'sd0;
        input_w = 8'sd0;
{chr(10).join(pe_terms)}
    end

    always_comb begin
        requant_w = ((acc_r + pe_sum_w) * REQUANT_MULT) >>> REQUANT_SHIFT;
{relu_block}        final_output_w = clamp_i8(requant_w);
    end

    always_comb begin
        state_n = state_r;
        case (state_r)
            IDLE: begin
                if (start_i) begin
                    state_n = RUN;
                end
            end
            RUN: begin
                if (last_chunk_w && last_output_w) begin
                    state_n = DONE;
                end
            end
            DONE: begin
                if (!start_i) begin
                    state_n = IDLE;
                end
            end
            default: begin
                state_n = IDLE;
            end
        endcase
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            out_idx_r <= 0;
            in_base_r <= 0;
            acc_r <= 32'sd0;
            data_r <= '0;
        end else begin
            state_r <= state_n;
            case (state_r)
                IDLE: begin
                    if (start_i) begin
                        out_idx_r <= 0;
                        in_base_r <= 0;
                        acc_r <= bias_at(0);
                        data_r <= '0;
                    end
                end
                RUN: begin
                    if (last_chunk_w) begin
                        data_r[out_idx_r*8 +: 8] <= final_output_w;
                        if (!last_output_w) begin
                            out_idx_r <= out_idx_r + 1;
                            in_base_r <= 0;
                            acc_r <= bias_at(out_idx_r + 1);
                        end
                    end else begin
                        in_base_r <= in_base_r + PE_COUNT;
                        acc_r <= acc_r + pe_sum_w;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        out_idx_r <= 0;
                        in_base_r <= 0;
                        acc_r <= 32'sd0;
                    end
                end
                default: begin
                end
            endcase
        end
    end
endmodule
"""


def _top_sv(qmodel: QuantizedModel, pe_count: int) -> str:
    states = ["IDLE"]
    for idx, _ in enumerate(qmodel.layers):
        states.extend([f"START_LAYER_{idx}", f"WAIT_LAYER_{idx}"])
    states.append("DONE")

    state_decl = ", ".join(states)
    regs = [f"    logic signed [{qmodel.input_size}*8-1:0] layer_input_r;"]
    layer_wires = [
        f"    logic signed [{layer.output_size}*8-1:0] layer_{idx}_out_w;"
        for idx, layer in enumerate(qmodel.layers)
    ]
    start_wires = [f"    logic layer_{idx}_start_w;" for idx, _ in enumerate(qmodel.layers)]
    done_wires = [f"    logic layer_{idx}_done_w;" for idx, _ in enumerate(qmodel.layers)]

    instances = []
    for idx, _layer in enumerate(qmodel.layers):
        source = "layer_input_r" if idx == 0 else f"layer_{idx - 1}_out_w"
        instances.append(
            f"""    assign layer_{idx}_start_w = (state_r == START_LAYER_{idx});

    dense_layer_{idx} #(
        .PE_COUNT({max(1, int(pe_count))})
    ) u_dense_layer_{idx} (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(layer_{idx}_start_w),
        .data_i({source}),
        .done_o(layer_{idx}_done_w),
        .data_o(layer_{idx}_out_w)
    );"""
        )

    comb_cases = [
        "            IDLE: begin",
        "                if (start_i) begin",
        "                    state_n = START_LAYER_0;",
        "                end",
        "            end",
    ]
    for idx, _ in enumerate(qmodel.layers):
        next_state = f"START_LAYER_{idx + 1}" if idx + 1 < len(qmodel.layers) else "DONE"
        comb_cases.extend(
            [
                f"            START_LAYER_{idx}: begin",
                f"                state_n = WAIT_LAYER_{idx};",
                "            end",
                f"            WAIT_LAYER_{idx}: begin",
                f"                if (layer_{idx}_done_w) begin",
                f"                    state_n = {next_state};",
                "                end",
                "            end",
            ]
        )
    comb_cases.extend(
        [
            "            DONE: begin",
            "                if (!start_i) begin",
            "                    state_n = IDLE;",
            "                end",
            "            end",
        ]
    )

    return f"""`timescale 1ns/1ps

module model_top (
    input  logic                       aclk,
    input  logic                       aresetn,
    input  logic                       start_i,
    input  logic signed [{qmodel.input_size}*8-1:0]  input_vec_i,
    output logic                       done_o,
    output logic signed [{qmodel.output_size}*8-1:0] output_vec_o
);
    typedef enum logic [7:0] {{{state_decl}}} state_t;

    state_t state_r;
    state_t state_n;
{chr(10).join(regs)}
{chr(10).join(start_wires)}
{chr(10).join(done_wires)}
{chr(10).join(layer_wires)}

{chr(10).join(instances)}

    assign done_o = (state_r == DONE);
    assign output_vec_o = layer_{len(qmodel.layers) - 1}_out_w;

    always_comb begin
        state_n = state_r;
        case (state_r)
{chr(10).join(comb_cases)}
            default: begin
                state_n = IDLE;
            end
        endcase
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            layer_input_r <= '0;
        end else begin
            state_r <= state_n;
            case (state_r)
                IDLE: begin
                    if (start_i) begin
                        layer_input_r <= input_vec_i;
                    end
                end
                default: begin
                end
            endcase
        end
    end
endmodule
"""


def _testbench_sv(qmodel: QuantizedModel, tolerance: int) -> str:
    input_assigns = [
        f"        input_vec[{idx}*8 +: 8] = {_sv_signed(int(value), 8)};"
        for idx, value in enumerate(qmodel.input_i8.reshape(-1))
    ]
    checks = []
    for idx, expected in enumerate(qmodel.expected_i8.reshape(-1)):
        checks.append(
            f"""        observed = $signed(output_vec[{idx}*8 +: 8]);
        expected = {int(expected)};
        diff = observed - expected;
        if (diff < 0) begin
            diff = -diff;
        end
        if (diff > {tolerance}) begin
            $display("FAIL output[{idx}] observed=%0d expected=%0d diff=%0d", observed, expected, diff);
            $fatal;
        end"""
        )
    return f"""`timescale 1ns/1ps

module tb_model_top;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic signed [{qmodel.input_size}*8-1:0] input_vec;
    logic done_o;
    logic signed [{qmodel.output_size}*8-1:0] output_vec;
    integer observed;
    integer expected;
    integer diff;

    model_top dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .input_vec_i(input_vec),
        .done_o(done_o),
        .output_vec_o(output_vec)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        input_vec = '0;
{chr(10).join(input_assigns)}
        repeat (3) @(posedge aclk);
        aresetn = 1'b1;
        @(posedge aclk);
        start_i = 1'b1;
        @(posedge done_o);
{chr(10).join(checks)}
        $display("PASS nl2hdl integer simulation");
        start_i = 1'b0;
        repeat (2) @(posedge aclk);
        $finish;
    end
endmodule
"""


def _vivado_tcl(config: AgentConfig, qmodel: QuantizedModel) -> str:
    layer_reads = "\n".join([f"read_verilog -sv dense_layer_{idx}.sv" for idx, _ in enumerate(qmodel.layers)])
    period = 1000.0 / config.hardware.target_clock_mhz
    return f"""set_part {config.hardware.fpga_part}
read_verilog -sv model_top.sv
{layer_reads}
synth_design -top model_top -part {config.hardware.fpga_part}
create_clock -period {period:.3f} [get_ports aclk]
report_utilization -file utilization.rpt
report_timing_summary -file timing_summary.rpt
write_checkpoint -force post_synth.dcp
"""


def emit_systemverilog(qmodel: QuantizedModel, config: AgentConfig, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, layer in enumerate(qmodel.layers):
        path = out_dir / f"dense_layer_{idx}.sv"
        path.write_text(_dense_layer_sv(layer, f"dense_layer_{idx}", config.design.pe_count), encoding="utf-8")
        paths.append(path)
    top = out_dir / "model_top.sv"
    top.write_text(_top_sv(qmodel, config.design.pe_count), encoding="utf-8")
    paths.append(top)
    tb = out_dir / "tb_model_top.sv"
    tb.write_text(_testbench_sv(qmodel, config.verification.tolerance_lsb), encoding="utf-8")
    paths.append(tb)
    tcl = out_dir / "vivado_synth.tcl"
    tcl.write_text(_vivado_tcl(config, qmodel), encoding="utf-8")
    paths.append(tcl)
    return paths
