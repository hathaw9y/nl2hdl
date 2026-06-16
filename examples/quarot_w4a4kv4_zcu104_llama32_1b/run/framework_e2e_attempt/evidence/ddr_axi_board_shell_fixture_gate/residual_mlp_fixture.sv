`timescale 1ns/1ps

module residual_mlp_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int ACC_WIDTH = 32,
    parameter int STATUS_WIDTH = 96
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    input  logic signed [VALUES*ELEM_WIDTH-1:0]       hidden_input_i,
    input  logic signed [VALUES*ELEM_WIDTH-1:0]       attention_output_i,
    output logic                                      done_o,
    output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    localparam int TRACE_WIDTH = 80;

    typedef enum logic [3:0] {
        IDLE           = 4'd0,
        RESIDUAL0      = 4'd1,
        GATE_UP        = 4'd2,
        SWIGLU_SIGMOID = 4'd3,
        SWIGLU_SILU    = 4'd4,
        SWIGLU         = 4'd5,
        DOWN           = 4'd6,
        RESIDUAL1      = 4'd7,
        DONE           = 4'd8
    } state_t;

    state_t state_r;
    logic done_r;
    logic [TRACE_WIDTH-1:0] trace_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] hidden_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] attention_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] residual0_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] gate_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] up_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] sigmoid_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] silu_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] swiglu_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] down_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    int idx;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = {16'h4d52, trace_r};

    function automatic logic signed [ELEM_WIDTH-1:0] lane(
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input int lane_i
    );
        begin
            lane = vec_i[lane_i*ELEM_WIDTH +: ELEM_WIDTH];
        end
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] add_lanes(
        input logic signed [ELEM_WIDTH-1:0] a_i,
        input logic signed [ELEM_WIDTH-1:0] b_i
    );
        logic signed [ACC_WIDTH-1:0] a_ext;
        logic signed [ACC_WIDTH-1:0] b_ext;
        begin
            a_ext = {{(ACC_WIDTH-ELEM_WIDTH){a_i[ELEM_WIDTH-1]}}, a_i};
            b_ext = {{(ACC_WIDTH-ELEM_WIDTH){b_i[ELEM_WIDTH-1]}}, b_i};
            add_lanes = a_ext + b_ext;
        end
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] sext_lane(
        input logic signed [ELEM_WIDTH-1:0] value_i
    );
        begin
            sext_lane = {{(ACC_WIDTH-ELEM_WIDTH){value_i[ELEM_WIDTH-1]}}, value_i};
        end
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] mul_coeff(
        input logic signed [ELEM_WIDTH-1:0] value_i,
        input logic signed [ACC_WIDTH-1:0] coeff_i
    );
        logic signed [ACC_WIDTH-1:0] value_ext;
        logic signed [2*ACC_WIDTH-1:0] product;
        begin
            value_ext = {{(ACC_WIDTH-ELEM_WIDTH){value_i[ELEM_WIDTH-1]}}, value_i};
            product = value_ext * coeff_i;
            mul_coeff = product[ACC_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [ELEM_WIDTH-1:0] clip16(
        input logic signed [ACC_WIDTH-1:0] value_i
    );
        begin
            if (value_i > 32'sd32767) begin
                clip16 = 16'sd32767;
            end else if (value_i < -32'sd32768) begin
                clip16 = -16'sd32768;
            end else begin
                clip16 = value_i[ELEM_WIDTH-1:0];
            end
        end
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] gate_acc(
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input int row_i
    );
        begin
            case (row_i)
                0: gate_acc = sext_lane(lane(vec_i, 0)) - sext_lane(lane(vec_i, 1)) + (sext_lane(lane(vec_i, 3)) <<< 1);
                1: gate_acc = (sext_lane(lane(vec_i, 1)) <<< 1) - sext_lane(lane(vec_i, 2)) + sext_lane(lane(vec_i, 3));
                2: gate_acc = -(sext_lane(lane(vec_i, 0)) <<< 1) + sext_lane(lane(vec_i, 2)) + sext_lane(lane(vec_i, 3));
                default: gate_acc = sext_lane(lane(vec_i, 0)) + sext_lane(lane(vec_i, 1)) + sext_lane(lane(vec_i, 2)) - sext_lane(lane(vec_i, 3));
            endcase
        end
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] up_acc(
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input int row_i
    );
        begin
            case (row_i)
                0: up_acc = (sext_lane(lane(vec_i, 0)) <<< 1) - sext_lane(lane(vec_i, 2)) + sext_lane(lane(vec_i, 3));
                1: up_acc = -sext_lane(lane(vec_i, 0)) + sext_lane(lane(vec_i, 1)) + (sext_lane(lane(vec_i, 2)) <<< 1);
                2: up_acc = -(sext_lane(lane(vec_i, 1)) <<< 1) + sext_lane(lane(vec_i, 2)) + sext_lane(lane(vec_i, 3));
                default: up_acc = sext_lane(lane(vec_i, 0)) + sext_lane(lane(vec_i, 1)) - (sext_lane(lane(vec_i, 3)) <<< 1);
            endcase
        end
    endfunction

    function automatic logic signed [ACC_WIDTH-1:0] down_acc(
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input int row_i
    );
        begin
            case (row_i)
                0: down_acc = sext_lane(lane(vec_i, 0)) + sext_lane(lane(vec_i, 2)) - sext_lane(lane(vec_i, 3));
                1: down_acc = -sext_lane(lane(vec_i, 0)) + sext_lane(lane(vec_i, 1)) + sext_lane(lane(vec_i, 3));
                2: down_acc = (sext_lane(lane(vec_i, 0)) <<< 1) - sext_lane(lane(vec_i, 1)) + sext_lane(lane(vec_i, 2));
                default: down_acc = sext_lane(lane(vec_i, 1)) - (sext_lane(lane(vec_i, 2)) <<< 1) + sext_lane(lane(vec_i, 3));
            endcase
        end
    endfunction

    function automatic logic signed [ELEM_WIDTH-1:0] sigmoid_approx_value(
        input logic signed [ELEM_WIDTH-1:0] gate_i
    );
        logic signed [ACC_WIDTH-1:0] gate_ext;
        logic signed [ACC_WIDTH-1:0] sigmoid_ext;
        begin
            gate_ext = {{(ACC_WIDTH-ELEM_WIDTH){gate_i[ELEM_WIDTH-1]}}, gate_i};
            if ((gate_ext + 32'sd8) < 32'sd0) begin
                sigmoid_ext = 32'sd0;
            end else if ((gate_ext + 32'sd8) > 32'sd16) begin
                sigmoid_ext = 32'sd16;
            end else begin
                sigmoid_ext = gate_ext + 32'sd8;
            end
            sigmoid_approx_value = clip16(sigmoid_ext);
        end
    endfunction

    function automatic logic signed [ELEM_WIDTH-1:0] silu_approx_value(
        input logic signed [ELEM_WIDTH-1:0] gate_i,
        input logic signed [ELEM_WIDTH-1:0] sigmoid_i
    );
        logic signed [ACC_WIDTH-1:0] gate_ext;
        logic signed [ACC_WIDTH-1:0] sigmoid_ext;
        logic signed [ACC_WIDTH-1:0] silu_trunc;
        logic signed [2*ACC_WIDTH-1:0] silu_product;
        begin
            gate_ext = {{(ACC_WIDTH-ELEM_WIDTH){gate_i[ELEM_WIDTH-1]}}, gate_i};
            sigmoid_ext = {{(ACC_WIDTH-ELEM_WIDTH){sigmoid_i[ELEM_WIDTH-1]}}, sigmoid_i};
            silu_product = gate_ext * sigmoid_ext;
            silu_trunc = silu_product[ACC_WIDTH-1:0];
            silu_approx_value = clip16(silu_trunc >>> 4);
        end
    endfunction

    function automatic logic signed [ELEM_WIDTH-1:0] swiglu_product_value(
        input logic signed [ELEM_WIDTH-1:0] silu_i,
        input logic signed [ELEM_WIDTH-1:0] up_i
    );
        logic signed [ACC_WIDTH-1:0] silu_ext;
        logic signed [ACC_WIDTH-1:0] up_ext;
        logic signed [ACC_WIDTH-1:0] swiglu_trunc;
        logic signed [2*ACC_WIDTH-1:0] swiglu_product;
        begin
            silu_ext = {{(ACC_WIDTH-ELEM_WIDTH){silu_i[ELEM_WIDTH-1]}}, silu_i};
            up_ext = {{(ACC_WIDTH-ELEM_WIDTH){up_i[ELEM_WIDTH-1]}}, up_i};
            swiglu_product = silu_ext * up_ext;
            swiglu_trunc = swiglu_product[ACC_WIDTH-1:0];
            swiglu_product_value = clip16(swiglu_trunc >>> 3);
        end
    endfunction

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            trace_r <= '0;
            hidden_r <= '0;
            attention_r <= '0;
            residual0_r <= '0;
            gate_r <= '0;
            up_r <= '0;
            sigmoid_r <= '0;
            silu_r <= '0;
            swiglu_r <= '0;
            down_r <= '0;
            final_output_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    if (start_i) begin
                        hidden_r <= hidden_input_i;
                        attention_r <= attention_output_i;
                        trace_r <= '0;
                        trace_r[0*8 +: 8] <= 8'h11;
                        state_r <= RESIDUAL0;
                    end
                end
                RESIDUAL0: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        residual0_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= clip16(add_lanes(lane(hidden_r, idx), lane(attention_r, idx)));
                    end
                    trace_r[1*8 +: 8] <= 8'h12;
                    trace_r[2*8 +: 8] <= 8'h21;
                    state_r <= GATE_UP;
                end
                GATE_UP: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        gate_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= clip16(gate_acc(residual0_r, idx));
                        up_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= clip16(up_acc(residual0_r, idx));
                    end
                    trace_r[3*8 +: 8] <= 8'h22;
                    trace_r[4*8 +: 8] <= 8'h31;
                    state_r <= SWIGLU_SIGMOID;
                end
                SWIGLU_SIGMOID: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        sigmoid_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= sigmoid_approx_value(lane(gate_r, idx));
                    end
                    state_r <= SWIGLU_SILU;
                end
                SWIGLU_SILU: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        silu_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= silu_approx_value(lane(gate_r, idx), lane(sigmoid_r, idx));
                    end
                    state_r <= SWIGLU;
                end
                SWIGLU: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        swiglu_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= swiglu_product_value(lane(silu_r, idx), lane(up_r, idx));
                    end
                    trace_r[5*8 +: 8] <= 8'h32;
                    trace_r[6*8 +: 8] <= 8'h41;
                    state_r <= DOWN;
                end
                DOWN: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        down_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= clip16(down_acc(swiglu_r, idx));
                    end
                    trace_r[7*8 +: 8] <= 8'h42;
                    trace_r[8*8 +: 8] <= 8'h51;
                    state_r <= RESIDUAL1;
                end
                RESIDUAL1: begin
                    for (idx = 0; idx < VALUES; idx = idx + 1) begin
                        final_output_r[idx*ELEM_WIDTH +: ELEM_WIDTH] <= clip16(add_lanes(lane(residual0_r, idx), lane(down_r, idx)));
                    end
                    trace_r[9*8 +: 8] <= 8'h52;
                    done_r <= 1'b1;
                    state_r <= DONE;
                end
                DONE: begin
                    if (!start_i) begin
                        done_r <= 1'b0;
                        state_r <= IDLE;
                    end
                end
                default: begin
                    done_r <= 1'b0;
                    state_r <= IDLE;
                end
            endcase
        end
    end
endmodule
