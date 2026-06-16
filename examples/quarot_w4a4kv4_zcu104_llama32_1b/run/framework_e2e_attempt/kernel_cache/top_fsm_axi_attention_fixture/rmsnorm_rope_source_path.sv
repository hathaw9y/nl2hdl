`timescale 1ns/1ps

module rmsnorm_rope_source_path #(
    parameter int HIDDEN_SIZE = 4,
    parameter int PAIR_COUNT = 2,
    parameter int IN_WIDTH = 8,
    parameter int GAMMA_WIDTH = 8,
    parameter int INV_RMS_WIDTH = 16,
    parameter int RMS_OUT_WIDTH = 18,
    parameter int ROPE_OUT_WIDTH = 18,
    parameter int COS_SIN_WIDTH = 8,
    parameter int RMS_OUTPUT_SHIFT = 12,
    parameter int ROPE_FRACTIONAL_BITS = 4,
    parameter int STATUS_WIDTH = 80
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    input  logic [1:0]                                norm_token_i,
    input  logic [3:0]                                rope_position_i,
    output logic                                      done_o,
    output logic signed [HIDDEN_SIZE*RMS_OUT_WIDTH-1:0] rms_output_o,
    output logic signed [HIDDEN_SIZE*ROPE_OUT_WIDTH-1:0] rope_output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        RMS_LOOKUP  = 3'b001,
        RMS_APPLY   = 3'b010,
        ROPE_LOOKUP = 3'b011,
        ROPE_APPLY  = 3'b100,
        DONE        = 3'b101
    } state_t;

    state_t state_r;
    logic done_r;
    logic [1:0] norm_selector_r;
    logic [3:0] rope_position_r;
    logic [INV_RMS_WIDTH-1:0] inv_rms_r;
    logic [15:0] sumsq_r;
    logic signed [COS_SIN_WIDTH-1:0] cos0_r;
    logic signed [COS_SIN_WIDTH-1:0] sin0_r;
    logic signed [COS_SIN_WIDTH-1:0] cos1_r;
    logic signed [COS_SIN_WIDTH-1:0] sin1_r;
    logic signed [HIDDEN_SIZE*RMS_OUT_WIDTH-1:0] rms_output_r;
    logic signed [HIDDEN_SIZE*ROPE_OUT_WIDTH-1:0] rope_output_r;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign rms_output_o = rms_output_r;
    assign rope_output_o = rope_output_r;
    assign status_o = status_r;

    function automatic logic signed [IN_WIDTH-1:0] rms_input_at(input logic [1:0] selector_i, input int idx_i);
        begin
            if (selector_i[0]) begin
                case (idx_i)
                    0: rms_input_at = -8'sd8;
                    1: rms_input_at = 8'sd32;
                    2: rms_input_at = -8'sd48;
                    3: rms_input_at = 8'sd64;
                    default: rms_input_at = '0;
                endcase
            end else begin
                case (idx_i)
                    0: rms_input_at = 8'sd24;
                    1: rms_input_at = -8'sd40;
                    2: rms_input_at = 8'sd56;
                    3: rms_input_at = -8'sd16;
                    default: rms_input_at = '0;
                endcase
            end
        end
    endfunction

    function automatic logic signed [GAMMA_WIDTH-1:0] rms_gamma_at(input int idx_i);
        begin
            case (idx_i)
                0: rms_gamma_at = 8'sd16;
                1: rms_gamma_at = 8'sd20;
                2: rms_gamma_at = 8'sd12;
                3: rms_gamma_at = 8'sd24;
                default: rms_gamma_at = '0;
            endcase
        end
    endfunction

    function automatic logic [INV_RMS_WIDTH-1:0] lookup_inv_rms(input logic [1:0] selector_i);
        begin
            if (selector_i[0]) begin
                lookup_inv_rms = 16'd6057;
            end else begin
                lookup_inv_rms = 16'd7024;
            end
        end
    endfunction

    function automatic logic [15:0] lookup_sumsq(input logic [1:0] selector_i);
        begin
            if (selector_i[0]) begin
                lookup_sumsq = 16'd7488;
            end else begin
                lookup_sumsq = 16'd5568;
            end
        end
    endfunction

    function automatic logic signed [RMS_OUT_WIDTH-1:0] rms_apply_at(
        input logic [1:0] selector_i,
        input int idx_i,
        input logic [INV_RMS_WIDTH-1:0] inv_rms_value
    );
        logic signed [IN_WIDTH-1:0] input_s;
        logic signed [GAMMA_WIDTH-1:0] gamma_s;
        logic signed [31:0] input_ext_s;
        logic signed [31:0] gamma_ext_s;
        logic signed [47:0] inv_ext_s;
        logic signed [95:0] product_s;
        logic signed [95:0] shifted_s;
        begin
            input_s = rms_input_at(selector_i, idx_i);
            gamma_s = rms_gamma_at(idx_i);
            input_ext_s = {{(32-IN_WIDTH){input_s[IN_WIDTH-1]}}, input_s};
            gamma_ext_s = {{(32-GAMMA_WIDTH){gamma_s[GAMMA_WIDTH-1]}}, gamma_s};
            inv_ext_s = {{(48-INV_RMS_WIDTH){1'b0}}, inv_rms_value};
            product_s = input_ext_s * gamma_ext_s * inv_ext_s;
            shifted_s = product_s >>> RMS_OUTPUT_SHIFT;
            rms_apply_at = shifted_s[RMS_OUT_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [IN_WIDTH-1:0] rope_input_at(input int idx_i);
        begin
            case (idx_i)
                0: rope_input_at = 8'sd24;
                1: rope_input_at = -8'sd32;
                2: rope_input_at = 8'sd40;
                3: rope_input_at = -8'sd16;
                default: rope_input_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [COS_SIN_WIDTH-1:0] lookup_rope_cos(input logic [3:0] position_i, input logic pair_i);
        begin
            if (position_i[0]) begin
                lookup_rope_cos = pair_i ? 8'sd7 : 8'sd13;
            end else begin
                lookup_rope_cos = pair_i ? 8'sd15 : 8'sd16;
            end
        end
    endfunction

    function automatic logic signed [COS_SIN_WIDTH-1:0] lookup_rope_sin(input logic [3:0] position_i, input logic pair_i);
        begin
            if (position_i[0]) begin
                lookup_rope_sin = pair_i ? -8'sd14 : 8'sd9;
            end else begin
                lookup_rope_sin = pair_i ? 8'sd4 : 8'sd0;
            end
        end
    endfunction

    function automatic logic signed [ROPE_OUT_WIDTH-1:0] rope_even_at(
        input int pair_i,
        input logic signed [COS_SIN_WIDTH-1:0] cos_value,
        input logic signed [COS_SIN_WIDTH-1:0] sin_value
    );
        logic signed [IN_WIDTH-1:0] even_s;
        logic signed [IN_WIDTH-1:0] odd_s;
        logic signed [31:0] even_ext_s;
        logic signed [31:0] odd_ext_s;
        logic signed [31:0] cos_ext_s;
        logic signed [31:0] sin_ext_s;
        logic signed [63:0] shifted_s;
        begin
            even_s = rope_input_at(pair_i * 2);
            odd_s = rope_input_at((pair_i * 2) + 1);
            even_ext_s = {{(32-IN_WIDTH){even_s[IN_WIDTH-1]}}, even_s};
            odd_ext_s = {{(32-IN_WIDTH){odd_s[IN_WIDTH-1]}}, odd_s};
            cos_ext_s = {{(32-COS_SIN_WIDTH){cos_value[COS_SIN_WIDTH-1]}}, cos_value};
            sin_ext_s = {{(32-COS_SIN_WIDTH){sin_value[COS_SIN_WIDTH-1]}}, sin_value};
            shifted_s = ((even_ext_s * cos_ext_s) - (odd_ext_s * sin_ext_s)) >>> ROPE_FRACTIONAL_BITS;
            rope_even_at = shifted_s[ROPE_OUT_WIDTH-1:0];
        end
    endfunction

    function automatic logic signed [ROPE_OUT_WIDTH-1:0] rope_odd_at(
        input int pair_i,
        input logic signed [COS_SIN_WIDTH-1:0] cos_value,
        input logic signed [COS_SIN_WIDTH-1:0] sin_value
    );
        logic signed [IN_WIDTH-1:0] even_s;
        logic signed [IN_WIDTH-1:0] odd_s;
        logic signed [31:0] even_ext_s;
        logic signed [31:0] odd_ext_s;
        logic signed [31:0] cos_ext_s;
        logic signed [31:0] sin_ext_s;
        logic signed [63:0] shifted_s;
        begin
            even_s = rope_input_at(pair_i * 2);
            odd_s = rope_input_at((pair_i * 2) + 1);
            even_ext_s = {{(32-IN_WIDTH){even_s[IN_WIDTH-1]}}, even_s};
            odd_ext_s = {{(32-IN_WIDTH){odd_s[IN_WIDTH-1]}}, odd_s};
            cos_ext_s = {{(32-COS_SIN_WIDTH){cos_value[COS_SIN_WIDTH-1]}}, cos_value};
            sin_ext_s = {{(32-COS_SIN_WIDTH){sin_value[COS_SIN_WIDTH-1]}}, sin_value};
            shifted_s = ((even_ext_s * sin_ext_s) + (odd_ext_s * cos_ext_s)) >>> ROPE_FRACTIONAL_BITS;
            rope_odd_at = shifted_s[ROPE_OUT_WIDTH-1:0];
        end
    endfunction

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            norm_selector_r <= '0;
            rope_position_r <= '0;
            inv_rms_r <= '0;
            sumsq_r <= '0;
            cos0_r <= '0;
            sin0_r <= '0;
            cos1_r <= '0;
            sin1_r <= '0;
            rms_output_r <= '0;
            rope_output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    if (start_i) begin
                        norm_selector_r <= norm_token_i;
                        rope_position_r <= rope_position_i;
                        inv_rms_r <= '0;
                        sumsq_r <= '0;
                        cos0_r <= '0;
                        sin0_r <= '0;
                        cos1_r <= '0;
                        sin1_r <= '0;
                        rms_output_r <= '0;
                        rope_output_r <= '0;
                        status_r <= '0;
                        state_r <= RMS_LOOKUP;
                    end
                end
                RMS_LOOKUP: begin
                    inv_rms_r <= lookup_inv_rms(norm_selector_r);
                    sumsq_r <= lookup_sumsq(norm_selector_r);
                    status_r[0] <= 1'b1;
                    status_r[2:1] <= norm_selector_r;
                    status_r[18:3] <= lookup_inv_rms(norm_selector_r);
                    status_r[34:19] <= lookup_sumsq(norm_selector_r);
                    state_r <= RMS_APPLY;
                end
                RMS_APPLY: begin
                    rms_output_r[0*RMS_OUT_WIDTH +: RMS_OUT_WIDTH] <= rms_apply_at(norm_selector_r, 0, inv_rms_r);
                    rms_output_r[1*RMS_OUT_WIDTH +: RMS_OUT_WIDTH] <= rms_apply_at(norm_selector_r, 1, inv_rms_r);
                    rms_output_r[2*RMS_OUT_WIDTH +: RMS_OUT_WIDTH] <= rms_apply_at(norm_selector_r, 2, inv_rms_r);
                    rms_output_r[3*RMS_OUT_WIDTH +: RMS_OUT_WIDTH] <= rms_apply_at(norm_selector_r, 3, inv_rms_r);
                    state_r <= ROPE_LOOKUP;
                end
                ROPE_LOOKUP: begin
                    cos0_r <= lookup_rope_cos(rope_position_r, 1'b0);
                    sin0_r <= lookup_rope_sin(rope_position_r, 1'b0);
                    cos1_r <= lookup_rope_cos(rope_position_r, 1'b1);
                    sin1_r <= lookup_rope_sin(rope_position_r, 1'b1);
                    status_r[35] <= 1'b1;
                    status_r[39:36] <= rope_position_r;
                    status_r[40] <= 1'b0;
                    status_r[48:41] <= lookup_rope_cos(rope_position_r, 1'b0);
                    status_r[56:49] <= lookup_rope_sin(rope_position_r, 1'b0);
                    status_r[57] <= 1'b1;
                    status_r[61:58] <= rope_position_r;
                    status_r[62] <= 1'b1;
                    status_r[70:63] <= lookup_rope_cos(rope_position_r, 1'b1);
                    status_r[78:71] <= lookup_rope_sin(rope_position_r, 1'b1);
                    state_r <= ROPE_APPLY;
                end
                ROPE_APPLY: begin
                    rope_output_r[0*ROPE_OUT_WIDTH +: ROPE_OUT_WIDTH] <= rope_even_at(0, cos0_r, sin0_r);
                    rope_output_r[1*ROPE_OUT_WIDTH +: ROPE_OUT_WIDTH] <= rope_odd_at(0, cos0_r, sin0_r);
                    rope_output_r[2*ROPE_OUT_WIDTH +: ROPE_OUT_WIDTH] <= rope_even_at(1, cos1_r, sin1_r);
                    rope_output_r[3*ROPE_OUT_WIDTH +: ROPE_OUT_WIDTH] <= rope_odd_at(1, cos1_r, sin1_r);
                    done_r <= 1'b1;
                    status_r[79] <= 1'b1;
                    state_r <= DONE;
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
