`timescale 1ns/1ps

module attention_kv_cache_fixture #(
    parameter int HEAD_DIM = 4,
    parameter int CACHE_SLOTS = 2,
    parameter int DATA_WIDTH = 8,
    parameter int SCORE_WIDTH = 32,
    parameter int WEIGHT_WIDTH = 8,
    parameter int OUT_WIDTH = 16,
    parameter int OUTPUT_SHIFT = 4,
    parameter int STATUS_WIDTH = 64
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    output logic signed [HEAD_DIM*OUT_WIDTH-1:0]      output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    typedef enum logic [3:0] {
        IDLE         = 4'd0,
        WRITE_HOLD   = 4'd1,
        WRITE_ACCEPT = 4'd2,
        READ_KEY0    = 4'd3,
        SCORE0       = 4'd4,
        READ_KEY1    = 4'd5,
        SCORE1       = 4'd6,
        CONTROL      = 4'd7,
        READ_VALUE0  = 4'd8,
        READ_VALUE1  = 4'd9,
        OUTPUT       = 4'd10,
        DONE         = 4'd11
    } state_t;

    state_t state_r;
    logic done_r;
    logic signed [DATA_WIDTH-1:0] key_cache_r [0:CACHE_SLOTS-1][0:HEAD_DIM-1];
    logic signed [DATA_WIDTH-1:0] value_cache_r [0:CACHE_SLOTS-1][0:HEAD_DIM-1];
    logic cache_write_valid_r;
    logic cache_write_accept_r;
    logic [0:0] cache_write_slot_r;
    logic signed [HEAD_DIM*DATA_WIDTH-1:0] cache_write_key_r;
    logic signed [HEAD_DIM*DATA_WIDTH-1:0] cache_write_value_r;
    logic key_read_valid_r;
    logic [0:0] key_read_slot_r;
    logic signed [HEAD_DIM*DATA_WIDTH-1:0] key_read_data_r;
    logic value_read_valid_r;
    logic [0:0] value_read_slot_r;
    logic signed [HEAD_DIM*DATA_WIDTH-1:0] value_read_data_r;
    logic score0_valid_r;
    logic score1_valid_r;
    logic control_valid_r;
    logic output_valid_r;
    logic signed [SCORE_WIDTH-1:0] score0_r;
    logic signed [SCORE_WIDTH-1:0] score1_r;
    logic [WEIGHT_WIDTH-1:0] weight0_r;
    logic [WEIGHT_WIDTH-1:0] weight1_r;
    logic signed [HEAD_DIM*OUT_WIDTH-1:0] output_r;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign output_o = output_r;
    assign status_o = status_r;

    function automatic logic signed [DATA_WIDTH-1:0] query_at(input int idx_i);
        begin
            case (idx_i)
                0: query_at = 8'sd3;
                1: query_at = -8'sd2;
                2: query_at = 8'sd5;
                3: query_at = -8'sd1;
                default: query_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] initial_key_at(input int idx_i);
        begin
            case (idx_i)
                0: initial_key_at = 8'sd2;
                1: initial_key_at = -8'sd1;
                2: initial_key_at = 8'sd4;
                3: initial_key_at = 8'sd3;
                default: initial_key_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] update_key_at(input int idx_i);
        begin
            case (idx_i)
                0: update_key_at = -8'sd3;
                1: update_key_at = 8'sd2;
                2: update_key_at = 8'sd1;
                3: update_key_at = 8'sd6;
                default: update_key_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] initial_value_at(input int idx_i);
        begin
            case (idx_i)
                0: initial_value_at = 8'sd7;
                1: initial_value_at = -8'sd4;
                2: initial_value_at = 8'sd3;
                3: initial_value_at = 8'sd2;
                default: initial_value_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [DATA_WIDTH-1:0] update_value_at(input int idx_i);
        begin
            case (idx_i)
                0: update_value_at = -8'sd2;
                1: update_value_at = 8'sd6;
                2: update_value_at = -8'sd5;
                3: update_value_at = 8'sd4;
                default: update_value_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [HEAD_DIM*DATA_WIDTH-1:0] pack_update_key();
        begin
            pack_update_key[0*DATA_WIDTH +: DATA_WIDTH] = update_key_at(0);
            pack_update_key[1*DATA_WIDTH +: DATA_WIDTH] = update_key_at(1);
            pack_update_key[2*DATA_WIDTH +: DATA_WIDTH] = update_key_at(2);
            pack_update_key[3*DATA_WIDTH +: DATA_WIDTH] = update_key_at(3);
        end
    endfunction

    function automatic logic signed [HEAD_DIM*DATA_WIDTH-1:0] pack_update_value();
        begin
            pack_update_value[0*DATA_WIDTH +: DATA_WIDTH] = update_value_at(0);
            pack_update_value[1*DATA_WIDTH +: DATA_WIDTH] = update_value_at(1);
            pack_update_value[2*DATA_WIDTH +: DATA_WIDTH] = update_value_at(2);
            pack_update_value[3*DATA_WIDTH +: DATA_WIDTH] = update_value_at(3);
        end
    endfunction

    function automatic logic signed [HEAD_DIM*DATA_WIDTH-1:0] pack_key_slot(input logic [0:0] slot_i);
        begin
            pack_key_slot[0*DATA_WIDTH +: DATA_WIDTH] = key_cache_r[slot_i][0];
            pack_key_slot[1*DATA_WIDTH +: DATA_WIDTH] = key_cache_r[slot_i][1];
            pack_key_slot[2*DATA_WIDTH +: DATA_WIDTH] = key_cache_r[slot_i][2];
            pack_key_slot[3*DATA_WIDTH +: DATA_WIDTH] = key_cache_r[slot_i][3];
        end
    endfunction

    function automatic logic signed [HEAD_DIM*DATA_WIDTH-1:0] pack_value_slot(input logic [0:0] slot_i);
        begin
            pack_value_slot[0*DATA_WIDTH +: DATA_WIDTH] = value_cache_r[slot_i][0];
            pack_value_slot[1*DATA_WIDTH +: DATA_WIDTH] = value_cache_r[slot_i][1];
            pack_value_slot[2*DATA_WIDTH +: DATA_WIDTH] = value_cache_r[slot_i][2];
            pack_value_slot[3*DATA_WIDTH +: DATA_WIDTH] = value_cache_r[slot_i][3];
        end
    endfunction

    function automatic logic signed [SCORE_WIDTH-1:0] dot_key_slot(input logic [0:0] slot_i);
        logic signed [SCORE_WIDTH-1:0] acc_s;
        logic signed [SCORE_WIDTH-1:0] query_ext_s;
        logic signed [SCORE_WIDTH-1:0] key_ext_s;
        logic signed [DATA_WIDTH-1:0] query_val_s;
        logic signed [DATA_WIDTH-1:0] key_val_s;
        begin
            acc_s = '0;
            query_val_s = query_at(0);
            key_val_s = key_cache_r[slot_i][0];
            query_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){query_val_s[DATA_WIDTH-1]}}, query_val_s};
            key_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){key_val_s[DATA_WIDTH-1]}}, key_val_s};
            acc_s = acc_s + (query_ext_s * key_ext_s);
            query_val_s = query_at(1);
            key_val_s = key_cache_r[slot_i][1];
            query_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){query_val_s[DATA_WIDTH-1]}}, query_val_s};
            key_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){key_val_s[DATA_WIDTH-1]}}, key_val_s};
            acc_s = acc_s + (query_ext_s * key_ext_s);
            query_val_s = query_at(2);
            key_val_s = key_cache_r[slot_i][2];
            query_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){query_val_s[DATA_WIDTH-1]}}, query_val_s};
            key_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){key_val_s[DATA_WIDTH-1]}}, key_val_s};
            acc_s = acc_s + (query_ext_s * key_ext_s);
            query_val_s = query_at(3);
            key_val_s = key_cache_r[slot_i][3];
            query_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){query_val_s[DATA_WIDTH-1]}}, query_val_s};
            key_ext_s = {{(SCORE_WIDTH-DATA_WIDTH){key_val_s[DATA_WIDTH-1]}}, key_val_s};
            acc_s = acc_s + (query_ext_s * key_ext_s);
            dot_key_slot = acc_s;
        end
    endfunction

    function automatic logic signed [OUT_WIDTH-1:0] weighted_value_at(input int idx_i);
        logic signed [31:0] value0_ext_s;
        logic signed [31:0] value1_ext_s;
        logic signed [31:0] weight0_ext_s;
        logic signed [31:0] weight1_ext_s;
        logic signed [63:0] accum_s;
        logic signed [63:0] shifted_s;
        begin
            value0_ext_s = {{(32-DATA_WIDTH){value_cache_r[0][idx_i][DATA_WIDTH-1]}}, value_cache_r[0][idx_i]};
            value1_ext_s = {{(32-DATA_WIDTH){value_cache_r[1][idx_i][DATA_WIDTH-1]}}, value_cache_r[1][idx_i]};
            weight0_ext_s = {{(32-WEIGHT_WIDTH){1'b0}}, weight0_r};
            weight1_ext_s = {{(32-WEIGHT_WIDTH){1'b0}}, weight1_r};
            accum_s = (weight0_ext_s * value0_ext_s) + (weight1_ext_s * value1_ext_s);
            shifted_s = accum_s >>> OUTPUT_SHIFT;
            weighted_value_at = shifted_s[OUT_WIDTH-1:0];
        end
    endfunction

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            cache_write_valid_r <= 1'b0;
            cache_write_accept_r <= 1'b0;
            cache_write_slot_r <= '0;
            cache_write_key_r <= '0;
            cache_write_value_r <= '0;
            key_read_valid_r <= 1'b0;
            key_read_slot_r <= '0;
            key_read_data_r <= '0;
            value_read_valid_r <= 1'b0;
            value_read_slot_r <= '0;
            value_read_data_r <= '0;
            score0_valid_r <= 1'b0;
            score1_valid_r <= 1'b0;
            control_valid_r <= 1'b0;
            output_valid_r <= 1'b0;
            score0_r <= '0;
            score1_r <= '0;
            weight0_r <= '0;
            weight1_r <= '0;
            output_r <= '0;
            status_r <= '0;
            key_cache_r[0][0] <= 8'sd2;
            key_cache_r[0][1] <= -8'sd1;
            key_cache_r[0][2] <= 8'sd4;
            key_cache_r[0][3] <= 8'sd3;
            key_cache_r[1][0] <= '0;
            key_cache_r[1][1] <= '0;
            key_cache_r[1][2] <= '0;
            key_cache_r[1][3] <= '0;
            value_cache_r[0][0] <= 8'sd7;
            value_cache_r[0][1] <= -8'sd4;
            value_cache_r[0][2] <= 8'sd3;
            value_cache_r[0][3] <= 8'sd2;
            value_cache_r[1][0] <= '0;
            value_cache_r[1][1] <= '0;
            value_cache_r[1][2] <= '0;
            value_cache_r[1][3] <= '0;
        end else begin
            cache_write_valid_r <= 1'b0;
            cache_write_accept_r <= 1'b0;
            key_read_valid_r <= 1'b0;
            value_read_valid_r <= 1'b0;
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    if (start_i) begin
                        cache_write_slot_r <= 1'b1;
                        cache_write_key_r <= pack_update_key();
                        cache_write_value_r <= pack_update_value();
                        key_read_data_r <= '0;
                        value_read_data_r <= '0;
                        score0_valid_r <= 1'b0;
                        score1_valid_r <= 1'b0;
                        control_valid_r <= 1'b0;
                        output_valid_r <= 1'b0;
                        score0_r <= '0;
                        score1_r <= '0;
                        weight0_r <= '0;
                        weight1_r <= '0;
                        output_r <= '0;
                        status_r <= '0;
                        key_cache_r[0][0] <= initial_key_at(0);
                        key_cache_r[0][1] <= initial_key_at(1);
                        key_cache_r[0][2] <= initial_key_at(2);
                        key_cache_r[0][3] <= initial_key_at(3);
                        value_cache_r[0][0] <= initial_value_at(0);
                        value_cache_r[0][1] <= initial_value_at(1);
                        value_cache_r[0][2] <= initial_value_at(2);
                        value_cache_r[0][3] <= initial_value_at(3);
                        state_r <= WRITE_HOLD;
                    end
                end
                WRITE_HOLD: begin
                    cache_write_valid_r <= 1'b1;
                    cache_write_slot_r <= 1'b1;
                    cache_write_key_r <= pack_update_key();
                    cache_write_value_r <= pack_update_value();
                    state_r <= WRITE_ACCEPT;
                end
                WRITE_ACCEPT: begin
                    cache_write_valid_r <= 1'b1;
                    cache_write_accept_r <= 1'b1;
                    cache_write_slot_r <= 1'b1;
                    cache_write_key_r <= pack_update_key();
                    cache_write_value_r <= pack_update_value();
                    key_cache_r[1][0] <= update_key_at(0);
                    key_cache_r[1][1] <= update_key_at(1);
                    key_cache_r[1][2] <= update_key_at(2);
                    key_cache_r[1][3] <= update_key_at(3);
                    value_cache_r[1][0] <= update_value_at(0);
                    value_cache_r[1][1] <= update_value_at(1);
                    value_cache_r[1][2] <= update_value_at(2);
                    value_cache_r[1][3] <= update_value_at(3);
                    status_r[0] <= 1'b1;
                    status_r[2:1] <= 2'd1;
                    state_r <= READ_KEY0;
                end
                READ_KEY0: begin
                    key_read_valid_r <= 1'b1;
                    key_read_slot_r <= 1'b0;
                    key_read_data_r <= pack_key_slot(1'b0);
                    status_r[3] <= 1'b1;
                    state_r <= SCORE0;
                end
                SCORE0: begin
                    score0_r <= dot_key_slot(1'b0);
                    score0_valid_r <= 1'b1;
                    status_r[7] <= 1'b1;
                    state_r <= READ_KEY1;
                end
                READ_KEY1: begin
                    key_read_valid_r <= 1'b1;
                    key_read_slot_r <= 1'b1;
                    key_read_data_r <= pack_key_slot(1'b1);
                    status_r[4] <= 1'b1;
                    state_r <= SCORE1;
                end
                SCORE1: begin
                    score1_r <= dot_key_slot(1'b1);
                    score1_valid_r <= 1'b1;
                    status_r[8] <= 1'b1;
                    state_r <= CONTROL;
                end
                CONTROL: begin
                    control_valid_r <= 1'b1;
                    if (score0_r >= score1_r) begin
                        weight0_r <= 8'd12;
                        weight1_r <= 8'd4;
                        status_r[31:24] <= 8'd12;
                        status_r[39:32] <= 8'd4;
                    end else begin
                        weight0_r <= 8'd4;
                        weight1_r <= 8'd12;
                        status_r[31:24] <= 8'd4;
                        status_r[39:32] <= 8'd12;
                    end
                    status_r[9] <= 1'b1;
                    state_r <= READ_VALUE0;
                end
                READ_VALUE0: begin
                    value_read_valid_r <= 1'b1;
                    value_read_slot_r <= 1'b0;
                    value_read_data_r <= pack_value_slot(1'b0);
                    status_r[5] <= 1'b1;
                    state_r <= READ_VALUE1;
                end
                READ_VALUE1: begin
                    value_read_valid_r <= 1'b1;
                    value_read_slot_r <= 1'b1;
                    value_read_data_r <= pack_value_slot(1'b1);
                    status_r[6] <= 1'b1;
                    state_r <= OUTPUT;
                end
                OUTPUT: begin
                    output_r[0*OUT_WIDTH +: OUT_WIDTH] <= weighted_value_at(0);
                    output_r[1*OUT_WIDTH +: OUT_WIDTH] <= weighted_value_at(1);
                    output_r[2*OUT_WIDTH +: OUT_WIDTH] <= weighted_value_at(2);
                    output_r[3*OUT_WIDTH +: OUT_WIDTH] <= weighted_value_at(3);
                    output_valid_r <= 1'b1;
                    done_r <= 1'b1;
                    status_r[10] <= 1'b1;
                    status_r[63] <= 1'b1;
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
