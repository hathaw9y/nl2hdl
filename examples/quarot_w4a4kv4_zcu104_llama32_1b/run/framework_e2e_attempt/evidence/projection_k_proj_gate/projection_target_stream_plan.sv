`timescale 1ns/1ps

module projection_target_stream_plan_core #(
    parameter int MEM_WORD_WIDTH = 128,
    parameter int PAYLOAD_WIDTH = 32,
    parameter int MAX_MEM_WORDS = 4,
    parameter int PAYLOADS_PER_WORD = 4,
    parameter int MAX_PAYLOADS = 16,
    parameter int TILE_ROWS = 2,
    parameter int TILE_COLS = 64,
    parameter int GROUP_SIZE = 4,
    parameter int GROUPS = 16,
    parameter int TRUE_PARALLEL_LANES = 2,
    parameter int PAIRS_PER_ROW = 32,
    parameter int SCALE_WIDTH = 8,
    parameter int ZERO_WIDTH = 8,
    parameter int ACT_WIDTH = 8,
    parameter int OUT_WIDTH = 32
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    input  logic [MEM_WORD_WIDTH-1:0]                 mem_word_i,
    input  logic                                      mem_valid_i,
    output logic                                      mem_ready_o,
    input  logic                                      mem_last_i,
    output logic                                      done_o,
    output logic signed [TILE_ROWS*OUT_WIDTH-1:0]     output_o,
    output logic [PAYLOAD_WIDTH-1:0]                  payload_link_word_o,
    output logic                                      payload_link_valid_o,
    output logic                                      payload_link_ready_o,
    output logic                                      payload_link_last_o,
    output logic [MAX_MEM_WORDS-1:0]                  input_accepted_trace_o,
    output logic [MAX_MEM_WORDS-1:0]                  input_last_trace_o,
    output logic [MAX_PAYLOADS-1:0]                   adapter_emit_trace_o,
    output logic [MAX_PAYLOADS-1:0]                   projection_consume_trace_o,
    output logic [MAX_PAYLOADS-1:0]                   ready_low_trace_o,
    output logic [$clog2(MAX_MEM_WORDS+1)-1:0]        input_count_o,
    output logic [$clog2(MAX_PAYLOADS+1)-1:0]         payload_emit_count_o,
    output logic [$clog2(MAX_PAYLOADS+1)-1:0]         projection_consume_count_o,
    output logic [PAIRS_PER_ROW*TILE_ROWS-1:0]        parallel_pair_trace_o
);
    localparam int CHUNK_IDX_W = (PAYLOADS_PER_WORD <= 1) ? 1 : $clog2(PAYLOADS_PER_WORD);
    localparam int INPUT_COUNT_W = (MAX_MEM_WORDS <= 1) ? 1 : $clog2(MAX_MEM_WORDS + 1);
    localparam int PAYLOAD_COUNT_W = (MAX_PAYLOADS <= 1) ? 1 : $clog2(MAX_PAYLOADS + 1);
    localparam int ROW_IDX_W = (TILE_ROWS <= 1) ? 1 : $clog2(TILE_ROWS);
    localparam int PAIR_IDX_W = (PAIRS_PER_ROW <= 1) ? 1 : $clog2(PAIRS_PER_ROW);
    localparam int PACKED_WEIGHT_WIDTH = MAX_PAYLOADS * PAYLOAD_WIDTH;
    localparam logic [CHUNK_IDX_W-1:0] LAST_CHUNK = CHUNK_IDX_W'(PAYLOADS_PER_WORD - 1);
    localparam logic [ROW_IDX_W-1:0] LAST_ROW = ROW_IDX_W'(TILE_ROWS - 1);
    localparam logic [PAIR_IDX_W-1:0] LAST_PAIR = PAIR_IDX_W'(PAIRS_PER_ROW - 1);

    typedef enum logic [2:0] {
        IDLE         = 3'b000,
        MEM_LOAD     = 3'b001,
        EMIT         = 3'b010,
        LOAD_PAIR    = 3'b011,
        DEQUANT_PAIR = 3'b100,
        MAC_PAIR     = 3'b101,
        ACC_PAIR     = 3'b110,
        DONE         = 3'b111
    } state_t;

    state_t state_r;
    logic [MEM_WORD_WIDTH-1:0] mem_word_r;
    logic [CHUNK_IDX_W-1:0] payload_idx_r;
    logic [PAYLOAD_WIDTH-1:0] payload_word_r;
    logic payload_valid_r;
    logic payload_last_r;
    logic [PACKED_WEIGHT_WIDTH-1:0] packed_weight_r;
    logic [MAX_MEM_WORDS-1:0] input_accepted_trace_r;
    logic [MAX_MEM_WORDS-1:0] input_last_trace_r;
    logic [MAX_PAYLOADS-1:0] adapter_emit_trace_r;
    logic [MAX_PAYLOADS-1:0] projection_consume_trace_r;
    logic [MAX_PAYLOADS-1:0] ready_low_trace_r;
    logic [INPUT_COUNT_W-1:0] input_count_r;
    logic [PAYLOAD_COUNT_W-1:0] payload_emit_count_r;
    logic [PAYLOAD_COUNT_W-1:0] projection_consume_count_r;
    logic current_mem_last_r;
    logic [ROW_IDX_W-1:0] row_idx_r;
    logic [PAIR_IDX_W-1:0] pair_idx_r;
    logic signed [OUT_WIDTH-1:0] acc_r;
    logic [3:0] nibble_lane0_r;
    logic [3:0] nibble_lane1_r;
    logic signed [ZERO_WIDTH-1:0] zero_lane0_r;
    logic signed [ZERO_WIDTH-1:0] zero_lane1_r;
    logic signed [SCALE_WIDTH-1:0] scale_lane0_r;
    logic signed [SCALE_WIDTH-1:0] scale_lane1_r;
    logic signed [ACT_WIDTH-1:0] activation_lane0_r;
    logic signed [ACT_WIDTH-1:0] activation_lane1_r;
    logic signed [31:0] dequant_lane0_r;
    logic signed [31:0] dequant_lane1_r;
    logic signed [31:0] product_lane0_r;
    logic signed [31:0] product_lane1_r;
    logic signed [TILE_ROWS*OUT_WIDTH-1:0] output_r;
    logic [PAIRS_PER_ROW*TILE_ROWS-1:0] parallel_pair_trace_r;
    logic done_r;
    logic input_fire_w;
    logic payload_link_valid_w;
    logic payload_link_ready_w;
    logic payload_link_fire_w;
    logic ready_low_request_w;
    logic [CHUNK_IDX_W-1:0] next_payload_idx_w;
    logic last_chunk_w;
    logic final_payload_w;
    logic [31:0] col_lane0_w;
    logic [31:0] col_lane1_w;
    logic [31:0] group_lane0_w;
    logic [31:0] group_lane1_w;
    logic signed [7:0] unpacked_lane0_w;
    logic signed [7:0] unpacked_lane1_w;
    logic signed [8:0] centered_lane0_w;
    logic signed [8:0] centered_lane1_w;
    logic signed [31:0] pair_sum_w;

    assign mem_ready_o = (state_r == MEM_LOAD) && (int'(input_count_r) < MAX_MEM_WORDS);
    assign done_o = done_r;
    assign output_o = output_r;
    assign payload_link_word_o = payload_word_r;
    assign payload_link_valid_o = payload_link_valid_w;
    assign payload_link_ready_o = payload_link_ready_w;
    assign payload_link_last_o = payload_last_r;
    assign input_accepted_trace_o = input_accepted_trace_r;
    assign input_last_trace_o = input_last_trace_r;
    assign adapter_emit_trace_o = adapter_emit_trace_r;
    assign projection_consume_trace_o = projection_consume_trace_r;
    assign ready_low_trace_o = ready_low_trace_r;
    assign input_count_o = input_count_r;
    assign payload_emit_count_o = payload_emit_count_r;
    assign projection_consume_count_o = projection_consume_count_r;
    assign parallel_pair_trace_o = parallel_pair_trace_r;
    assign input_fire_w = mem_valid_i && mem_ready_o;
    assign payload_link_valid_w = payload_valid_r;
    assign ready_low_request_w =
        ((projection_consume_count_r == PAYLOAD_COUNT_W'(0)) && !ready_low_trace_r[0]) ||
        ((projection_consume_count_r == PAYLOAD_COUNT_W'(PAYLOADS_PER_WORD - 1)) && !ready_low_trace_r[PAYLOADS_PER_WORD - 1]) ||
        ((projection_consume_count_r == PAYLOAD_COUNT_W'(PAYLOADS_PER_WORD)) && !ready_low_trace_r[PAYLOADS_PER_WORD]);
    assign payload_link_ready_w = (state_r == EMIT) && payload_valid_r && !ready_low_request_w;
    assign payload_link_fire_w = payload_link_valid_w && payload_link_ready_w;
    assign next_payload_idx_w = payload_idx_r + CHUNK_IDX_W'(1);
    assign last_chunk_w = (payload_idx_r == LAST_CHUNK);
    assign final_payload_w = last_chunk_w && current_mem_last_r;
    assign col_lane0_w = int'(pair_idx_r) * TRUE_PARALLEL_LANES;
    assign col_lane1_w = col_lane0_w + 32'd1;
    assign group_lane0_w = col_lane0_w / GROUP_SIZE;
    assign group_lane1_w = col_lane1_w / GROUP_SIZE;
    assign unpacked_lane0_w = sign_extend_int4(nibble_lane0_r);
    assign unpacked_lane1_w = sign_extend_int4(nibble_lane1_r);
    assign centered_lane0_w = $signed({unpacked_lane0_w[7], unpacked_lane0_w}) - $signed({zero_lane0_r[ZERO_WIDTH-1], zero_lane0_r});
    assign centered_lane1_w = $signed({unpacked_lane1_w[7], unpacked_lane1_w}) - $signed({zero_lane1_r[ZERO_WIDTH-1], zero_lane1_r});
    assign pair_sum_w = product_lane0_r + product_lane1_r;

    function automatic logic signed [7:0] sign_extend_int4(input logic [3:0] nibble_i);
        begin
            sign_extend_int4 = { {4{nibble_i[3]}}, nibble_i };
        end
    endfunction

    function automatic logic [3:0] packed_nibble_at(
        input logic [ROW_IDX_W-1:0] row_idx_i,
        input int col_idx_i
    );
        int flat_idx;
        begin
            flat_idx = (int'(row_idx_i) * TILE_COLS) + col_idx_i;
            packed_nibble_at = packed_weight_r[flat_idx*4 +: 4];
        end
    endfunction

    function automatic logic signed [ZERO_WIDTH-1:0] zero_at(
        input logic [ROW_IDX_W-1:0] row_idx_i,
        input int group_idx_i
    );
        int meta_idx;
        begin
            meta_idx = (int'(row_idx_i) * GROUPS) + group_idx_i;
            case (meta_idx)
                0: zero_at = -8'sd2;
                1: zero_at = 8'sd3;
                2: zero_at = 8'sd1;
                3: zero_at = -8'sd1;
                4: zero_at = -8'sd3;
                5: zero_at = 8'sd2;
                6: zero_at = 8'sd0;
                7: zero_at = -8'sd2;
                8: zero_at = 8'sd3;
                9: zero_at = 8'sd1;
                10: zero_at = -8'sd1;
                11: zero_at = -8'sd3;
                12: zero_at = 8'sd2;
                13: zero_at = 8'sd0;
                14: zero_at = -8'sd2;
                15: zero_at = 8'sd3;
                16: zero_at = 8'sd1;
                17: zero_at = -8'sd1;
                18: zero_at = -8'sd3;
                19: zero_at = 8'sd2;
                20: zero_at = 8'sd0;
                21: zero_at = -8'sd2;
                22: zero_at = 8'sd3;
                23: zero_at = 8'sd1;
                24: zero_at = -8'sd1;
                25: zero_at = -8'sd3;
                26: zero_at = 8'sd2;
                27: zero_at = 8'sd0;
                28: zero_at = -8'sd2;
                29: zero_at = 8'sd3;
                30: zero_at = 8'sd1;
                31: zero_at = -8'sd1;
                default: zero_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [SCALE_WIDTH-1:0] scale_at(
        input logic [ROW_IDX_W-1:0] row_idx_i,
        input int group_idx_i
    );
        int meta_idx;
        begin
            meta_idx = (int'(row_idx_i) * GROUPS) + group_idx_i;
            case (meta_idx)
                0: scale_at = -8'sd12;
                1: scale_at = -8'sd8;
                2: scale_at = -8'sd4;
                3: scale_at = 8'sd0;
                4: scale_at = 8'sd4;
                5: scale_at = 8'sd8;
                6: scale_at = 8'sd12;
                7: scale_at = 8'sd16;
                8: scale_at = -8'sd20;
                9: scale_at = -8'sd16;
                10: scale_at = -8'sd12;
                11: scale_at = -8'sd8;
                12: scale_at = -8'sd4;
                13: scale_at = 8'sd0;
                14: scale_at = 8'sd4;
                15: scale_at = 8'sd8;
                16: scale_at = -8'sd4;
                17: scale_at = 8'sd4;
                18: scale_at = 8'sd12;
                19: scale_at = -8'sd20;
                20: scale_at = -8'sd12;
                21: scale_at = -8'sd4;
                22: scale_at = 8'sd4;
                23: scale_at = 8'sd12;
                24: scale_at = -8'sd20;
                25: scale_at = -8'sd12;
                26: scale_at = -8'sd4;
                27: scale_at = 8'sd4;
                28: scale_at = 8'sd12;
                29: scale_at = -8'sd20;
                30: scale_at = -8'sd12;
                31: scale_at = -8'sd4;
                default: scale_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [ACT_WIDTH-1:0] activation_at(input int col_idx_i);
        begin
            case (col_idx_i)
                0: activation_at = -8'sd6;
                1: activation_at = 8'sd1;
                2: activation_at = 8'sd8;
                3: activation_at = -8'sd2;
                4: activation_at = 8'sd5;
                5: activation_at = -8'sd5;
                6: activation_at = 8'sd2;
                7: activation_at = -8'sd8;
                8: activation_at = -8'sd1;
                9: activation_at = 8'sd6;
                10: activation_at = -8'sd4;
                11: activation_at = 8'sd3;
                12: activation_at = -8'sd7;
                13: activation_at = 8'sd0;
                14: activation_at = 8'sd7;
                15: activation_at = -8'sd3;
                16: activation_at = 8'sd4;
                17: activation_at = -8'sd6;
                18: activation_at = 8'sd1;
                19: activation_at = 8'sd8;
                20: activation_at = -8'sd2;
                21: activation_at = 8'sd5;
                22: activation_at = -8'sd5;
                23: activation_at = 8'sd2;
                24: activation_at = -8'sd8;
                25: activation_at = -8'sd1;
                26: activation_at = 8'sd6;
                27: activation_at = -8'sd4;
                28: activation_at = 8'sd3;
                29: activation_at = -8'sd7;
                30: activation_at = 8'sd0;
                31: activation_at = 8'sd7;
                32: activation_at = -8'sd3;
                33: activation_at = 8'sd4;
                34: activation_at = -8'sd6;
                35: activation_at = 8'sd1;
                36: activation_at = 8'sd8;
                37: activation_at = -8'sd2;
                38: activation_at = 8'sd5;
                39: activation_at = -8'sd5;
                40: activation_at = 8'sd2;
                41: activation_at = -8'sd8;
                42: activation_at = -8'sd1;
                43: activation_at = 8'sd6;
                44: activation_at = -8'sd4;
                45: activation_at = 8'sd3;
                46: activation_at = -8'sd7;
                47: activation_at = 8'sd0;
                48: activation_at = 8'sd7;
                49: activation_at = -8'sd3;
                50: activation_at = 8'sd4;
                51: activation_at = -8'sd6;
                52: activation_at = 8'sd1;
                53: activation_at = 8'sd8;
                54: activation_at = -8'sd2;
                55: activation_at = 8'sd5;
                56: activation_at = -8'sd5;
                57: activation_at = 8'sd2;
                58: activation_at = -8'sd8;
                59: activation_at = -8'sd1;
                60: activation_at = 8'sd6;
                61: activation_at = -8'sd4;
                62: activation_at = 8'sd3;
                63: activation_at = -8'sd7;
                default: activation_at = '0;
            endcase
        end
    endfunction

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            mem_word_r <= '0;
            payload_idx_r <= '0;
            payload_word_r <= '0;
            payload_valid_r <= 1'b0;
            payload_last_r <= 1'b0;
            packed_weight_r <= '0;
            input_accepted_trace_r <= '0;
            input_last_trace_r <= '0;
            adapter_emit_trace_r <= '0;
            projection_consume_trace_r <= '0;
            ready_low_trace_r <= '0;
            input_count_r <= '0;
            payload_emit_count_r <= '0;
            projection_consume_count_r <= '0;
            current_mem_last_r <= 1'b0;
            row_idx_r <= '0;
            pair_idx_r <= '0;
            acc_r <= '0;
            nibble_lane0_r <= '0;
            nibble_lane1_r <= '0;
            zero_lane0_r <= '0;
            zero_lane1_r <= '0;
            scale_lane0_r <= '0;
            scale_lane1_r <= '0;
            activation_lane0_r <= '0;
            activation_lane1_r <= '0;
            dequant_lane0_r <= '0;
            dequant_lane1_r <= '0;
            product_lane0_r <= '0;
            product_lane1_r <= '0;
            output_r <= '0;
            parallel_pair_trace_r <= '0;
            done_r <= 1'b0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    payload_valid_r <= 1'b0;
                    payload_last_r <= 1'b0;
                    if (start_i) begin
                        state_r <= MEM_LOAD;
                        mem_word_r <= '0;
                        payload_idx_r <= '0;
                        payload_word_r <= '0;
                        packed_weight_r <= '0;
                        input_accepted_trace_r <= '0;
                        input_last_trace_r <= '0;
                        adapter_emit_trace_r <= '0;
                        projection_consume_trace_r <= '0;
                        ready_low_trace_r <= '0;
                        input_count_r <= '0;
                        payload_emit_count_r <= '0;
                        projection_consume_count_r <= '0;
                        current_mem_last_r <= 1'b0;
                        row_idx_r <= '0;
                        pair_idx_r <= '0;
                        acc_r <= '0;
                        output_r <= '0;
                        parallel_pair_trace_r <= '0;
                    end
                end
                MEM_LOAD: begin
                    payload_valid_r <= 1'b0;
                    payload_last_r <= 1'b0;
                    if (input_fire_w) begin
                        mem_word_r <= mem_word_i;
                        input_accepted_trace_r[int'(input_count_r)] <= 1'b1;
                        input_last_trace_r[int'(input_count_r)] <= mem_last_i;
                        input_count_r <= input_count_r + INPUT_COUNT_W'(1);
                        current_mem_last_r <= mem_last_i;
                        payload_idx_r <= '0;
                        payload_word_r <= mem_word_i[0 +: PAYLOAD_WIDTH];
                        payload_valid_r <= 1'b1;
                        payload_last_r <= mem_last_i && (LAST_CHUNK == '0);
                        state_r <= EMIT;
                    end
                end
                EMIT: begin
                    if (payload_valid_r && !payload_link_ready_w) begin
                        ready_low_trace_r[int'(projection_consume_count_r)] <= 1'b1;
                    end else if (payload_link_fire_w) begin
                        packed_weight_r[int'(projection_consume_count_r)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH] <= payload_word_r;
                        adapter_emit_trace_r[int'(projection_consume_count_r)] <= 1'b1;
                        projection_consume_trace_r[int'(projection_consume_count_r)] <= 1'b1;
                        payload_emit_count_r <= payload_emit_count_r + PAYLOAD_COUNT_W'(1);
                        projection_consume_count_r <= projection_consume_count_r + PAYLOAD_COUNT_W'(1);
                        if (final_payload_w) begin
                            payload_valid_r <= 1'b0;
                            state_r <= LOAD_PAIR;
                            row_idx_r <= '0;
                            pair_idx_r <= '0;
                            acc_r <= '0;
                        end else if (last_chunk_w) begin
                            payload_valid_r <= 1'b0;
                            payload_last_r <= 1'b0;
                            payload_idx_r <= '0;
                            state_r <= MEM_LOAD;
                        end else begin
                            payload_idx_r <= next_payload_idx_w;
                            payload_word_r <= mem_word_r[int'(next_payload_idx_w)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH];
                            payload_last_r <= current_mem_last_r && (next_payload_idx_w == LAST_CHUNK);
                        end
                    end
                end
                LOAD_PAIR: begin
                    nibble_lane0_r <= packed_nibble_at(row_idx_r, int'(col_lane0_w));
                    nibble_lane1_r <= packed_nibble_at(row_idx_r, int'(col_lane1_w));
                    zero_lane0_r <= zero_at(row_idx_r, int'(group_lane0_w));
                    zero_lane1_r <= zero_at(row_idx_r, int'(group_lane1_w));
                    scale_lane0_r <= scale_at(row_idx_r, int'(group_lane0_w));
                    scale_lane1_r <= scale_at(row_idx_r, int'(group_lane1_w));
                    activation_lane0_r <= activation_at(int'(col_lane0_w));
                    activation_lane1_r <= activation_at(int'(col_lane1_w));
                    state_r <= DEQUANT_PAIR;
                end
                DEQUANT_PAIR: begin
                    dequant_lane0_r <= {{23{centered_lane0_w[8]}}, centered_lane0_w} * {{24{scale_lane0_r[SCALE_WIDTH-1]}}, scale_lane0_r};
                    dequant_lane1_r <= {{23{centered_lane1_w[8]}}, centered_lane1_w} * {{24{scale_lane1_r[SCALE_WIDTH-1]}}, scale_lane1_r};
                    state_r <= MAC_PAIR;
                end
                MAC_PAIR: begin
                    product_lane0_r <= dequant_lane0_r * {{24{activation_lane0_r[ACT_WIDTH-1]}}, activation_lane0_r};
                    product_lane1_r <= dequant_lane1_r * {{24{activation_lane1_r[ACT_WIDTH-1]}}, activation_lane1_r};
                    state_r <= ACC_PAIR;
                end
                ACC_PAIR: begin
                    parallel_pair_trace_r[(int'(row_idx_r) * PAIRS_PER_ROW) + int'(pair_idx_r)] <= 1'b1;
                    if (pair_idx_r == LAST_PAIR) begin
                        output_r[row_idx_r*OUT_WIDTH +: OUT_WIDTH] <= acc_r + pair_sum_w;
                        if (row_idx_r == LAST_ROW) begin
                            state_r <= DONE;
                            done_r <= 1'b1;
                        end else begin
                            row_idx_r <= row_idx_r + ROW_IDX_W'(1);
                            pair_idx_r <= '0;
                            acc_r <= '0;
                            state_r <= LOAD_PAIR;
                        end
                    end else begin
                        pair_idx_r <= pair_idx_r + PAIR_IDX_W'(1);
                        acc_r <= acc_r + pair_sum_w;
                        state_r <= LOAD_PAIR;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        payload_valid_r <= 1'b0;
                        payload_last_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    payload_valid_r <= 1'b0;
                    payload_last_r <= 1'b0;
                end
            endcase
        end
    end
endmodule


module projection_target_stream_plan #(
    parameter int MEM_WORD_WIDTH = 128,
    parameter int PAYLOAD_WIDTH = 32,
    parameter int MAX_MEM_WORDS = 4,
    parameter int PAYLOADS_PER_WORD = 4,
    parameter int MAX_PAYLOADS = 16,
    parameter int TILE_ROWS = 2,
    parameter int TILE_COLS = 64,
    parameter int GROUP_SIZE = 4,
    parameter int GROUPS = 16,
    parameter int TRUE_PARALLEL_LANES = 2,
    parameter int PAIRS_PER_ROW = 32,
    parameter int OUT_WIDTH = 32
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    input  logic [MEM_WORD_WIDTH-1:0]                 mem_word_i,
    input  logic                                      mem_valid_i,
    output logic                                      mem_ready_o,
    input  logic                                      mem_last_i,
    output logic                                      done_o,
    output logic signed [TILE_ROWS*OUT_WIDTH-1:0]     output_o,
    output logic [PAYLOAD_WIDTH-1:0]                  payload_link_word_o,
    output logic                                      payload_link_valid_o,
    output logic                                      payload_link_ready_o,
    output logic                                      payload_link_last_o,
    output logic [47:0]                               debug_trace_o
);
    logic [MAX_MEM_WORDS-1:0] input_accepted_trace_w;
    logic [MAX_MEM_WORDS-1:0] input_last_trace_w;
    logic [MAX_PAYLOADS-1:0] adapter_emit_trace_w;
    logic [MAX_PAYLOADS-1:0] projection_consume_trace_w;
    logic [MAX_PAYLOADS-1:0] ready_low_trace_w;
    logic [$clog2(MAX_MEM_WORDS+1)-1:0] input_count_w;
    logic [$clog2(MAX_PAYLOADS+1)-1:0] payload_emit_count_w;
    logic [$clog2(MAX_PAYLOADS+1)-1:0] projection_consume_count_w;
    logic [PAIRS_PER_ROW*TILE_ROWS-1:0] parallel_pair_trace_w;
    logic [6:0] parallel_pair_count_w;

    assign parallel_pair_count_w = (parallel_pair_trace_w == {(PAIRS_PER_ROW*TILE_ROWS){1'b1}}) ? 7'd64 : 7'd0;
    assign debug_trace_o = {
        7'b0,
        parallel_pair_count_w,
        projection_consume_count_w,
        payload_emit_count_w,
        ready_low_trace_w,
        input_last_trace_w,
        input_accepted_trace_w
    };

    projection_target_stream_plan_core #(
        .MEM_WORD_WIDTH(MEM_WORD_WIDTH),
        .PAYLOAD_WIDTH(PAYLOAD_WIDTH),
        .MAX_MEM_WORDS(MAX_MEM_WORDS),
        .PAYLOADS_PER_WORD(PAYLOADS_PER_WORD),
        .MAX_PAYLOADS(MAX_PAYLOADS),
        .TILE_ROWS(TILE_ROWS),
        .TILE_COLS(TILE_COLS),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUPS(GROUPS),
        .TRUE_PARALLEL_LANES(TRUE_PARALLEL_LANES),
        .PAIRS_PER_ROW(PAIRS_PER_ROW)
    ) u_core (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .mem_word_i(mem_word_i),
        .mem_valid_i(mem_valid_i),
        .mem_ready_o(mem_ready_o),
        .mem_last_i(mem_last_i),
        .done_o(done_o),
        .output_o(output_o),
        .payload_link_word_o(payload_link_word_o),
        .payload_link_valid_o(payload_link_valid_o),
        .payload_link_ready_o(payload_link_ready_o),
        .payload_link_last_o(payload_link_last_o),
        .input_accepted_trace_o(input_accepted_trace_w),
        .input_last_trace_o(input_last_trace_w),
        .adapter_emit_trace_o(adapter_emit_trace_w),
        .projection_consume_trace_o(projection_consume_trace_w),
        .ready_low_trace_o(ready_low_trace_w),
        .input_count_o(input_count_w),
        .payload_emit_count_o(payload_emit_count_w),
        .projection_consume_count_o(projection_consume_count_w),
        .parallel_pair_trace_o(parallel_pair_trace_w)
    );
endmodule
