`timescale 1ns/1ps

module projection_axi_stream_integration #(
    parameter int ADDR_WIDTH = 32,
    parameter int MEM_DATA_WIDTH = 128,
    parameter int PAYLOAD_WIDTH = 32,
    parameter int FIXTURE_READ_BEATS = 2,
    parameter int PAYLOADS_PER_BEAT = 4,
    parameter int TILE_ROWS = 2,
    parameter int TILE_COLS = 32,
    parameter int GROUP_SIZE = 4,
    parameter int GROUPS = 8,
    parameter int OUT_WIDTH = 32,
    parameter logic [ADDR_WIDTH-1:0] FIXTURE_REQUEST_ADDR = 32'h00120000,
    parameter logic [7:0] EXPECTED_AXI_ID = 8'h02
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    output logic signed [TILE_ROWS*OUT_WIDTH-1:0]     output_o,
    output logic [63:0]                               integration_status_o
);
    localparam int BYTES_PER_BEAT = MEM_DATA_WIDTH / 8;
    localparam int FIXTURE_PAYLOADS = FIXTURE_READ_BEATS * PAYLOADS_PER_BEAT;
    localparam int TOTAL_WEIGHTS = TILE_ROWS * TILE_COLS;
    localparam int BEAT_IDX_W = (FIXTURE_READ_BEATS <= 1) ? 1 : $clog2(FIXTURE_READ_BEATS + 1);
    localparam int CHUNK_IDX_W = (PAYLOADS_PER_BEAT <= 1) ? 1 : $clog2(PAYLOADS_PER_BEAT);
    localparam int PAYLOAD_COUNT_W = (FIXTURE_PAYLOADS <= 1) ? 1 : $clog2(FIXTURE_PAYLOADS + 1);
    localparam int COMPUTE_IDX_W = (TOTAL_WEIGHTS <= 1) ? 1 : $clog2(TOTAL_WEIGHTS + 1);
    localparam logic [2:0] ARSIZE_VALUE = 3'($clog2(BYTES_PER_BEAT));
    localparam logic [7:0] ARLEN_VALUE = 8'(FIXTURE_READ_BEATS - 1);
    localparam logic [1:0] ARBURST_INCR = 2'b01;
    localparam logic [CHUNK_IDX_W-1:0] LAST_CHUNK = CHUNK_IDX_W'(PAYLOADS_PER_BEAT - 1);

    typedef enum logic [3:0] {
        IDLE            = 4'h0,
        ISSUE_AR        = 4'h1,
        R_WAIT          = 4'h2,
        EMIT            = 4'h3,
        COMPUTE_LOAD    = 4'h4,
        COMPUTE_CENTER  = 4'h5,
        COMPUTE_DEQUANT = 4'h6,
        COMPUTE_PRODUCT = 4'h7,
        COMPUTE_ACC     = 4'h8,
        DONE            = 4'h9
    } state_t;

    state_t state_r;
    logic done_r;
    logic arvalid_r;
    logic [1:0] ar_ready_low_count_r;
    logic axi_arready_w;
    logic axi_arvalid_w;
    logic [ADDR_WIDTH-1:0] axi_araddr_w;
    logic [7:0] axi_arlen_w;
    logic [2:0] axi_arsize_w;
    logic [1:0] axi_arburst_w;
    logic [7:0] axi_arid_w;
    logic axi_rvalid_r;
    logic [BEAT_IDX_W-1:0] axi_rbeat_idx_r;
    logic axi_rvalid_w;
    logic axi_rready_w;
    logic [MEM_DATA_WIDTH-1:0] axi_rdata_w;
    logic [7:0] axi_rid_w;
    logic [1:0] axi_rresp_w;
    logic axi_rlast_w;
    logic [MEM_DATA_WIDTH-1:0] mem_word_r;
    logic [BEAT_IDX_W-1:0] beat_count_r;
    logic [CHUNK_IDX_W-1:0] chunk_idx_r;
    logic [PAYLOAD_COUNT_W-1:0] payload_emit_count_r;
    logic [PAYLOAD_COUNT_W-1:0] payload_consume_count_r;
    logic [PAYLOAD_WIDTH-1:0] payload_link_word_r;
    logic payload_link_valid_r;
    logic payload_link_last_r;
    logic payload_link_ready_w;
    logic current_beat_final_r;
    logic [FIXTURE_READ_BEATS-1:0] accepted_beat_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rlast_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rid_error_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rresp_error_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rlast_error_trace_r;
    logic [FIXTURE_PAYLOADS-1:0] ready_low_trace_r;
    logic [FIXTURE_PAYLOADS-1:0] ready_low_seen_r;
    logic [TOTAL_WEIGHTS*4-1:0] packed_weight_r;
    logic signed [TILE_ROWS*OUT_WIDTH-1:0] output_r;
    logic [COMPUTE_IDX_W-1:0] compute_idx_r;
    logic compute_row_w;
    logic compute_row_r;
    logic signed [7:0] compute_weight_r;
    logic signed [7:0] compute_zero_r;
    logic signed [7:0] compute_scale_r;
    logic signed [7:0] compute_activation_r;
    logic signed [15:0] compute_centered_r;
    logic signed [23:0] compute_dequant_r;
    logic signed [OUT_WIDTH-1:0] compute_product_r;
    logic r_fire_w;
    logic payload_fire_w;
    logic [CHUNK_IDX_W-1:0] next_chunk_idx_w;
    logic last_chunk_w;
    logic final_payload_w;
    logic expected_rlast_w;
    logic ready_low_required_w;

    assign done_o = done_r;
    assign output_o = output_r;
    assign axi_arvalid_w = arvalid_r;
    assign axi_arready_w = (state_r == ISSUE_AR) && (ar_ready_low_count_r >= 2'd2);
    assign axi_araddr_w = FIXTURE_REQUEST_ADDR;
    assign axi_arlen_w = ARLEN_VALUE;
    assign axi_arsize_w = ARSIZE_VALUE;
    assign axi_arburst_w = ARBURST_INCR;
    assign axi_arid_w = EXPECTED_AXI_ID;
    assign axi_rvalid_w = axi_rvalid_r;
    assign axi_rready_w = (state_r == R_WAIT);
    assign axi_rdata_w = axi_word_at(axi_rbeat_idx_r);
    assign axi_rid_w = EXPECTED_AXI_ID;
    assign axi_rresp_w = 2'b00;
    assign axi_rlast_w = (int'(axi_rbeat_idx_r) == (FIXTURE_READ_BEATS - 1));
    assign payload_link_ready_w = payload_link_valid_r && !ready_low_required_w;
    assign r_fire_w = axi_rvalid_w && axi_rready_w;
    assign payload_fire_w = payload_link_valid_r && payload_link_ready_w;
    assign next_chunk_idx_w = chunk_idx_r + CHUNK_IDX_W'(1);
    assign last_chunk_w = (chunk_idx_r == LAST_CHUNK);
    assign final_payload_w = current_beat_final_r && last_chunk_w;
    assign expected_rlast_w = (int'(beat_count_r) == (FIXTURE_READ_BEATS - 1));
    assign ready_low_required_w = payload_link_valid_r &&
        ((int'(payload_consume_count_r) == 0 && !ready_low_seen_r[0]) ||
         (int'(payload_consume_count_r) == (PAYLOADS_PER_BEAT - 1) && !ready_low_seen_r[PAYLOADS_PER_BEAT - 1]) ||
         (int'(payload_consume_count_r) == PAYLOADS_PER_BEAT && !ready_low_seen_r[PAYLOADS_PER_BEAT]));
    assign compute_row_w = (int'(compute_idx_r) >= TILE_COLS);
    assign integration_status_o[7:0] = {4'(state_r), done_r, payload_link_ready_w, axi_rready_w, axi_arready_w};
    assign integration_status_o[15:8] = 8'(payload_emit_count_r);
    assign integration_status_o[23:16] = 8'(payload_consume_count_r);
    assign integration_status_o[31:24] = ready_low_trace_r;
    assign integration_status_o[33:32] = accepted_beat_trace_r;
    assign integration_status_o[35:34] = rlast_trace_r;
    assign integration_status_o[37:36] = rid_error_trace_r;
    assign integration_status_o[39:38] = rresp_error_trace_r;
    assign integration_status_o[41:40] = rlast_error_trace_r;
    assign integration_status_o[42] = (rid_error_trace_r == '0);
    assign integration_status_o[43] = (rresp_error_trace_r == '0);
    assign integration_status_o[44] = (rlast_error_trace_r == '0);
    assign integration_status_o[45] = (rid_error_trace_r == '0) && (rresp_error_trace_r == '0) && (rlast_error_trace_r == '0);
    assign integration_status_o[46] = (axi_arlen_w == ARLEN_VALUE);
    assign integration_status_o[47] = done_r;
    assign integration_status_o[63:48] = 16'hc1a5;

    function automatic logic [MEM_DATA_WIDTH-1:0] axi_word_at(input logic [BEAT_IDX_W-1:0] idx_i);
        begin
            case (int'(idx_i))
                0: axi_word_at = 128'h61c72d83e94fa50b61c72d83e94fa50b;
                1: axi_word_at = 128'h61c72d83e94fa50b61c72d83e94fa50b;
                default: axi_word_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] sign_extend_int4(input logic [3:0] nibble_i);
        begin
            sign_extend_int4 = { {4{nibble_i[3]}}, nibble_i };
        end
    endfunction

    function automatic logic signed [7:0] activation_at(input int col_idx_i);
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
                default: activation_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] zero_at(input int row_idx_i, input int group_idx_i);
        int meta_idx;
        begin
            meta_idx = row_idx_i * GROUPS + group_idx_i;
            case (meta_idx)
                0: zero_at = -8'sd2;
                1: zero_at = 8'sd3;
                2: zero_at = 8'sd1;
                3: zero_at = -8'sd1;
                4: zero_at = -8'sd3;
                5: zero_at = 8'sd2;
                6: zero_at = 8'sd0;
                7: zero_at = -8'sd2;
                8: zero_at = 8'sd1;
                9: zero_at = -8'sd1;
                10: zero_at = -8'sd3;
                11: zero_at = 8'sd2;
                12: zero_at = 8'sd0;
                13: zero_at = -8'sd2;
                14: zero_at = 8'sd3;
                15: zero_at = 8'sd1;
                default: zero_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] scale_at(input int row_idx_i, input int group_idx_i);
        int meta_idx;
        begin
            meta_idx = row_idx_i * GROUPS + group_idx_i;
            case (meta_idx)
                0: scale_at = -8'sd12;
                1: scale_at = -8'sd8;
                2: scale_at = -8'sd4;
                3: scale_at = 8'sd0;
                4: scale_at = 8'sd4;
                5: scale_at = 8'sd8;
                6: scale_at = 8'sd12;
                7: scale_at = 8'sd16;
                8: scale_at = -8'sd4;
                9: scale_at = 8'sd4;
                10: scale_at = 8'sd12;
                11: scale_at = -8'sd20;
                12: scale_at = -8'sd12;
                13: scale_at = -8'sd4;
                14: scale_at = 8'sd4;
                15: scale_at = 8'sd12;
                default: scale_at = '0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] activation_for_flat(input int flat_idx_i);
        begin
            case (flat_idx_i)
                0: activation_for_flat = -8'sd6;
                1: activation_for_flat = 8'sd1;
                2: activation_for_flat = 8'sd8;
                3: activation_for_flat = -8'sd2;
                4: activation_for_flat = 8'sd5;
                5: activation_for_flat = -8'sd5;
                6: activation_for_flat = 8'sd2;
                7: activation_for_flat = -8'sd8;
                8: activation_for_flat = -8'sd1;
                9: activation_for_flat = 8'sd6;
                10: activation_for_flat = -8'sd4;
                11: activation_for_flat = 8'sd3;
                12: activation_for_flat = -8'sd7;
                13: activation_for_flat = 8'sd0;
                14: activation_for_flat = 8'sd7;
                15: activation_for_flat = -8'sd3;
                16: activation_for_flat = 8'sd4;
                17: activation_for_flat = -8'sd6;
                18: activation_for_flat = 8'sd1;
                19: activation_for_flat = 8'sd8;
                20: activation_for_flat = -8'sd2;
                21: activation_for_flat = 8'sd5;
                22: activation_for_flat = -8'sd5;
                23: activation_for_flat = 8'sd2;
                24: activation_for_flat = -8'sd8;
                25: activation_for_flat = -8'sd1;
                26: activation_for_flat = 8'sd6;
                27: activation_for_flat = -8'sd4;
                28: activation_for_flat = 8'sd3;
                29: activation_for_flat = -8'sd7;
                30: activation_for_flat = 8'sd0;
                31: activation_for_flat = 8'sd7;
                32: activation_for_flat = -8'sd6;
                33: activation_for_flat = 8'sd1;
                34: activation_for_flat = 8'sd8;
                35: activation_for_flat = -8'sd2;
                36: activation_for_flat = 8'sd5;
                37: activation_for_flat = -8'sd5;
                38: activation_for_flat = 8'sd2;
                39: activation_for_flat = -8'sd8;
                40: activation_for_flat = -8'sd1;
                41: activation_for_flat = 8'sd6;
                42: activation_for_flat = -8'sd4;
                43: activation_for_flat = 8'sd3;
                44: activation_for_flat = -8'sd7;
                45: activation_for_flat = 8'sd0;
                46: activation_for_flat = 8'sd7;
                47: activation_for_flat = -8'sd3;
                48: activation_for_flat = 8'sd4;
                49: activation_for_flat = -8'sd6;
                50: activation_for_flat = 8'sd1;
                51: activation_for_flat = 8'sd8;
                52: activation_for_flat = -8'sd2;
                53: activation_for_flat = 8'sd5;
                54: activation_for_flat = -8'sd5;
                55: activation_for_flat = 8'sd2;
                56: activation_for_flat = -8'sd8;
                57: activation_for_flat = -8'sd1;
                58: activation_for_flat = 8'sd6;
                59: activation_for_flat = -8'sd4;
                60: activation_for_flat = 8'sd3;
                61: activation_for_flat = -8'sd7;
                62: activation_for_flat = 8'sd0;
                63: activation_for_flat = 8'sd7;
                default: activation_for_flat = '0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] zero_for_flat(input int flat_idx_i);
        begin
            case (flat_idx_i)
                0: zero_for_flat = -8'sd2;
                1: zero_for_flat = -8'sd2;
                2: zero_for_flat = -8'sd2;
                3: zero_for_flat = -8'sd2;
                4: zero_for_flat = 8'sd3;
                5: zero_for_flat = 8'sd3;
                6: zero_for_flat = 8'sd3;
                7: zero_for_flat = 8'sd3;
                8: zero_for_flat = 8'sd1;
                9: zero_for_flat = 8'sd1;
                10: zero_for_flat = 8'sd1;
                11: zero_for_flat = 8'sd1;
                12: zero_for_flat = -8'sd1;
                13: zero_for_flat = -8'sd1;
                14: zero_for_flat = -8'sd1;
                15: zero_for_flat = -8'sd1;
                16: zero_for_flat = -8'sd3;
                17: zero_for_flat = -8'sd3;
                18: zero_for_flat = -8'sd3;
                19: zero_for_flat = -8'sd3;
                20: zero_for_flat = 8'sd2;
                21: zero_for_flat = 8'sd2;
                22: zero_for_flat = 8'sd2;
                23: zero_for_flat = 8'sd2;
                24: zero_for_flat = 8'sd0;
                25: zero_for_flat = 8'sd0;
                26: zero_for_flat = 8'sd0;
                27: zero_for_flat = 8'sd0;
                28: zero_for_flat = -8'sd2;
                29: zero_for_flat = -8'sd2;
                30: zero_for_flat = -8'sd2;
                31: zero_for_flat = -8'sd2;
                32: zero_for_flat = 8'sd1;
                33: zero_for_flat = 8'sd1;
                34: zero_for_flat = 8'sd1;
                35: zero_for_flat = 8'sd1;
                36: zero_for_flat = -8'sd1;
                37: zero_for_flat = -8'sd1;
                38: zero_for_flat = -8'sd1;
                39: zero_for_flat = -8'sd1;
                40: zero_for_flat = -8'sd3;
                41: zero_for_flat = -8'sd3;
                42: zero_for_flat = -8'sd3;
                43: zero_for_flat = -8'sd3;
                44: zero_for_flat = 8'sd2;
                45: zero_for_flat = 8'sd2;
                46: zero_for_flat = 8'sd2;
                47: zero_for_flat = 8'sd2;
                48: zero_for_flat = 8'sd0;
                49: zero_for_flat = 8'sd0;
                50: zero_for_flat = 8'sd0;
                51: zero_for_flat = 8'sd0;
                52: zero_for_flat = -8'sd2;
                53: zero_for_flat = -8'sd2;
                54: zero_for_flat = -8'sd2;
                55: zero_for_flat = -8'sd2;
                56: zero_for_flat = 8'sd3;
                57: zero_for_flat = 8'sd3;
                58: zero_for_flat = 8'sd3;
                59: zero_for_flat = 8'sd3;
                60: zero_for_flat = 8'sd1;
                61: zero_for_flat = 8'sd1;
                62: zero_for_flat = 8'sd1;
                63: zero_for_flat = 8'sd1;
                default: zero_for_flat = '0;
            endcase
        end
    endfunction

    function automatic logic signed [7:0] scale_for_flat(input int flat_idx_i);
        begin
            case (flat_idx_i)
                0: scale_for_flat = -8'sd12;
                1: scale_for_flat = -8'sd12;
                2: scale_for_flat = -8'sd12;
                3: scale_for_flat = -8'sd12;
                4: scale_for_flat = -8'sd8;
                5: scale_for_flat = -8'sd8;
                6: scale_for_flat = -8'sd8;
                7: scale_for_flat = -8'sd8;
                8: scale_for_flat = -8'sd4;
                9: scale_for_flat = -8'sd4;
                10: scale_for_flat = -8'sd4;
                11: scale_for_flat = -8'sd4;
                12: scale_for_flat = 8'sd0;
                13: scale_for_flat = 8'sd0;
                14: scale_for_flat = 8'sd0;
                15: scale_for_flat = 8'sd0;
                16: scale_for_flat = 8'sd4;
                17: scale_for_flat = 8'sd4;
                18: scale_for_flat = 8'sd4;
                19: scale_for_flat = 8'sd4;
                20: scale_for_flat = 8'sd8;
                21: scale_for_flat = 8'sd8;
                22: scale_for_flat = 8'sd8;
                23: scale_for_flat = 8'sd8;
                24: scale_for_flat = 8'sd12;
                25: scale_for_flat = 8'sd12;
                26: scale_for_flat = 8'sd12;
                27: scale_for_flat = 8'sd12;
                28: scale_for_flat = 8'sd16;
                29: scale_for_flat = 8'sd16;
                30: scale_for_flat = 8'sd16;
                31: scale_for_flat = 8'sd16;
                32: scale_for_flat = -8'sd4;
                33: scale_for_flat = -8'sd4;
                34: scale_for_flat = -8'sd4;
                35: scale_for_flat = -8'sd4;
                36: scale_for_flat = 8'sd4;
                37: scale_for_flat = 8'sd4;
                38: scale_for_flat = 8'sd4;
                39: scale_for_flat = 8'sd4;
                40: scale_for_flat = 8'sd12;
                41: scale_for_flat = 8'sd12;
                42: scale_for_flat = 8'sd12;
                43: scale_for_flat = 8'sd12;
                44: scale_for_flat = -8'sd20;
                45: scale_for_flat = -8'sd20;
                46: scale_for_flat = -8'sd20;
                47: scale_for_flat = -8'sd20;
                48: scale_for_flat = -8'sd12;
                49: scale_for_flat = -8'sd12;
                50: scale_for_flat = -8'sd12;
                51: scale_for_flat = -8'sd12;
                52: scale_for_flat = -8'sd4;
                53: scale_for_flat = -8'sd4;
                54: scale_for_flat = -8'sd4;
                55: scale_for_flat = -8'sd4;
                56: scale_for_flat = 8'sd4;
                57: scale_for_flat = 8'sd4;
                58: scale_for_flat = 8'sd4;
                59: scale_for_flat = 8'sd4;
                60: scale_for_flat = 8'sd12;
                61: scale_for_flat = 8'sd12;
                62: scale_for_flat = 8'sd12;
                63: scale_for_flat = 8'sd12;
                default: scale_for_flat = '0;
            endcase
        end
    endfunction

    function automatic logic signed [OUT_WIDTH-1:0] product_at(input int flat_idx_i);
        int row_idx;
        int col_idx;
        int group_idx;
        logic signed [7:0] weight_s;
        logic signed [7:0] zero_s;
        logic signed [7:0] scale_s;
        logic signed [7:0] activation_s;
        logic signed [15:0] centered_s;
        logic signed [23:0] dequant_s;
        logic signed [31:0] product_s;
        begin
            row_idx = flat_idx_i / TILE_COLS;
            col_idx = flat_idx_i % TILE_COLS;
            group_idx = col_idx / GROUP_SIZE;
            weight_s = sign_extend_int4(packed_weight_r[flat_idx_i*4 +: 4]);
            zero_s = zero_at(row_idx, group_idx);
            scale_s = scale_at(row_idx, group_idx);
            activation_s = activation_at(col_idx);
            centered_s = $signed({{8{weight_s[7]}}, weight_s}) - $signed({{8{zero_s[7]}}, zero_s});
            dequant_s = centered_s * scale_s;
            product_s = dequant_s * activation_s;
            product_at = product_s[OUT_WIDTH-1:0];
        end
    endfunction

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            arvalid_r <= 1'b0;
            ar_ready_low_count_r <= '0;
            axi_rvalid_r <= 1'b0;
            axi_rbeat_idx_r <= '0;
            mem_word_r <= '0;
            beat_count_r <= '0;
            chunk_idx_r <= '0;
            payload_emit_count_r <= '0;
            payload_consume_count_r <= '0;
            payload_link_word_r <= '0;
            payload_link_valid_r <= 1'b0;
            payload_link_last_r <= 1'b0;
            current_beat_final_r <= 1'b0;
            accepted_beat_trace_r <= '0;
            rlast_trace_r <= '0;
            rid_error_trace_r <= '0;
            rresp_error_trace_r <= '0;
            rlast_error_trace_r <= '0;
            ready_low_trace_r <= '0;
            ready_low_seen_r <= '0;
            packed_weight_r <= '0;
            output_r <= '0;
            compute_idx_r <= '0;
            compute_row_r <= 1'b0;
            compute_weight_r <= '0;
            compute_zero_r <= '0;
            compute_scale_r <= '0;
            compute_activation_r <= '0;
            compute_centered_r <= '0;
            compute_dequant_r <= '0;
            compute_product_r <= '0;
        end else begin
            if (payload_link_valid_r && ready_low_required_w) begin
                ready_low_trace_r[int'(payload_consume_count_r)] <= 1'b1;
                ready_low_seen_r[int'(payload_consume_count_r)] <= 1'b1;
            end
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    arvalid_r <= 1'b0;
                    axi_rvalid_r <= 1'b0;
                    payload_link_valid_r <= 1'b0;
                    payload_link_last_r <= 1'b0;
                    if (start_i) begin
                        arvalid_r <= 1'b1;
                        ar_ready_low_count_r <= '0;
                        axi_rbeat_idx_r <= '0;
                        beat_count_r <= '0;
                        chunk_idx_r <= '0;
                        payload_emit_count_r <= '0;
                        payload_consume_count_r <= '0;
                        accepted_beat_trace_r <= '0;
                        rlast_trace_r <= '0;
                        rid_error_trace_r <= '0;
                        rresp_error_trace_r <= '0;
                        rlast_error_trace_r <= '0;
                        ready_low_trace_r <= '0;
                        ready_low_seen_r <= '0;
                        packed_weight_r <= '0;
                        output_r <= '0;
                        compute_idx_r <= '0;
                        compute_row_r <= 1'b0;
                        compute_weight_r <= '0;
                        compute_zero_r <= '0;
                        compute_scale_r <= '0;
                        compute_activation_r <= '0;
                        compute_centered_r <= '0;
                        compute_dequant_r <= '0;
                        compute_product_r <= '0;
                        state_r <= ISSUE_AR;
                    end
                end
                ISSUE_AR: begin
                    if (axi_arvalid_w && !axi_arready_w) begin
                        ar_ready_low_count_r <= ar_ready_low_count_r + 2'd1;
                    end else if (axi_arvalid_w && axi_arready_w) begin
                        arvalid_r <= 1'b0;
                        axi_rvalid_r <= 1'b1;
                        axi_rbeat_idx_r <= '0;
                        state_r <= R_WAIT;
                    end
                end
                R_WAIT: begin
                    payload_link_valid_r <= 1'b0;
                    payload_link_last_r <= 1'b0;
                    if (r_fire_w) begin
                        mem_word_r <= axi_rdata_w;
                        accepted_beat_trace_r[int'(beat_count_r)] <= 1'b1;
                        rlast_trace_r[int'(beat_count_r)] <= axi_rlast_w;
                        rid_error_trace_r[int'(beat_count_r)] <= (axi_rid_w != EXPECTED_AXI_ID);
                        rresp_error_trace_r[int'(beat_count_r)] <= (axi_rresp_w != 2'b00);
                        rlast_error_trace_r[int'(beat_count_r)] <= (axi_rlast_w != expected_rlast_w);
                        if (expected_rlast_w) begin
                            axi_rvalid_r <= 1'b0;
                        end else begin
                            axi_rvalid_r <= 1'b1;
                            axi_rbeat_idx_r <= axi_rbeat_idx_r + BEAT_IDX_W'(1);
                        end
                        payload_link_word_r <= axi_rdata_w[0 +: PAYLOAD_WIDTH];
                        payload_link_valid_r <= 1'b1;
                        payload_link_last_r <= expected_rlast_w && (LAST_CHUNK == '0);
                        payload_emit_count_r <= payload_emit_count_r + PAYLOAD_COUNT_W'(1);
                        current_beat_final_r <= expected_rlast_w;
                        chunk_idx_r <= '0;
                        beat_count_r <= beat_count_r + BEAT_IDX_W'(1);
                        state_r <= EMIT;
                    end
                end
                EMIT: begin
                    if (payload_fire_w) begin
                        packed_weight_r[int'(payload_consume_count_r)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH] <= payload_link_word_r;
                        payload_consume_count_r <= payload_consume_count_r + PAYLOAD_COUNT_W'(1);
                        if (final_payload_w) begin
                            payload_link_valid_r <= 1'b0;
                            payload_link_last_r <= 1'b0;
                            compute_idx_r <= '0;
                            state_r <= COMPUTE_LOAD;
                        end else if (last_chunk_w) begin
                            payload_link_valid_r <= 1'b0;
                            payload_link_last_r <= 1'b0;
                            chunk_idx_r <= '0;
                            state_r <= R_WAIT;
                        end else begin
                            chunk_idx_r <= next_chunk_idx_w;
                            payload_link_word_r <= mem_word_r[int'(next_chunk_idx_w)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH];
                            payload_link_last_r <= current_beat_final_r && (next_chunk_idx_w == LAST_CHUNK);
                            payload_emit_count_r <= payload_emit_count_r + PAYLOAD_COUNT_W'(1);
                        end
                    end
                end
                COMPUTE_LOAD: begin
                    compute_row_r <= compute_row_w;
                    compute_weight_r <= sign_extend_int4(packed_weight_r[int'(compute_idx_r)*4 +: 4]);
                    compute_zero_r <= zero_for_flat(int'(compute_idx_r));
                    compute_scale_r <= scale_for_flat(int'(compute_idx_r));
                    compute_activation_r <= activation_for_flat(int'(compute_idx_r));
                    state_r <= COMPUTE_CENTER;
                end
                COMPUTE_CENTER: begin
                    compute_centered_r <= $signed({{8{compute_weight_r[7]}}, compute_weight_r}) -
                        $signed({{8{compute_zero_r[7]}}, compute_zero_r});
                    state_r <= COMPUTE_DEQUANT;
                end
                COMPUTE_DEQUANT: begin
                    compute_dequant_r <= compute_centered_r * compute_scale_r;
                    state_r <= COMPUTE_PRODUCT;
                end
                COMPUTE_PRODUCT: begin
                    compute_product_r <= compute_dequant_r * compute_activation_r;
                    state_r <= COMPUTE_ACC;
                end
                COMPUTE_ACC: begin
                    output_r[int'(compute_row_r)*OUT_WIDTH +: OUT_WIDTH] <=
                        $signed(output_r[int'(compute_row_r)*OUT_WIDTH +: OUT_WIDTH]) + compute_product_r;
                    if (int'(compute_idx_r) == (TOTAL_WEIGHTS - 1)) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end else begin
                        compute_idx_r <= compute_idx_r + COMPUTE_IDX_W'(1);
                        state_r <= COMPUTE_LOAD;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        done_r <= 1'b0;
                        payload_link_valid_r <= 1'b0;
                        payload_link_last_r <= 1'b0;
                        state_r <= IDLE;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    arvalid_r <= 1'b0;
                    axi_rvalid_r <= 1'b0;
                    payload_link_valid_r <= 1'b0;
                    payload_link_last_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
