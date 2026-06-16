`timescale 1ns/1ps

module tb_token_loop_axi_attention_fixture;
    localparam int OUT_VALUES = 4;
    localparam int OUT_WIDTH = 16;
    localparam int STATUS_WIDTH = 96;
    localparam int TOP_STATUS_WIDTH = 128;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] output_vec;
    logic [STATUS_WIDTH-1:0] status_vec;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    logic top_start_prev_r;
    logic top_done_prev_r;
    logic token0_done_seen_while_start_high_r;
    logic token1_done_seen_while_start_high_r;
    logic token0_start_deasserted_after_done_r;
    logic token1_start_deasserted_after_done_r;
    logic token0_release_seen_r;
    logic token1_release_seen_r;
    logic token0_busy_seen_r;
    logic token1_busy_seen_r;
    integer active_token;
    integer token_start_count;
    integer token_done_count;
    integer token0_busy_cycles;
    integer token1_busy_cycles;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] token0_output_seen;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] token1_output_seen;
    logic [TOP_STATUS_WIDTH-1:0] token0_status_seen;
    logic [TOP_STATUS_WIDTH-1:0] token1_status_seen;

    logic held_write_seen;
    logic accepted_write_stable;
    logic [0:0] held_write_slot;
    logic signed [31:0] held_write_key;
    logic signed [31:0] held_write_value;
    logic signed [31:0] observed_write_key;
    logic signed [31:0] observed_write_value;
    logic signed [31:0] observed_key0;
    logic signed [31:0] observed_key1;
    logic signed [31:0] observed_value0;
    logic signed [31:0] observed_value1;
    logic [31:0] observed_emitted_payload [0:15];
    logic [31:0] observed_consumed_payload [0:15];
    logic [15:0] observed_ready_low_trace;
    logic payload_seen_pending;
    logic payload_hold_ok;
    integer emitted_count;
    integer consumed_count;
    integer accepted_r_count;
    integer rvalid_while_projection_not_ready_cycles;
    integer write_count;
    integer key_read_count;
    integer value_read_count;
    integer idx;
    integer observed;

    token_loop_axi_attention_fixture dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
        .output_o(output_vec),
        .status_o(status_vec)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    function automatic logic signed [7:0] lane_s8(input logic signed [31:0] packed_i, input int idx_i);
        begin
            lane_s8 = packed_i[idx_i*8 +: 8];
        end
    endfunction

    function automatic integer lane_s16(input logic signed [OUT_VALUES*OUT_WIDTH-1:0] packed_i, input int idx_i);
        logic signed [OUT_WIDTH-1:0] tmp;
        begin
            tmp = packed_i[idx_i*OUT_WIDTH +: OUT_WIDTH];
            lane_s16 = int'($signed(tmp));
        end
    endfunction

    function automatic logic signed [7:0] sign_extend_int4(input logic [3:0] nibble_i);
        begin
            sign_extend_int4 = { {4{nibble_i[3]}}, nibble_i };
        end
    endfunction

    function automatic logic signed [7:0] expected_axi_weight_at(input int idx_i);
        begin
            case (idx_i)
                0: expected_axi_weight_at = -8'sd5;
                1: expected_axi_weight_at = 8'sd0;
                2: expected_axi_weight_at = 8'sd5;
                3: expected_axi_weight_at = -8'sd6;
                4: expected_axi_weight_at = -8'sd1;
                5: expected_axi_weight_at = 8'sd4;
                6: expected_axi_weight_at = -8'sd7;
                7: expected_axi_weight_at = -8'sd2;
                8: expected_axi_weight_at = 8'sd3;
                9: expected_axi_weight_at = -8'sd8;
                10: expected_axi_weight_at = -8'sd3;
                11: expected_axi_weight_at = 8'sd2;
                12: expected_axi_weight_at = 8'sd7;
                13: expected_axi_weight_at = -8'sd4;
                14: expected_axi_weight_at = 8'sd1;
                15: expected_axi_weight_at = 8'sd6;
                16: expected_axi_weight_at = -8'sd5;
                17: expected_axi_weight_at = 8'sd0;
                18: expected_axi_weight_at = 8'sd5;
                19: expected_axi_weight_at = -8'sd6;
                20: expected_axi_weight_at = -8'sd1;
                21: expected_axi_weight_at = 8'sd4;
                22: expected_axi_weight_at = -8'sd7;
                23: expected_axi_weight_at = -8'sd2;
                24: expected_axi_weight_at = 8'sd3;
                25: expected_axi_weight_at = -8'sd8;
                26: expected_axi_weight_at = -8'sd3;
                27: expected_axi_weight_at = 8'sd2;
                28: expected_axi_weight_at = 8'sd7;
                29: expected_axi_weight_at = -8'sd4;
                30: expected_axi_weight_at = 8'sd1;
                31: expected_axi_weight_at = 8'sd6;
                32: expected_axi_weight_at = -8'sd5;
                33: expected_axi_weight_at = 8'sd0;
                34: expected_axi_weight_at = 8'sd5;
                35: expected_axi_weight_at = -8'sd6;
                36: expected_axi_weight_at = -8'sd1;
                37: expected_axi_weight_at = 8'sd4;
                38: expected_axi_weight_at = -8'sd7;
                39: expected_axi_weight_at = -8'sd2;
                40: expected_axi_weight_at = 8'sd3;
                41: expected_axi_weight_at = -8'sd8;
                42: expected_axi_weight_at = -8'sd3;
                43: expected_axi_weight_at = 8'sd2;
                44: expected_axi_weight_at = 8'sd7;
                45: expected_axi_weight_at = -8'sd4;
                46: expected_axi_weight_at = 8'sd1;
                47: expected_axi_weight_at = 8'sd6;
                48: expected_axi_weight_at = -8'sd5;
                49: expected_axi_weight_at = 8'sd0;
                50: expected_axi_weight_at = 8'sd5;
                51: expected_axi_weight_at = -8'sd6;
                52: expected_axi_weight_at = -8'sd1;
                53: expected_axi_weight_at = 8'sd4;
                54: expected_axi_weight_at = -8'sd7;
                55: expected_axi_weight_at = -8'sd2;
                56: expected_axi_weight_at = 8'sd3;
                57: expected_axi_weight_at = -8'sd8;
                58: expected_axi_weight_at = -8'sd3;
                59: expected_axi_weight_at = 8'sd2;
                60: expected_axi_weight_at = 8'sd7;
                61: expected_axi_weight_at = -8'sd4;
                62: expected_axi_weight_at = 8'sd1;
                63: expected_axi_weight_at = 8'sd6;
                default: expected_axi_weight_at = '0;
            endcase
        end
    endfunction

    function automatic logic [31:0] expected_axi_payload_at(input int idx_i);
        begin
            case (idx_i % 8)
                0: expected_axi_payload_at = 32'he94fa50b;
                1: expected_axi_payload_at = 32'h61c72d83;
                2: expected_axi_payload_at = 32'he94fa50b;
                3: expected_axi_payload_at = 32'h61c72d83;
                4: expected_axi_payload_at = 32'he94fa50b;
                5: expected_axi_payload_at = 32'h61c72d83;
                6: expected_axi_payload_at = 32'he94fa50b;
                7: expected_axi_payload_at = 32'h61c72d83;
                default: expected_axi_payload_at = '0;
            endcase
        end
    endfunction

    always @(posedge aclk) begin
        if (!aresetn) begin
            top_start_prev_r <= 1'b0;
            top_done_prev_r <= 1'b0;
            token0_done_seen_while_start_high_r <= 1'b0;
            token1_done_seen_while_start_high_r <= 1'b0;
            token0_start_deasserted_after_done_r <= 1'b0;
            token1_start_deasserted_after_done_r <= 1'b0;
            token0_release_seen_r <= 1'b0;
            token1_release_seen_r <= 1'b0;
            token0_busy_seen_r <= 1'b0;
            token1_busy_seen_r <= 1'b0;
            active_token <= -1;
            token_start_count <= 0;
            token_done_count <= 0;
            token0_busy_cycles <= 0;
            token1_busy_cycles <= 0;
            token0_output_seen <= '0;
            token1_output_seen <= '0;
            token0_status_seen <= '0;
            token1_status_seen <= '0;
        end else begin
            if (dut.top_start_r && !top_start_prev_r) begin
                active_token <= token_start_count;
                token_start_count <= token_start_count + 1;
            end
            if (dut.top_start_r && !dut.top_done_w) begin
                if (active_token == 0) begin
                    token0_busy_seen_r <= 1'b1;
                    token0_busy_cycles <= token0_busy_cycles + 1;
                end else if (active_token == 1) begin
                    token1_busy_seen_r <= 1'b1;
                    token1_busy_cycles <= token1_busy_cycles + 1;
                end
            end
            if (active_token == 0 && token0_busy_seen_r && !token0_done_seen_while_start_high_r && !dut.top_done_w && !dut.top_start_r) begin
                $display("FAIL token_loop_axi_attention_fixture token0 child start was not held while top child busy");
                $fatal;
            end
            if (active_token == 1 && token1_busy_seen_r && !token1_done_seen_while_start_high_r && !dut.top_done_w && !dut.top_start_r) begin
                $display("FAIL token_loop_axi_attention_fixture token1 child start was not held while top child busy");
                $fatal;
            end
            if (dut.top_start_r && dut.top_done_w) begin
                token_done_count <= token_done_count + 1;
                if (active_token == 0) begin
                    token0_done_seen_while_start_high_r <= 1'b1;
                    token0_output_seen <= dut.top_output_w;
                    token0_status_seen <= dut.top_status_w;
                end else if (active_token == 1) begin
                    token1_done_seen_while_start_high_r <= 1'b1;
                    token1_output_seen <= dut.top_output_w;
                    token1_status_seen <= dut.top_status_w;
                end
            end
            if (top_done_prev_r && !dut.top_start_r) begin
                if (active_token == 0) begin
                    token0_start_deasserted_after_done_r <= 1'b1;
                end else if (active_token == 1) begin
                    token1_start_deasserted_after_done_r <= 1'b1;
                end
            end
            if (top_done_prev_r && dut.top_start_r) begin
                $display("FAIL token_loop_axi_attention_fixture child start was not deasserted after top child done_o");
                $fatal;
            end
            if (token0_start_deasserted_after_done_r && active_token == 0 && !dut.top_done_w) begin
                token0_release_seen_r <= 1'b1;
            end
            if (token1_start_deasserted_after_done_r && active_token == 1 && !dut.top_done_w) begin
                token1_release_seen_r <= 1'b1;
            end
            top_start_prev_r <= dut.top_start_r;
            top_done_prev_r <= dut.top_done_w;
        end
    end

    always @(negedge aclk) begin
        if (!aresetn || !start_i) begin
            payload_seen_pending = 1'b0;
        end else begin
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rvalid_w &&
                !dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rready_w &&
                dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_valid_r) begin
                rvalid_while_projection_not_ready_cycles = rvalid_while_projection_not_ready_cycles + 1;
            end
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_valid_r) begin
                if (!dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_ready_w && consumed_count < 16) begin
                    observed_ready_low_trace[consumed_count] = 1'b1;
                    if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r != expected_axi_payload_at(consumed_count)) begin
                        payload_hold_ok = 1'b0;
                    end
                end
                if (!payload_seen_pending) begin
                    if (emitted_count >= 16) begin
                        $display("FAIL token_loop_axi_attention_fixture too many emitted AXI payloads");
                        $fatal;
                    end
                    observed_emitted_payload[emitted_count] = dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r;
                    emitted_count = emitted_count + 1;
                    payload_seen_pending = 1'b1;
                end
                if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_ready_w) begin
                    if (consumed_count >= 16) begin
                        $display("FAIL token_loop_axi_attention_fixture too many consumed AXI payloads");
                        $fatal;
                    end
                    observed_consumed_payload[consumed_count] = dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r;
                    consumed_count = consumed_count + 1;
                    payload_seen_pending = 1'b0;
                end
            end else begin
                payload_seen_pending = 1'b0;
            end
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            held_write_seen <= 1'b0;
            accepted_write_stable <= 1'b0;
            held_write_slot <= '0;
            held_write_key <= '0;
            held_write_value <= '0;
            observed_write_key <= '0;
            observed_write_value <= '0;
            observed_key0 <= '0;
            observed_key1 <= '0;
            observed_value0 <= '0;
            observed_value1 <= '0;
            accepted_r_count <= 0;
            write_count <= 0;
            key_read_count <= 0;
            value_read_count <= 0;
        end else begin
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.r_fire_w) begin
                accepted_r_count <= accepted_r_count + 1;
                if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rid_w != 8'h02 ||
                    dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rresp_w != 2'b00 ||
                    dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rlast_w != (accepted_r_count[0] == 1'b1)) begin
                    $display("FAIL token_loop_axi_attention_fixture AXI R metadata beat=%0d", accepted_r_count);
                    $fatal;
                end
            end
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_valid_r && !dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_accept_r) begin
                held_write_seen <= 1'b1;
                held_write_slot <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r;
                held_write_key <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r;
                held_write_value <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r;
            end
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_valid_r && dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_accept_r) begin
                write_count <= write_count + 1;
                if (!held_write_seen || held_write_slot != dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r || held_write_key != dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r || held_write_value != dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r) begin
                    $display("FAIL token_loop_axi_attention_fixture attention write fields changed before accept");
                    $fatal;
                end
                accepted_write_stable <= 1'b1;
                observed_write_key <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r;
                observed_write_value <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r;
            end
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_valid_r) begin
                key_read_count <= key_read_count + 1;
                if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_slot_r == 1'b0) begin
                    observed_key0 <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_data_r;
                end else begin
                    observed_key1 <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_data_r;
                end
            end
            if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_valid_r) begin
                value_read_count <= value_read_count + 1;
                if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_slot_r == 1'b0) begin
                    observed_value0 <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_data_r;
                end else begin
                    observed_value1 <= dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_data_r;
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        stable_passed = 1'b0;
        emitted_count = 0;
        consumed_count = 0;
        rvalid_while_projection_not_ready_cycles = 0;
        observed_ready_low_trace = '0;
        payload_seen_pending = 1'b0;
        payload_hold_ok = 1'b1;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #24000;
        if (!done_o) begin
            $display("FAIL token_loop_axi_attention_fixture done_o was not asserted");
            $fatal;
        end
        if (status_vec[31:0] != 32'h64636261) begin
            $display("FAIL token_loop_axi_attention_fixture token trace observed=0x%0h expected=0x64636261", status_vec[31:0]);
            $fatal;
        end
        if (status_vec[32 +: 16] != 16'h5453) begin
            $display("FAIL token_loop_axi_attention_fixture final top trace observed=0x%0h expected=0x5453", status_vec[32 +: 16]);
            $fatal;
        end
        if (status_vec[48 +: 16] != 16'h4241) begin
            $display("FAIL token_loop_axi_attention_fixture final layer trace observed=0x%0h expected=0x4241", status_vec[48 +: 16]);
            $fatal;
        end
        if (status_vec[64 +: 4] != 4'hf || status_vec[64 +: 4] != token1_status_seen[95:92]) begin
            $display("FAIL token_loop_axi_attention_fixture compact AXI metadata token=0x%0h top=0x%0h status=0x%0h", status_vec[64 +: 4], token1_status_seen[95:92], status_vec);
            $fatal;
        end
        if (token_start_count != 2 || token_done_count != 2) begin
            $display("FAIL token_loop_axi_attention_fixture token call counts start=%0d done=%0d", token_start_count, token_done_count);
            $fatal;
        end
        if (!token0_done_seen_while_start_high_r || !token1_done_seen_while_start_high_r || !token0_start_deasserted_after_done_r || !token1_start_deasserted_after_done_r || !token0_release_seen_r || !token1_release_seen_r || token0_busy_cycles == 0 || token1_busy_cycles == 0) begin
            $display("FAIL token_loop_axi_attention_fixture child start protocol token0_busy=%0d token0_done_seen=%0d token0_deassert=%0d token0_release=%0d token1_busy=%0d token1_done_seen=%0d token1_deassert=%0d token1_release=%0d", token0_busy_cycles, token0_done_seen_while_start_high_r, token0_start_deasserted_after_done_r, token0_release_seen_r, token1_busy_cycles, token1_done_seen_while_start_high_r, token1_start_deasserted_after_done_r, token1_release_seen_r);
            $fatal;
        end
        if (token0_status_seen[15:0] != 16'h5453 || token1_status_seen[15:0] != 16'h5453) begin
            $display("FAIL token_loop_axi_attention_fixture captured top traces token0=0x%0h token1=0x%0h", token0_status_seen[15:0], token1_status_seen[15:0]);
            $fatal;
        end
        if (token0_status_seen[16 +: 16] != 16'h4241 || token1_status_seen[16 +: 16] != 16'h4241) begin
            $display("FAIL token_loop_axi_attention_fixture captured layer traces token0=0x%0h token1=0x%0h", token0_status_seen[16 +: 16], token1_status_seen[16 +: 16]);
            $fatal;
        end
        if (token0_status_seen[32 +: 48] != 48'h323122211211 || token1_status_seen[32 +: 48] != 48'h323122211211) begin
            $display("FAIL token_loop_axi_attention_fixture captured child traces token0=0x%0h token1=0x%0h", token0_status_seen[32 +: 48], token1_status_seen[32 +: 48]);
            $fatal;
        end
        if (token0_status_seen[95:92] != 4'hf || token1_status_seen[95:92] != 4'hf) begin
            $display("FAIL token_loop_axi_attention_fixture captured top AXI metadata token0=0x%0h token1=0x%0h", token0_status_seen[95:92], token1_status_seen[95:92]);
            $fatal;
        end
        if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.status_o[79:76] != 4'hf ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42] != 4'hf) begin
            $display("FAIL token_loop_axi_attention_fixture nested AXI metadata layer=0x%0h nested=0x%0h", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.status_o[79:76], dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42]);
            $fatal;
        end
        if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w != 32'h00120000 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w != 8'h01 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w != 3'h4 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w != 2'b01 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w != 8'h02) begin
            $display("FAIL token_loop_axi_attention_fixture AXI AR fields addr=0x%0h", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w);
            $fatal;
        end
        if (accepted_r_count != 4 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r != 2'h3 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r != 2'h2 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rid_error_trace_r != '0 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rresp_error_trace_r != '0 ||
            dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_error_trace_r != '0) begin
            $display("FAIL token_loop_axi_attention_fixture AXI metadata accepted=%0d trace=0x%0h status=0x%0h", accepted_r_count, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
            $fatal;
        end
        if (emitted_count != 16 || consumed_count != 16) begin
            $display("FAIL token_loop_axi_attention_fixture AXI payload counts emitted=%0d consumed=%0d", emitted_count, consumed_count);
            $fatal;
        end
        for (idx = 0; idx < 16; idx = idx + 1) begin
            if (observed_emitted_payload[idx] != observed_consumed_payload[idx] || observed_emitted_payload[idx] != expected_axi_payload_at(idx)) begin
                $display("FAIL token_loop_axi_attention_fixture AXI payload mismatch[%0d] emitted=0x%0h consumed=0x%0h expected=0x%0h", idx, observed_emitted_payload[idx], observed_consumed_payload[idx], expected_axi_payload_at(idx));
                $fatal;
            end
        end
        if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r != 8'h19 || !payload_hold_ok || rvalid_while_projection_not_ready_cycles < 2) begin
            $display("FAIL token_loop_axi_attention_fixture AXI backpressure dut=0x%0h hold=%0d rstall=%0d observed=0x%0h", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r, payload_hold_ok, rvalid_while_projection_not_ready_cycles, observed_ready_low_trace);
            $fatal;
        end
        for (idx = 0; idx < 64; idx = idx + 1) begin
            if (sign_extend_int4(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) != expected_axi_weight_at(idx)) begin
                $display("FAIL token_loop_axi_attention_fixture AXI round-trip[%0d]=%0d expected=%0d", idx, sign_extend_int4(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]), expected_axi_weight_at(idx));
                $fatal;
            end
        end
        if (dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o != 64'h00000770000001e4) begin
            $display("FAIL token_loop_axi_attention_fixture AXI projection output=0x%0h", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o);
            $fatal;
        end
        if (write_count != 2 || !accepted_write_stable || dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r != 1'b1) begin
            $display("FAIL token_loop_axi_attention_fixture attention write count=%0d stable=%0d slot=%0d", write_count, accepted_write_stable, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r);
            $fatal;
        end
        if (key_read_count != 4 || value_read_count != 4) begin
            $display("FAIL token_loop_axi_attention_fixture attention read counts key=%0d value=%0d", key_read_count, value_read_count);
            $fatal;
        end
        if (!dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_valid_r || !dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_valid_r || dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r != 25 || dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r != -14) begin
            $display("FAIL token_loop_axi_attention_fixture attention scores observed=%0d,%0d", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r);
            $fatal;
        end
        if (!dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.control_valid_r || dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r != 8'd12 || dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r != 8'd4) begin
            $display("FAIL token_loop_axi_attention_fixture attention weights observed=%0d,%0d", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r);
            $fatal;
        end
        observed = lane_s16(output_vec, 0);
        if (observed != 4) begin
            $display("FAIL token_loop_axi_attention_fixture output[%0d] observed=%0d expected=4", 0, observed);
            $fatal;
        end
        observed = lane_s16(output_vec, 1);
        if (observed != -2) begin
            $display("FAIL token_loop_axi_attention_fixture output[%0d] observed=%0d expected=-2", 1, observed);
            $fatal;
        end
        observed = lane_s16(output_vec, 2);
        if (observed != 1) begin
            $display("FAIL token_loop_axi_attention_fixture output[%0d] observed=%0d expected=1", 2, observed);
            $fatal;
        end
        observed = lane_s16(output_vec, 3);
        if (observed != 2) begin
            $display("FAIL token_loop_axi_attention_fixture output[%0d] observed=%0d expected=2", 3, observed);
            $fatal;
        end
        if (token0_output_seen != output_vec || token1_output_seen != output_vec) begin
            $display("FAIL token_loop_axi_attention_fixture per-token outputs changed token0=0x%0h token1=0x%0h final=0x%0h", token0_output_seen, token1_output_seen, output_vec);
            $fatal;
        end
        stable_output_snapshot = output_vec;
        stable_status_snapshot = status_vec;
        #20;
        if (!done_o || output_vec != stable_output_snapshot || status_vec != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL token_loop_axi_attention_fixture output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("TOKEN_LOOP_TRACE token_loop_axi_attention_fixture trace_hex=0x%0h events=token0_start,token0_done,token1_start,token1_done", status_vec[31:0]);
        $display("TOKEN_CHILD_CALL_TRACE token_loop_axi_attention_fixture token=0 top_trace_hex=0x%0h layer_trace_hex=0x%0h child_trace_hex=0x%0h axi_bits=0x%0h output=%0d,%0d,%0d,%0d", token0_status_seen[15:0], token0_status_seen[16 +: 16], token0_status_seen[32 +: 48], token0_status_seen[95:92], lane_s16(token0_output_seen, 0), lane_s16(token0_output_seen, 1), lane_s16(token0_output_seen, 2), lane_s16(token0_output_seen, 3));
        $display("TOKEN_CHILD_CALL_TRACE token_loop_axi_attention_fixture token=1 top_trace_hex=0x%0h layer_trace_hex=0x%0h child_trace_hex=0x%0h axi_bits=0x%0h output=%0d,%0d,%0d,%0d", token1_status_seen[15:0], token1_status_seen[16 +: 16], token1_status_seen[32 +: 48], token1_status_seen[95:92], lane_s16(token1_output_seen, 0), lane_s16(token1_output_seen, 1), lane_s16(token1_output_seen, 2), lane_s16(token1_output_seen, 3));
        $display("TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_attention_fixture token=0 busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", token0_busy_cycles, token0_done_seen_while_start_high_r, token0_start_deasserted_after_done_r, token0_release_seen_r);
        $display("TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_attention_fixture token=1 busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", token1_busy_cycles, token1_done_seen_while_start_high_r, token1_start_deasserted_after_done_r, token1_release_seen_r);
        $display("TOP_AXI_ATTENTION_TRACE token_loop_axi_attention_fixture token=1 top_trace_hex=0x%0h layer_trace_hex=0x%0h child_trace_hex=0x%0h events=layer_fsm_axi_attention_fixture_start,layer_fsm_axi_attention_fixture_done", status_vec[32 +: 16], status_vec[48 +: 16], token1_status_seen[32 +: 48]);
        $display("LAYER_AXI_ATTENTION_TRACE token_loop_axi_attention_fixture token=1 layer_trace_hex=0x%0h child_trace_hex=0x%0h events=decoder_child_axi_attention_datapath_start,decoder_child_axi_attention_datapath_done", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.status_o[15:0], dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.status_o[16 +: 48]);
        $display("CHILD_TRACE token_loop_axi_attention_fixture token=1 trace_hex=0x%0h events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.status_o[47:0]);
        $display("AXI_PROJECTION_AR_TRACE token_loop_axi_attention_fixture addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_R_METADATA_TRACE token_loop_axi_attention_fixture accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[42], dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[43], dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[44]);
        $write("AXI_PROJECTION_EMITTED_PAYLOADS token_loop_axi_attention_fixture");
        for (idx = 8; idx < 16; idx = idx + 1) begin
            $write(" 0x%08h", observed_emitted_payload[idx]);
        end
        $write("\n");
        $write("AXI_PROJECTION_CONSUMED_PAYLOADS token_loop_axi_attention_fixture");
        for (idx = 8; idx < 16; idx = idx + 1) begin
            $write(" 0x%08h", observed_consumed_payload[idx]);
        end
        $write("\n");
        $display("AXI_PROJECTION_PAYLOAD_TRACE token_loop_axi_attention_fixture emitted=%0d consumed=%0d payload_match=%0d first=0x%0h last=0x%0h", emitted_count, consumed_count, payload_hold_ok && observed_emitted_payload[0] == observed_consumed_payload[0] && observed_emitted_payload[15] == observed_consumed_payload[15], observed_emitted_payload[0], observed_consumed_payload[15]);
        $display("AXI_PROJECTION_BACKPRESSURE_TRACE token_loop_axi_attention_fixture ready_low_payload_idx=0,3,4 trace=0x%0h payload_hold_ok=%0d", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r, payload_hold_ok);
        $display("AXI_PROJECTION_OUTPUT_TRACE token_loop_axi_attention_fixture output=%0d,%0d", $signed(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]));
        $display("AXI_PROJECTION_ROUND_TRIP_TRACE token_loop_axi_attention_fixture packed_bytes=32 unpacked_values=64 round_trip_passed=1");
        $display("TOKEN_AXI_METADATA_PROPAGATION_TRACE token_loop_axi_attention_fixture token_bits=0x%0h token_bit_lsb=64 token_bit_msb=67 top_bits=0x%0h layer_bits=0x%0h nested_bits=0x%0h source=top_fsm_axi_attention_fixture.status_o[95:92]:layer_fsm_axi_attention_fixture.status_o[79:76]:decoder_child_axi_attention_datapath.status_o[63:60]:projection_axi_stream_integration.integration_status_o[45:42] status=0x%0h", status_vec[64 +: 4], token1_status_seen[95:92], dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.status_o[79:76], dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], status_vec);
        $display("CACHE_WRITE_TRACE token_loop_axi_attention_fixture count=%0d slot=%0d key=%0d,%0d,%0d,%0d value=%0d,%0d,%0d,%0d stable=%0d", write_count, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r, lane_s8(observed_write_key, 0), lane_s8(observed_write_key, 1), lane_s8(observed_write_key, 2), lane_s8(observed_write_key, 3), lane_s8(observed_write_value, 0), lane_s8(observed_write_value, 1), lane_s8(observed_write_value, 2), lane_s8(observed_write_value, 3), accepted_write_stable);
        $display("KEY_READ_TRACE token_loop_axi_attention_fixture slots=0,1 keys=%0d,%0d,%0d,%0d|%0d,%0d,%0d,%0d", lane_s8(observed_key0, 0), lane_s8(observed_key0, 1), lane_s8(observed_key0, 2), lane_s8(observed_key0, 3), lane_s8(observed_key1, 0), lane_s8(observed_key1, 1), lane_s8(observed_key1, 2), lane_s8(observed_key1, 3));
        $display("VALUE_READ_TRACE token_loop_axi_attention_fixture slots=0,1 values=%0d,%0d,%0d,%0d|%0d,%0d,%0d,%0d", lane_s8(observed_value0, 0), lane_s8(observed_value0, 1), lane_s8(observed_value0, 2), lane_s8(observed_value0, 3), lane_s8(observed_value1, 0), lane_s8(observed_value1, 1), lane_s8(observed_value1, 2), lane_s8(observed_value1, 3));
        $display("SCORE_TRACE token_loop_axi_attention_fixture scores=%0d,%0d", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r);
        $display("SOFTMAX_CONTROL_TRACE token_loop_axi_attention_fixture policy=two_score_winner_loser_q0_4 weights=%0d,%0d exp=0", dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r, dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r);
        $display("ATTENTION_OUTPUT_TRACE token_loop_axi_attention_fixture output=%0d,%0d,%0d,%0d", lane_s16(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 0), lane_s16(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 1), lane_s16(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 2), lane_s16(dut.u_top_fsm_axi_attention_fixture.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 3));
        $display("FINAL_OUTPUT_TRACE token_loop_axi_attention_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(output_vec, 0), lane_s16(output_vec, 1), lane_s16(output_vec, 2), lane_s16(output_vec, 3), status_vec);
        $display("TOKEN_LOOP_STABILITY_TRACE token_loop_axi_attention_fixture stable=%0d", stable_passed);
        $display("TOKEN_OUTPUT_POLICY_TRACE token_loop_axi_attention_fixture repeated_deterministic_outputs=%0d token_dependent_outputs=0", token0_output_seen == token1_output_seen && token1_output_seen == output_vec);
        $display("COMPACT_IO_TRACE token_loop_axi_attention_fixture estimated_iob_bits=164 prior_top_fsm_bonded_iob=196 prior_top_fsm_status_bits=128 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_kv_arrays=0 exposed_axi_debug=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL token_loop_axi_attention_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS token_loop_axi_attention_fixture");
        $finish;
    end
endmodule
