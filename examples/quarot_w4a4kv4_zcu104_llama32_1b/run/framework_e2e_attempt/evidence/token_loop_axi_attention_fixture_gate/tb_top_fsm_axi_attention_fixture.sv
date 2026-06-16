`timescale 1ns/1ps

module tb_top_fsm_axi_attention_fixture;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [4*16-1:0] output_vec;
    logic [127:0] status_vec;
    logic signed [4*16-1:0] stable_output_snapshot;
    logic [127:0] stable_status_snapshot;
    logic stable_passed;
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
    logic [31:0] observed_emitted_payload [0:7];
    logic [31:0] observed_consumed_payload [0:7];
    logic [7:0] observed_ready_low_trace;
    logic payload_seen_pending;
    logic payload_hold_ok;
    integer emitted_count;
    integer consumed_count;
    integer accepted_r_count;
    integer ar_ready_low_cycles;
    integer rvalid_while_projection_not_ready_cycles;
    integer write_count;
    integer key_read_count;
    integer value_read_count;
    integer idx;
    integer observed;
    logic layer_busy_seen_r;
    logic layer_done_seen_while_start_high_r;
    logic layer_start_deasserted_after_done_r;
    logic layer_done_release_seen_after_start_deassert_r;
    logic layer_done_prev_r;
    integer layer_busy_cycles;

    top_fsm_axi_attention_fixture dut (
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

    function automatic logic signed [15:0] lane_s16(input logic signed [63:0] packed_i, input int idx_i);
        begin
            lane_s16 = packed_i[idx_i*16 +: 16];
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
            case (idx_i)
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
            layer_busy_seen_r <= 1'b0;
            layer_done_seen_while_start_high_r <= 1'b0;
            layer_start_deasserted_after_done_r <= 1'b0;
            layer_done_release_seen_after_start_deassert_r <= 1'b0;
            layer_done_prev_r <= 1'b0;
            layer_busy_cycles <= 0;
        end else begin
            if (dut.layer_start_r && !dut.layer_done_w) begin
                layer_busy_seen_r <= 1'b1;
                layer_busy_cycles <= layer_busy_cycles + 1;
            end
            if (layer_busy_seen_r && !layer_done_seen_while_start_high_r && !dut.layer_done_w && !dut.layer_start_r) begin
                $display("FAIL top_fsm_axi_attention_fixture layer start was not held while layer busy");
                $fatal;
            end
            if (dut.layer_start_r && dut.layer_done_w) begin
                layer_done_seen_while_start_high_r <= 1'b1;
            end
            if (layer_done_prev_r && !dut.layer_start_r) begin
                layer_start_deasserted_after_done_r <= 1'b1;
            end
            if (layer_done_prev_r && dut.layer_start_r) begin
                $display("FAIL top_fsm_axi_attention_fixture layer start was not deasserted after layer done_o");
                $fatal;
            end
            if (layer_start_deasserted_after_done_r && !dut.layer_start_r && !dut.layer_done_w) begin
                layer_done_release_seen_after_start_deassert_r <= 1'b1;
            end
            layer_done_prev_r <= dut.layer_done_w;
        end
    end

    always @(negedge aclk) begin
        if (!aresetn || !start_i) begin
            payload_seen_pending = 1'b0;
        end else begin
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arvalid_w && !dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arready_w) begin
                ar_ready_low_cycles = ar_ready_low_cycles + 1;
            end
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rvalid_w && !dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rready_w && dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_valid_r) begin
                rvalid_while_projection_not_ready_cycles = rvalid_while_projection_not_ready_cycles + 1;
            end
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_valid_r) begin
                if (!dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_ready_w && consumed_count < 8) begin
                    observed_ready_low_trace[consumed_count] = 1'b1;
                    if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r != expected_axi_payload_at(consumed_count)) begin
                        payload_hold_ok = 1'b0;
                    end
                end
                if (!payload_seen_pending) begin
                    if (emitted_count >= 8) begin
                        $display("FAIL top_fsm_axi_attention_fixture too many emitted AXI payloads");
                        $fatal;
                    end
                    observed_emitted_payload[emitted_count] = dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r;
                    emitted_count = emitted_count + 1;
                    payload_seen_pending = 1'b1;
                end
                if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_ready_w) begin
                    if (consumed_count >= 8) begin
                        $display("FAIL top_fsm_axi_attention_fixture too many consumed AXI payloads");
                        $fatal;
                    end
                    observed_consumed_payload[consumed_count] = dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r;
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
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.r_fire_w) begin
                accepted_r_count <= accepted_r_count + 1;
                if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rid_w != 8'h02 ||
                    dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rresp_w != 2'b00 ||
                    dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rlast_w != (accepted_r_count == 1)) begin
                    $display("FAIL top_fsm_axi_attention_fixture AXI R metadata beat=%0d", accepted_r_count);
                    $fatal;
                end
            end
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_valid_r && !dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_accept_r) begin
                held_write_seen <= 1'b1;
                held_write_slot <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r;
                held_write_key <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r;
                held_write_value <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r;
            end
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_valid_r && dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_accept_r) begin
                write_count <= write_count + 1;
                if (!held_write_seen || held_write_slot != dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r || held_write_key != dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r || held_write_value != dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r) begin
                    $display("FAIL top_fsm_axi_attention_fixture attention write fields changed before accept");
                    $fatal;
                end
                accepted_write_stable <= 1'b1;
                observed_write_key <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r;
                observed_write_value <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r;
            end
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_valid_r) begin
                key_read_count <= key_read_count + 1;
                if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_slot_r == 1'b0) begin
                    observed_key0 <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_data_r;
                end else begin
                    observed_key1 <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.key_read_data_r;
                end
            end
            if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_valid_r) begin
                value_read_count <= value_read_count + 1;
                if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_slot_r == 1'b0) begin
                    observed_value0 <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_data_r;
                end else begin
                    observed_value1 <= dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.value_read_data_r;
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
        ar_ready_low_cycles = 0;
        rvalid_while_projection_not_ready_cycles = 0;
        observed_ready_low_trace = '0;
        payload_seen_pending = 1'b0;
        payload_hold_ok = 1'b1;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #9000;
        if (!done_o) begin
            $display("FAIL top_fsm_axi_attention_fixture done_o was not asserted");
            $fatal;
        end
        if (status_vec[15:0] != 16'h5453) begin
            $display("FAIL top_fsm_axi_attention_fixture top trace observed=0x%0h expected=0x5453", status_vec[15:0]);
            $fatal;
        end
        if (status_vec[16 +: 16] != 16'h4241) begin
            $display("FAIL top_fsm_axi_attention_fixture layer trace observed=0x%0h expected=0x4241", status_vec[16 +: 16]);
            $fatal;
        end
        if (status_vec[32 +: 48] != 48'h323122211211) begin
            $display("FAIL top_fsm_axi_attention_fixture trace observed=0x%0h expected=0x323122211211", status_vec[32 +: 48]);
            $fatal;
        end
        if (status_vec[104 +: 8] != 8'hac) begin
            $display("FAIL top_fsm_axi_attention_fixture compact status=0x%0h", status_vec);
            $fatal;
        end
        if (!layer_done_seen_while_start_high_r || !layer_start_deasserted_after_done_r || !layer_done_release_seen_after_start_deassert_r || layer_busy_cycles == 0) begin
            $display("FAIL top_fsm_axi_attention_fixture layer start protocol busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d child_done_release_seen_after_start_deassert=%0d", layer_busy_cycles, layer_done_seen_while_start_high_r, layer_start_deasserted_after_done_r, layer_done_release_seen_after_start_deassert_r);
            $fatal;
        end
        if (status_vec[92 +: 4] != dut.layer_status_w[79:76] ||
            dut.layer_status_w[79:76] != dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42]) begin
            $display("FAIL top_fsm_axi_attention_fixture layer compact AXI metadata bits layer=0x%0h child=0x%0h nested=0x%0h status=0x%0h",
                     status_vec[92 +: 4], dut.layer_status_w[79:76], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], status_vec);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w != 32'h00120000 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w != 8'h01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w != 3'h4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w != 2'b01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w != 8'h02) begin
            $display("FAIL top_fsm_axi_attention_fixture AXI AR fields addr=0x%0h", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w);
            $fatal;
        end
        if (accepted_r_count != 2 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r != 2'h3 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r != 2'h2 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rid_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rresp_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42] != 4'hf) begin
            $display("FAIL top_fsm_axi_attention_fixture AXI metadata accepted=%0d trace=0x%0h status=0x%0h", accepted_r_count, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
            $fatal;
        end
        if (status_vec[92 +: 4] != 4'hf ||
            status_vec[92 +: 4] != (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42] &
                                  dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42] &
                                  dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42] &
                                  dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42])) begin
            $display("FAIL top_fsm_axi_attention_fixture parent compact AXI metadata bits parent=0x%0h q=0x%0h k=0x%0h v=0x%0h o=0x%0h status=0x%0h",
                     status_vec[92 +: 4],
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42],
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42],
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42],
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42],
                     status_vec);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_araddr_w != 32'h00120000 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arlen_w != 8'h01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arsize_w != 3'h4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arburst_w != 2'b01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arid_w != 8'h02 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_araddr_w != 32'h00120000 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arlen_w != 8'h01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arsize_w != 3'h4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arburst_w != 2'b01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arid_w != 8'h02 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_araddr_w != 32'h00120000 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arlen_w != 8'h01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arsize_w != 3'h4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arburst_w != 2'b01 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arid_w != 8'h02) begin
            $display("FAIL top_fsm_axi_attention_fixture k/v/o AXI AR fields k_addr=0x%0h v_addr=0x%0h o_addr=0x%0h",
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_araddr_w,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_araddr_w,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_araddr_w);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.accepted_beat_trace_r != 2'h3 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rlast_trace_r != 2'h2 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rid_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rresp_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rlast_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42] != 4'hf ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.accepted_beat_trace_r != 2'h3 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rlast_trace_r != 2'h2 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rid_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rresp_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rlast_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42] != 4'hf ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.accepted_beat_trace_r != 2'h3 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rlast_trace_r != 2'h2 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rid_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rresp_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rlast_error_trace_r != '0 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42] != 4'hf) begin
            $display("FAIL top_fsm_axi_attention_fixture k/v/o AXI metadata k=0x%0h v=0x%0h o=0x%0h",
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
            $fatal;
        end
        if (emitted_count != 8 || consumed_count != 8) begin
            $display("FAIL top_fsm_axi_attention_fixture AXI payload counts emitted=%0d consumed=%0d", emitted_count, consumed_count);
            $fatal;
        end
        for (idx = 0; idx < 8; idx = idx + 1) begin
            if (observed_emitted_payload[idx] != observed_consumed_payload[idx]) begin
                $display("FAIL top_fsm_axi_attention_fixture AXI payload mismatch[%0d]", idx);
                $fatal;
            end
        end
        if (observed_ready_low_trace != 8'h19 || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r != 8'h19 || !payload_hold_ok || rvalid_while_projection_not_ready_cycles < 1) begin
            $display("FAIL top_fsm_axi_attention_fixture AXI backpressure observed=0x%0h dut=0x%0h hold=%0d rstall=%0d", observed_ready_low_trace, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r, payload_hold_ok, rvalid_while_projection_not_ready_cycles);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.ready_low_trace_r != 8'h19 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.ready_low_trace_r != 8'h19 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.ready_low_trace_r != 8'h19) begin
            $display("FAIL top_fsm_axi_attention_fixture k/v/o AXI backpressure k=0x%0h v=0x%0h o=0x%0h",
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.ready_low_trace_r,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.ready_low_trace_r,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.ready_low_trace_r);
            $fatal;
        end
        for (idx = 0; idx < 64; idx = idx + 1) begin
            if (sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) != expected_axi_weight_at(idx) ||
                sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) != expected_axi_weight_at(idx) ||
                sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) != expected_axi_weight_at(idx) ||
                sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) != expected_axi_weight_at(idx)) begin
                $display("FAIL top_fsm_axi_attention_fixture AXI round-trip[%0d] q=%0d k=%0d v=%0d o=%0d expected=%0d",
                         idx,
                         sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]),
                         sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]),
                         sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]),
                         sign_extend_int4(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]),
                         expected_axi_weight_at(idx));
                $fatal;
            end
        end
        if (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o != 64'h00000770000001e4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o != 64'h00000770000001e4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o != 64'h00000770000001e4 ||
            dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o != 64'h00000770000001e4) begin
            $display("FAIL top_fsm_axi_attention_fixture AXI projection output q=0x%0h k=0x%0h v=0x%0h o=0x%0h",
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o,
                     dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o);
            $fatal;
        end
        if (write_count != 1 || !accepted_write_stable || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r != 1'b1) begin
            $display("FAIL top_fsm_axi_attention_fixture attention write count=%0d stable=%0d slot=%0d", write_count, accepted_write_stable, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r);
            $fatal;
        end
        if (key_read_count != 2 || value_read_count != 2) begin
            $display("FAIL top_fsm_axi_attention_fixture attention read counts key=%0d value=%0d", key_read_count, value_read_count);
            $fatal;
        end
        if (!dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_valid_r || !dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_valid_r || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r != 25 || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r != -14) begin
            $display("FAIL top_fsm_axi_attention_fixture attention scores observed=%0d,%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r);
            $fatal;
        end
        if (!dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.control_valid_r || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r != 8'd12 || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r != 8'd4) begin
            $display("FAIL top_fsm_axi_attention_fixture attention weights observed=%0d,%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r);
            $fatal;
        end
        if (!dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_valid_r || dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o != output_vec) begin
            $display("FAIL top_fsm_axi_attention_fixture attention child output mismatch child=0x%0h top=0x%0h", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, output_vec);
            $fatal;
        end
        observed = $signed({ {16{output_vec[0*16 + 15]}}, output_vec[0*16 +: 16] });
        if (observed != 4) begin
            $display("FAIL top_fsm_axi_attention_fixture output[%0d] observed=%0d expected=4", 0, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[1*16 + 15]}}, output_vec[1*16 +: 16] });
        if (observed != -2) begin
            $display("FAIL top_fsm_axi_attention_fixture output[%0d] observed=%0d expected=-2", 1, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[2*16 + 15]}}, output_vec[2*16 +: 16] });
        if (observed != 1) begin
            $display("FAIL top_fsm_axi_attention_fixture output[%0d] observed=%0d expected=1", 2, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[3*16 + 15]}}, output_vec[3*16 +: 16] });
        if (observed != 2) begin
            $display("FAIL top_fsm_axi_attention_fixture output[%0d] observed=%0d expected=2", 3, observed);
            $fatal;
        end
        stable_output_snapshot = output_vec;
        stable_status_snapshot = status_vec;
        #20;
        if (output_vec != stable_output_snapshot || status_vec != stable_status_snapshot || !done_o) begin
            stable_passed = 1'b0;
            $display("FAIL top_fsm_axi_attention_fixture output/status changed while top done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;
        $display("TOP_AXI_ATTENTION_TRACE top_fsm_axi_attention_fixture top_trace_hex=0x%0h layer_trace_hex=0x%0h child_trace_hex=0x%0h events=layer_fsm_axi_attention_fixture_start,layer_fsm_axi_attention_fixture_done", status_vec[15:0], status_vec[16 +: 16], status_vec[32 +: 48]);
        $display("LAYER_AXI_ATTENTION_TRACE top_fsm_axi_attention_fixture layer_trace_hex=0x%0h child_trace_hex=0x%0h events=decoder_child_axi_attention_datapath_start,decoder_child_axi_attention_datapath_done", status_vec[16 +: 16], status_vec[32 +: 48]);
        $display("CHILD_TRACE top_fsm_axi_attention_fixture trace_hex=0x%0h events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done", status_vec[32 +: 48]);
        $display("RMS_LOOKUP_TRACE top_fsm_axi_attention_fixture selector=%0d valid=%0d inv_rms=%0d sumsq=%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[2:1], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[0], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[18:3], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[34:19]);
        $display("ROPE_LOOKUP_TRACE top_fsm_axi_attention_fixture position=%0d pair=0 valid=%0d cos=%0d sin=%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[39:36], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[35], $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[48:41]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[56:49]));
        $display("ROPE_LOOKUP_TRACE top_fsm_axi_attention_fixture position=%0d pair=1 valid=%0d cos=%0d sin=%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[61:58], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[57], $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[70:63]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_rmsnorm_rope_source_path.status_o[78:71]));
        $display("AXI_PROJECTION_AR_TRACE top_fsm_axi_attention_fixture addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w, ar_ready_low_cycles);
        $display("AXI_PROJECTION_R_METADATA_TRACE top_fsm_axi_attention_fixture accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r, (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_attention_fixture projection=q addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=%0d instance=u_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w, ar_ready_low_cycles);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_attention_fixture projection=k addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_k_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_attention_fixture projection=v addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_v_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_attention_fixture projection=o addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_o_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_attention_fixture projection=q accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r, (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_attention_fixture projection=k accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_k_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rlast_trace_r, (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_attention_fixture projection=v accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_v_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rlast_trace_r, (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_attention_fixture projection=o accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_o_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rlast_trace_r, (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("PARENT_AXI_METADATA_PROPAGATION_TRACE top_fsm_axi_attention_fixture parent_bits=0x%0h parent_mask=0x00000000f00000000000000000000000 parent_bit_lsb=92 parent_bit_msb=95 child_status_bits=0x%0h q_bits=0x%0h k_bits=0x%0h v_bits=0x%0h o_bits=0x%0h source=q_projection_axi_stream_integration.integration_status_o[45:42]:k_projection_axi_stream_integration.integration_status_o[45:42]:v_projection_axi_stream_integration.integration_status_o[45:42]:o_projection_axi_stream_integration.integration_status_o[45:42]:aggregate_and status=0x%0h", status_vec[92 +: 4], (dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42] & dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42] & dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42] & dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42]), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42], status_vec);
        $write("AXI_PROJECTION_EMITTED_PAYLOADS top_fsm_axi_attention_fixture");
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $write(" 0x%08h", observed_emitted_payload[idx]);
        end
        $write("\n");
        $write("AXI_PROJECTION_CONSUMED_PAYLOADS top_fsm_axi_attention_fixture");
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $write(" 0x%08h", observed_consumed_payload[idx]);
        end
        $write("\n");
        $display("AXI_PROJECTION_BACKPRESSURE_TRACE top_fsm_axi_attention_fixture ready_low_payload_idx=0,3,4 trace=0x%0h payload_hold_ok=%0d rvalid_while_projection_not_ready_cycles=%0d", observed_ready_low_trace, payload_hold_ok, rvalid_while_projection_not_ready_cycles);
        $display("AXI_PROJECTION_OUTPUT_TRACE top_fsm_axi_attention_fixture output=%0d,%0d status=0x%0h", $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_ROUND_TRIP_TRACE top_fsm_axi_attention_fixture packed_bytes=32 unpacked_values=64 round_trip_passed=1");
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_attention_fixture projection=q emitted=%0d consumed=%0d payload_match=%0d ready_low_trace=0x%0h instance=u_projection_axi_stream_integration", emitted_count, consumed_count, (emitted_count == 8 && consumed_count == 8), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_attention_fixture projection=k emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_k_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_attention_fixture projection=v emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_v_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_attention_fixture projection=o emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_o_projection_axi_stream_integration", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_attention_fixture projection=q output=%0d,%0d status=0x%0h instance=u_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_attention_fixture projection=k output=%0d,%0d status=0x%0h instance=u_k_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_attention_fixture projection=v output=%0d,%0d status=0x%0h instance=u_v_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_attention_fixture projection=o output=%0d,%0d status=0x%0h instance=u_o_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_attention_fixture projection=q packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_attention_fixture projection=k packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_k_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_attention_fixture projection=v packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_v_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_attention_fixture projection=o packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_o_projection_axi_stream_integration");
        $display("CACHE_WRITE_TRACE top_fsm_axi_attention_fixture count=%0d slot=%0d key=%0d,%0d,%0d,%0d value=%0d,%0d,%0d,%0d stable=%0d", write_count, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r, lane_s8(observed_write_key, 0), lane_s8(observed_write_key, 1), lane_s8(observed_write_key, 2), lane_s8(observed_write_key, 3), lane_s8(observed_write_value, 0), lane_s8(observed_write_value, 1), lane_s8(observed_write_value, 2), lane_s8(observed_write_value, 3), accepted_write_stable);
        $display("KEY_READ_TRACE top_fsm_axi_attention_fixture slots=0,1 keys=%0d,%0d,%0d,%0d|%0d,%0d,%0d,%0d", lane_s8(observed_key0, 0), lane_s8(observed_key0, 1), lane_s8(observed_key0, 2), lane_s8(observed_key0, 3), lane_s8(observed_key1, 0), lane_s8(observed_key1, 1), lane_s8(observed_key1, 2), lane_s8(observed_key1, 3));
        $display("VALUE_READ_TRACE top_fsm_axi_attention_fixture slots=0,1 values=%0d,%0d,%0d,%0d|%0d,%0d,%0d,%0d", lane_s8(observed_value0, 0), lane_s8(observed_value0, 1), lane_s8(observed_value0, 2), lane_s8(observed_value0, 3), lane_s8(observed_value1, 0), lane_s8(observed_value1, 1), lane_s8(observed_value1, 2), lane_s8(observed_value1, 3));
        $display("SCORE_TRACE top_fsm_axi_attention_fixture scores=%0d,%0d", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r);
        $display("SOFTMAX_CONTROL_TRACE top_fsm_axi_attention_fixture policy=two_score_winner_loser_q0_4 weights=%0d,%0d exp=0", dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r, dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r);
        $display("ATTENTION_OUTPUT_TRACE top_fsm_axi_attention_fixture output=%0d,%0d,%0d,%0d", lane_s16(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 0), lane_s16(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 1), lane_s16(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 2), lane_s16(dut.u_layer_fsm_axi_attention_fixture.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.output_o, 3));
        $display("FINAL_OUTPUT_TRACE top_fsm_axi_attention_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(output_vec, 0), lane_s16(output_vec, 1), lane_s16(output_vec, 2), lane_s16(output_vec, 3), status_vec);
        $display("TOP_STABILITY_TRACE top_fsm_axi_attention_fixture stable=%0d", stable_passed);
        $display("LAYER_START_HOLD_TRACE top_fsm_axi_attention_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d child_done_release_seen_after_start_deassert=%0d", layer_busy_cycles, layer_done_seen_while_start_high_r, layer_start_deasserted_after_done_r, layer_done_release_seen_after_start_deassert_r);
        $display("COMPACT_IO_TRACE top_fsm_axi_attention_fixture estimated_iob_bits=196 exposed_128b_axi_data=0 exposed_kv_arrays=0 exposed_axi_debug=0");
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL top_fsm_axi_attention_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS top_fsm_axi_attention_fixture");
        $finish;
    end
endmodule
