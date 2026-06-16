`timescale 1ns/1ps

module tb_decoder_child_attention_datapath;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [4*16-1:0] output_vec;
    logic [95:0] status_vec;
    logic signed [4*16-1:0] stable_output_snapshot;
    logic [95:0] stable_status_snapshot;
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
    logic projection_request_accepted_seen;
    logic [3:0] projection_response_mask_seen;
    logic [4:0] projection_payload_emit_count_seen;
    logic [4:0] projection_consume_count_seen;
    logic [15:0] projection_adapter_emit_trace_seen;
    logic [15:0] projection_consume_trace_seen;
    logic projection_payload_match;
    integer write_count;
    integer key_read_count;
    integer value_read_count;
    integer projection_response_count;
    integer projection_payload_emit_count;
    integer projection_consume_count;
    integer observed;

    decoder_child_attention_datapath dut (
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

    function automatic integer popcount4(input logic [3:0] bits_i);
        integer idx;
        begin
            popcount4 = 0;
            for (idx = 0; idx < 4; idx = idx + 1) begin
                if (bits_i[idx]) begin
                    popcount4 = popcount4 + 1;
                end
            end
        end
    endfunction

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
            projection_request_accepted_seen <= 1'b0;
            projection_response_mask_seen <= '0;
            projection_payload_emit_count_seen <= '0;
            projection_consume_count_seen <= '0;
            projection_adapter_emit_trace_seen <= '0;
            projection_consume_trace_seen <= '0;
            write_count <= 0;
            key_read_count <= 0;
            value_read_count <= 0;
        end else begin
            if (dut.projection_shell_start_r) begin
                projection_request_accepted_seen <= projection_request_accepted_seen | dut.u_projection_internal_stream_shell.u_boundary.req_accepted_r;
                projection_response_mask_seen <= projection_response_mask_seen | dut.u_projection_internal_stream_shell.u_boundary.response_accepted_trace_w;
                projection_payload_emit_count_seen <= dut.u_projection_internal_stream_shell.u_boundary.payload_emit_count_w;
                projection_consume_count_seen <= dut.u_projection_internal_stream_shell.u_boundary.projection_consume_count_w;
                projection_adapter_emit_trace_seen <= dut.u_projection_internal_stream_shell.u_boundary.adapter_emit_trace_w;
                projection_consume_trace_seen <= dut.u_projection_internal_stream_shell.u_boundary.projection_consume_trace_w;
            end
            if (dut.u_attention_kv_cache_fixture.cache_write_valid_r && !dut.u_attention_kv_cache_fixture.cache_write_accept_r) begin
                held_write_seen <= 1'b1;
                held_write_slot <= dut.u_attention_kv_cache_fixture.cache_write_slot_r;
                held_write_key <= dut.u_attention_kv_cache_fixture.cache_write_key_r;
                held_write_value <= dut.u_attention_kv_cache_fixture.cache_write_value_r;
            end
            if (dut.u_attention_kv_cache_fixture.cache_write_valid_r && dut.u_attention_kv_cache_fixture.cache_write_accept_r) begin
                write_count <= write_count + 1;
                if (!held_write_seen || held_write_slot != dut.u_attention_kv_cache_fixture.cache_write_slot_r || held_write_key != dut.u_attention_kv_cache_fixture.cache_write_key_r || held_write_value != dut.u_attention_kv_cache_fixture.cache_write_value_r) begin
                    $display("FAIL decoder_child_attention_datapath attention write fields changed before accept");
                    $fatal;
                end
                accepted_write_stable <= 1'b1;
                observed_write_key <= dut.u_attention_kv_cache_fixture.cache_write_key_r;
                observed_write_value <= dut.u_attention_kv_cache_fixture.cache_write_value_r;
            end
            if (dut.u_attention_kv_cache_fixture.key_read_valid_r) begin
                key_read_count <= key_read_count + 1;
                if (dut.u_attention_kv_cache_fixture.key_read_slot_r == 1'b0) begin
                    observed_key0 <= dut.u_attention_kv_cache_fixture.key_read_data_r;
                end else begin
                    observed_key1 <= dut.u_attention_kv_cache_fixture.key_read_data_r;
                end
            end
            if (dut.u_attention_kv_cache_fixture.value_read_valid_r) begin
                value_read_count <= value_read_count + 1;
                if (dut.u_attention_kv_cache_fixture.value_read_slot_r == 1'b0) begin
                    observed_value0 <= dut.u_attention_kv_cache_fixture.value_read_data_r;
                end else begin
                    observed_value1 <= dut.u_attention_kv_cache_fixture.value_read_data_r;
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        stable_passed = 1'b0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #8000;
        if (!done_o) begin
            $display("FAIL decoder_child_attention_datapath done_o was not asserted");
            $fatal;
        end
        if (status_vec[47:0] != 48'h323122211211) begin
            $display("FAIL decoder_child_attention_datapath trace observed=0x%0h expected=0x323122211211", status_vec[47:0]);
            $fatal;
        end
        if (status_vec[79:72] != 8'hac) begin
            $display("FAIL decoder_child_attention_datapath compact status=0x%0h", status_vec);
            $fatal;
        end
        if (dut.u_projection_internal_stream_shell.output_o != 64'h00000938000003d0) begin
            $display("FAIL decoder_child_attention_datapath projection output=0x%0h", dut.u_projection_internal_stream_shell.output_o);
            $fatal;
        end
        projection_response_count = popcount4(projection_response_mask_seen);
        projection_payload_emit_count = int'(projection_payload_emit_count_seen);
        projection_consume_count = int'(projection_consume_count_seen);
        projection_payload_match = (projection_adapter_emit_trace_seen == projection_consume_trace_seen) && (projection_payload_emit_count == projection_consume_count);
        if (!projection_request_accepted_seen || projection_response_count != 4 || projection_payload_emit_count != 16 || projection_consume_count != 16 || !projection_payload_match) begin
            $display("FAIL decoder_child_attention_datapath projection shell observed request=%0d responses=%0d emit=%0d consume=%0d payload_match=%0d", dut.u_projection_internal_stream_shell.u_boundary.req_accepted_r, projection_response_count, projection_payload_emit_count, projection_consume_count, projection_payload_match);
            $fatal;
        end
        if (write_count != 1 || !accepted_write_stable || dut.u_attention_kv_cache_fixture.cache_write_slot_r != 1'b1) begin
            $display("FAIL decoder_child_attention_datapath attention write count=%0d stable=%0d slot=%0d", write_count, accepted_write_stable, dut.u_attention_kv_cache_fixture.cache_write_slot_r);
            $fatal;
        end
        if (lane_s8(observed_write_key, 0) != -8'sd3 || lane_s8(observed_write_key, 1) != 8'sd2 || lane_s8(observed_write_key, 2) != 8'sd1 || lane_s8(observed_write_key, 3) != 8'sd6) begin
            $display("FAIL decoder_child_attention_datapath observed write key values=%0d,%0d,%0d,%0d", lane_s8(observed_write_key, 0), lane_s8(observed_write_key, 1), lane_s8(observed_write_key, 2), lane_s8(observed_write_key, 3));
            $fatal;
        end
        if (lane_s8(observed_write_value, 0) != -8'sd2 || lane_s8(observed_write_value, 1) != 8'sd6 || lane_s8(observed_write_value, 2) != -8'sd5 || lane_s8(observed_write_value, 3) != 8'sd4) begin
            $display("FAIL decoder_child_attention_datapath observed write value values=%0d,%0d,%0d,%0d", lane_s8(observed_write_value, 0), lane_s8(observed_write_value, 1), lane_s8(observed_write_value, 2), lane_s8(observed_write_value, 3));
            $fatal;
        end
        if (key_read_count != 2 || value_read_count != 2) begin
            $display("FAIL decoder_child_attention_datapath attention read counts key=%0d value=%0d", key_read_count, value_read_count);
            $fatal;
        end
        if (lane_s8(observed_key0, 0) != 8'sd2 || lane_s8(observed_key0, 1) != -8'sd1 || lane_s8(observed_key0, 2) != 8'sd4 || lane_s8(observed_key0, 3) != 8'sd3 || lane_s8(observed_key1, 0) != -8'sd3 || lane_s8(observed_key1, 1) != 8'sd2 || lane_s8(observed_key1, 2) != 8'sd1 || lane_s8(observed_key1, 3) != 8'sd6) begin
            $display("FAIL decoder_child_attention_datapath observed key reads slot0=%0d,%0d,%0d,%0d slot1=%0d,%0d,%0d,%0d", lane_s8(observed_key0, 0), lane_s8(observed_key0, 1), lane_s8(observed_key0, 2), lane_s8(observed_key0, 3), lane_s8(observed_key1, 0), lane_s8(observed_key1, 1), lane_s8(observed_key1, 2), lane_s8(observed_key1, 3));
            $fatal;
        end
        if (lane_s8(observed_value0, 0) != 8'sd7 || lane_s8(observed_value0, 1) != -8'sd4 || lane_s8(observed_value0, 2) != 8'sd3 || lane_s8(observed_value0, 3) != 8'sd2 || lane_s8(observed_value1, 0) != -8'sd2 || lane_s8(observed_value1, 1) != 8'sd6 || lane_s8(observed_value1, 2) != -8'sd5 || lane_s8(observed_value1, 3) != 8'sd4) begin
            $display("FAIL decoder_child_attention_datapath observed value reads slot0=%0d,%0d,%0d,%0d slot1=%0d,%0d,%0d,%0d", lane_s8(observed_value0, 0), lane_s8(observed_value0, 1), lane_s8(observed_value0, 2), lane_s8(observed_value0, 3), lane_s8(observed_value1, 0), lane_s8(observed_value1, 1), lane_s8(observed_value1, 2), lane_s8(observed_value1, 3));
            $fatal;
        end
        if (!dut.u_attention_kv_cache_fixture.score0_valid_r || !dut.u_attention_kv_cache_fixture.score1_valid_r || dut.u_attention_kv_cache_fixture.score0_r != 25 || dut.u_attention_kv_cache_fixture.score1_r != -14) begin
            $display("FAIL decoder_child_attention_datapath attention scores observed=%0d,%0d", dut.u_attention_kv_cache_fixture.score0_r, dut.u_attention_kv_cache_fixture.score1_r);
            $fatal;
        end
        if (!dut.u_attention_kv_cache_fixture.control_valid_r || dut.u_attention_kv_cache_fixture.weight0_r != 8'd12 || dut.u_attention_kv_cache_fixture.weight1_r != 8'd4) begin
            $display("FAIL decoder_child_attention_datapath attention weights observed=%0d,%0d", dut.u_attention_kv_cache_fixture.weight0_r, dut.u_attention_kv_cache_fixture.weight1_r);
            $fatal;
        end
        if (!dut.u_attention_kv_cache_fixture.output_valid_r || dut.u_attention_kv_cache_fixture.output_o != output_vec) begin
            $display("FAIL decoder_child_attention_datapath attention child output mismatch child=0x%0h top=0x%0h", dut.u_attention_kv_cache_fixture.output_o, output_vec);
            $fatal;
        end
        observed = $signed({ {16{output_vec[0*16 + 15]}}, output_vec[0*16 +: 16] });
        if (observed != 4) begin
            $display("FAIL decoder_child_attention_datapath output[%0d] observed=%0d expected=4", 0, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[1*16 + 15]}}, output_vec[1*16 +: 16] });
        if (observed != -2) begin
            $display("FAIL decoder_child_attention_datapath output[%0d] observed=%0d expected=-2", 1, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[2*16 + 15]}}, output_vec[2*16 +: 16] });
        if (observed != 1) begin
            $display("FAIL decoder_child_attention_datapath output[%0d] observed=%0d expected=1", 2, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[3*16 + 15]}}, output_vec[3*16 +: 16] });
        if (observed != 2) begin
            $display("FAIL decoder_child_attention_datapath output[%0d] observed=%0d expected=2", 3, observed);
            $fatal;
        end
        stable_output_snapshot = output_vec;
        stable_status_snapshot = status_vec;
        #20;
        if (output_vec != stable_output_snapshot || status_vec != stable_status_snapshot || !done_o) begin
            stable_passed = 1'b0;
            $display("FAIL decoder_child_attention_datapath output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;
        $display("CHILD_TRACE decoder_child_attention_datapath trace_hex=0x%0h events=source_path_start,source_path_done,projection_shell_start,projection_shell_done,attention_kv_start,attention_kv_done", status_vec[47:0]);
        $display("RMS_LOOKUP_TRACE decoder_child_attention_datapath selector=%0d valid=%0d inv_rms=%0d sumsq=%0d", dut.u_rmsnorm_rope_source_path.status_o[2:1], dut.u_rmsnorm_rope_source_path.status_o[0], dut.u_rmsnorm_rope_source_path.status_o[18:3], dut.u_rmsnorm_rope_source_path.status_o[34:19]);
        $display("ROPE_LOOKUP_TRACE decoder_child_attention_datapath position=%0d pair=0 valid=%0d cos=%0d sin=%0d", dut.u_rmsnorm_rope_source_path.status_o[39:36], dut.u_rmsnorm_rope_source_path.status_o[35], $signed(dut.u_rmsnorm_rope_source_path.status_o[48:41]), $signed(dut.u_rmsnorm_rope_source_path.status_o[56:49]));
        $display("ROPE_LOOKUP_TRACE decoder_child_attention_datapath position=%0d pair=1 valid=%0d cos=%0d sin=%0d", dut.u_rmsnorm_rope_source_path.status_o[61:58], dut.u_rmsnorm_rope_source_path.status_o[57], $signed(dut.u_rmsnorm_rope_source_path.status_o[70:63]), $signed(dut.u_rmsnorm_rope_source_path.status_o[78:71]));
        $display("PROJECTION_STREAM_TRACE decoder_child_attention_datapath shell_status=0x%0h request_accepted=%0d response_count=%0d response_mask=0x%0h payload_emit_count=%0d projection_consume_count=%0d payload_match=%0d source=hierarchical_child_status", dut.u_projection_internal_stream_shell.shell_status_o, projection_request_accepted_seen, projection_response_count, projection_response_mask_seen, projection_payload_emit_count, projection_consume_count, projection_payload_match);
        $display("PROJECTION_OUTPUT_TRACE decoder_child_attention_datapath output=%0d,%0d shell_status=0x%0h", $signed(dut.u_projection_internal_stream_shell.output_o[0*32 +: 32]), $signed(dut.u_projection_internal_stream_shell.output_o[1*32 +: 32]), dut.u_projection_internal_stream_shell.shell_status_o);
        $display("CACHE_WRITE_TRACE decoder_child_attention_datapath count=%0d slot=%0d key=%0d,%0d,%0d,%0d value=%0d,%0d,%0d,%0d stable=%0d", write_count, dut.u_attention_kv_cache_fixture.cache_write_slot_r, lane_s8(observed_write_key, 0), lane_s8(observed_write_key, 1), lane_s8(observed_write_key, 2), lane_s8(observed_write_key, 3), lane_s8(observed_write_value, 0), lane_s8(observed_write_value, 1), lane_s8(observed_write_value, 2), lane_s8(observed_write_value, 3), accepted_write_stable);
        $display("KEY_READ_TRACE decoder_child_attention_datapath slots=0,1 keys=%0d,%0d,%0d,%0d|%0d,%0d,%0d,%0d", lane_s8(observed_key0, 0), lane_s8(observed_key0, 1), lane_s8(observed_key0, 2), lane_s8(observed_key0, 3), lane_s8(observed_key1, 0), lane_s8(observed_key1, 1), lane_s8(observed_key1, 2), lane_s8(observed_key1, 3));
        $display("VALUE_READ_TRACE decoder_child_attention_datapath slots=0,1 values=%0d,%0d,%0d,%0d|%0d,%0d,%0d,%0d", lane_s8(observed_value0, 0), lane_s8(observed_value0, 1), lane_s8(observed_value0, 2), lane_s8(observed_value0, 3), lane_s8(observed_value1, 0), lane_s8(observed_value1, 1), lane_s8(observed_value1, 2), lane_s8(observed_value1, 3));
        $display("SCORE_TRACE decoder_child_attention_datapath scores=%0d,%0d", dut.u_attention_kv_cache_fixture.score0_r, dut.u_attention_kv_cache_fixture.score1_r);
        $display("SOFTMAX_CONTROL_TRACE decoder_child_attention_datapath policy=two_score_winner_loser_q0_4 weights=%0d,%0d exp=0", dut.u_attention_kv_cache_fixture.weight0_r, dut.u_attention_kv_cache_fixture.weight1_r);
        $display("ATTENTION_OUTPUT_TRACE decoder_child_attention_datapath output=%0d,%0d,%0d,%0d", lane_s16(dut.u_attention_kv_cache_fixture.output_o, 0), lane_s16(dut.u_attention_kv_cache_fixture.output_o, 1), lane_s16(dut.u_attention_kv_cache_fixture.output_o, 2), lane_s16(dut.u_attention_kv_cache_fixture.output_o, 3));
        $display("FINAL_OUTPUT_TRACE decoder_child_attention_datapath output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(output_vec, 0), lane_s16(output_vec, 1), lane_s16(output_vec, 2), lane_s16(output_vec, 3), status_vec);
        $display("TOP_STABILITY_TRACE decoder_child_attention_datapath stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE decoder_child_attention_datapath estimated_iob_bits=164 exposed_128b=0 exposed_kv_arrays=0");
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL decoder_child_attention_datapath done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS decoder_child_attention_datapath");
        $finish;
    end
endmodule
