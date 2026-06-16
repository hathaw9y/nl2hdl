`timescale 1ns/1ps

module tb_decoder_block_axi_attention_mlp_fixture;
    localparam int VALUES = 4;
    localparam int ELEM_WIDTH = 16;
    localparam int STATUS_WIDTH = 176;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_o;
    logic [STATUS_WIDTH-1:0] status_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    logic axi_attention_busy_seen_r;
    logic axi_attention_done_seen_while_start_high_r;
    logic axi_attention_start_deasserted_after_done_r;
    logic axi_attention_release_seen_r;
    logic axi_attention_done_prev_r;
    integer axi_attention_busy_cycles;

    logic mlp_busy_seen_r;
    logic mlp_done_seen_while_start_high_r;
    logic mlp_start_deasserted_after_done_r;
    logic mlp_release_seen_r;
    logic mlp_done_prev_r;
    integer mlp_busy_cycles;

    logic [31:0] observed_emitted_payload [0:7];
    logic [31:0] observed_consumed_payload [0:7];
    logic [7:0] observed_ready_low_trace;
    logic payload_seen_pending;
    logic payload_hold_ok;
    integer emitted_count;
    integer consumed_count;
    integer ar_ready_low_cycles;
    integer rvalid_while_projection_not_ready_cycles;
    integer idx;
    integer observed;

    decoder_block_axi_attention_mlp_fixture dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
        .final_output_o(final_output_o),
        .status_o(status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    function automatic integer lane_s16(
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input integer idx_i
    );
        logic signed [ELEM_WIDTH-1:0] tmp;
        begin
            tmp = vec_i[idx_i*ELEM_WIDTH +: ELEM_WIDTH];
            lane_s16 = int'($signed(tmp));
        end
    endfunction

    function automatic logic signed [7:0] lane_s8(input logic signed [31:0] packed_i, input int idx_i);
        begin
            lane_s8 = packed_i[idx_i*8 +: 8];
        end
    endfunction

    function automatic logic signed [7:0] sign_extend_int4(input logic [3:0] nibble_i);
        begin
            sign_extend_int4 = { {4{nibble_i[3]}}, nibble_i };
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

    task automatic check_lane(
        input string label_i,
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input integer idx_i,
        input integer expected_i
    );
        begin
            observed = lane_s16(vec_i, idx_i);
            if (observed != expected_i) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always @(negedge aclk) begin
        if (!aresetn || !start_i) begin
            payload_seen_pending = 1'b0;
        end else begin
            if (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arvalid_w &&
                !dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arready_w) begin
                ar_ready_low_cycles = ar_ready_low_cycles + 1;
            end
            if (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rvalid_w &&
                !dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_rready_w &&
                dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_valid_r) begin
                rvalid_while_projection_not_ready_cycles = rvalid_while_projection_not_ready_cycles + 1;
            end
            if (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_valid_r) begin
                if (!dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_ready_w && consumed_count < 8) begin
                    observed_ready_low_trace[consumed_count] = 1'b1;
                    if (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r != expected_axi_payload_at(consumed_count)) begin
                        payload_hold_ok = 1'b0;
                    end
                end
                if (!payload_seen_pending) begin
                    observed_emitted_payload[emitted_count] = dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r;
                    emitted_count = emitted_count + 1;
                    payload_seen_pending = 1'b1;
                end
                if (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_ready_w) begin
                    observed_consumed_payload[consumed_count] = dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.payload_link_word_r;
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
            axi_attention_busy_seen_r <= 1'b0;
            axi_attention_done_seen_while_start_high_r <= 1'b0;
            axi_attention_start_deasserted_after_done_r <= 1'b0;
            axi_attention_release_seen_r <= 1'b0;
            axi_attention_done_prev_r <= 1'b0;
            axi_attention_busy_cycles <= 0;
        end else begin
            if (dut.axi_attention_start_r && !dut.axi_attention_done_w) begin
                axi_attention_busy_seen_r <= 1'b1;
                axi_attention_busy_cycles <= axi_attention_busy_cycles + 1;
            end
            if (axi_attention_busy_seen_r && !axi_attention_done_seen_while_start_high_r && !dut.axi_attention_done_w && !dut.axi_attention_start_r) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture AXI attention child start was not held while busy");
                $fatal;
            end
            if (dut.axi_attention_start_r && dut.axi_attention_done_w) begin
                axi_attention_done_seen_while_start_high_r <= 1'b1;
            end
            if (axi_attention_done_prev_r && !dut.axi_attention_start_r) begin
                axi_attention_start_deasserted_after_done_r <= 1'b1;
            end
            if (axi_attention_done_prev_r && dut.axi_attention_start_r) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture AXI attention child start was not deasserted after done_o");
                $fatal;
            end
            if (axi_attention_start_deasserted_after_done_r && !dut.axi_attention_done_w) begin
                axi_attention_release_seen_r <= 1'b1;
            end
            axi_attention_done_prev_r <= dut.axi_attention_done_w;
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            mlp_busy_seen_r <= 1'b0;
            mlp_done_seen_while_start_high_r <= 1'b0;
            mlp_start_deasserted_after_done_r <= 1'b0;
            mlp_release_seen_r <= 1'b0;
            mlp_done_prev_r <= 1'b0;
            mlp_busy_cycles <= 0;
        end else begin
            if (dut.mlp_start_r && !dut.mlp_done_w) begin
                mlp_busy_seen_r <= 1'b1;
                mlp_busy_cycles <= mlp_busy_cycles + 1;
            end
            if (mlp_busy_seen_r && !mlp_done_seen_while_start_high_r && !dut.mlp_done_w && !dut.mlp_start_r) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture mlp child start was not held while busy");
                $fatal;
            end
            if (dut.mlp_start_r && dut.mlp_done_w) begin
                mlp_done_seen_while_start_high_r <= 1'b1;
            end
            if (mlp_done_prev_r && !dut.mlp_start_r) begin
                mlp_start_deasserted_after_done_r <= 1'b1;
            end
            if (mlp_done_prev_r && dut.mlp_start_r) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture mlp child start was not deasserted after done_o");
                $fatal;
            end
            if (mlp_start_deasserted_after_done_r && !dut.mlp_done_w) begin
                mlp_release_seen_r <= 1'b1;
            end
            mlp_done_prev_r <= dut.mlp_done_w;
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
        #12000;
        if (!done_o) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[31:0] != 32'hb2b1a2a1) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture block trace observed=0x%0h expected=0xb2b1a2a1", status_o[31:0]);
            $fatal;
        end
        if (status_o[32 +: 48] != 48'h323122211211) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture AXI attention trace observed=0x%0h", status_o[32 +: 48]);
            $fatal;
        end
        if (status_o[80 +: 4] != 4'hf ||
            status_o[80 +: 4] != dut.captured_attention_status_r[63:60] ||
            dut.captured_attention_status_r[63:60] !=
                (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42] &
                 dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42] &
                 dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42] &
                 dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42])) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture block AXI metadata block=0x%0h child=0x%0h status=0x%0h", status_o[80 +: 4], dut.captured_attention_status_r[63:60], status_o);
            $fatal;
        end
        if (status_o[96 +: 80] != 80'h52514241323122211211) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture mlp trace observed=0x%0h", status_o[96 +: 80]);
            $fatal;
        end
        if (!axi_attention_done_seen_while_start_high_r || !axi_attention_start_deasserted_after_done_r || !axi_attention_release_seen_r || axi_attention_busy_cycles == 0) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture AXI attention start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", axi_attention_busy_cycles, axi_attention_done_seen_while_start_high_r, axi_attention_start_deasserted_after_done_r, axi_attention_release_seen_r);
            $fatal;
        end
        if (!mlp_done_seen_while_start_high_r || !mlp_start_deasserted_after_done_r || !mlp_release_seen_r || mlp_busy_cycles == 0) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture mlp start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", mlp_busy_cycles, mlp_done_seen_while_start_high_r, mlp_start_deasserted_after_done_r, mlp_release_seen_r);
            $fatal;
        end
        if (emitted_count != 8 || consumed_count != 8 || observed_ready_low_trace != 8'h19 || !payload_hold_ok || rvalid_while_projection_not_ready_cycles < 1) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture q AXI payload evidence emitted=%0d consumed=%0d ready_low=0x%0h hold=%0d rstall=%0d", emitted_count, consumed_count, observed_ready_low_trace, payload_hold_ok, rvalid_while_projection_not_ready_cycles);
            $fatal;
        end
        for (idx = 0; idx < 8; idx = idx + 1) begin
            if (observed_emitted_payload[idx] != observed_consumed_payload[idx]) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture AXI payload mismatch[%0d]", idx);
                $fatal;
            end
        end
        for (idx = 0; idx < 64; idx = idx + 1) begin
            if (sign_extend_int4(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) !==
                sign_extend_int4(dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) ||
                sign_extend_int4(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) !==
                sign_extend_int4(dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) ||
                sign_extend_int4(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4]) !==
                sign_extend_int4(dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.packed_weight_r[idx*4 +: 4])) begin
                $display("FAIL decoder_block_axi_attention_mlp_fixture AXI round-trip mismatch[%0d]", idx);
                $fatal;
            end
        end

        check_lane("captured_attention", dut.captured_attention_r, 0, 4);
        check_lane("captured_attention", dut.captured_attention_r, 1, -2);
        check_lane("captured_attention", dut.captured_attention_r, 2, 1);
        check_lane("captured_attention", dut.captured_attention_r, 3, 2);
        if (dut.captured_attention_r != dut.u_decoder_child_axi_attention_datapath.output_o) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture captured attention did not match AXI attention child output");
            $fatal;
        end
        if (dut.u_residual_mlp_fixture.attention_r != dut.captured_attention_r) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture MLP attention input did not consume captured AXI attention output");
            $fatal;
        end
        check_lane("mlp_hidden", dut.u_residual_mlp_fixture.hidden_r, 0, 0);
        check_lane("mlp_hidden", dut.u_residual_mlp_fixture.hidden_r, 1, 4);
        check_lane("mlp_hidden", dut.u_residual_mlp_fixture.hidden_r, 2, 1);
        check_lane("mlp_hidden", dut.u_residual_mlp_fixture.hidden_r, 3, 1);
        check_lane("mlp_attention", dut.u_residual_mlp_fixture.attention_r, 0, 4);
        check_lane("mlp_attention", dut.u_residual_mlp_fixture.attention_r, 1, -2);
        check_lane("mlp_attention", dut.u_residual_mlp_fixture.attention_r, 2, 1);
        check_lane("mlp_attention", dut.u_residual_mlp_fixture.attention_r, 3, 2);
        check_lane("residual0", dut.u_residual_mlp_fixture.residual0_r, 0, 4);
        check_lane("residual0", dut.u_residual_mlp_fixture.residual0_r, 1, 2);
        check_lane("residual0", dut.u_residual_mlp_fixture.residual0_r, 2, 2);
        check_lane("residual0", dut.u_residual_mlp_fixture.residual0_r, 3, 3);
        check_lane("mlp_final", dut.u_residual_mlp_fixture.final_output_o, 0, 12);
        check_lane("mlp_final", dut.u_residual_mlp_fixture.final_output_o, 1, -6);
        check_lane("mlp_final", dut.u_residual_mlp_fixture.final_output_o, 2, 18);
        check_lane("mlp_final", dut.u_residual_mlp_fixture.final_output_o, 3, 6);
        check_lane("final_output", final_output_o, 0, 12);
        check_lane("final_output", final_output_o, 1, -6);
        check_lane("final_output", final_output_o, 2, 18);
        check_lane("final_output", final_output_o, 3, 6);

        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o != stable_output_snapshot || status_o != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL decoder_block_axi_attention_mlp_fixture output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("BLOCK_AXI_TRACE decoder_block_axi_attention_mlp_fixture block_trace_hex=0x%0h attention_trace_hex=0x%0h axi_metadata_bits=0x%0h mlp_trace_hex=0x%0h events=axi_attention_start,axi_attention_done,mlp_start,mlp_done", status_o[31:0], status_o[32 +: 48], status_o[80 +: 4], status_o[96 +: 80]);
        $display("AXI_ATTENTION_CHILD_TRACE decoder_block_axi_attention_mlp_fixture trace_hex=0x%0h events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done", dut.u_decoder_child_axi_attention_datapath.status_o[47:0]);
        $display("MLP_CHILD_TRACE decoder_block_axi_attention_mlp_fixture trace_hex=0x%0h events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done", dut.u_residual_mlp_fixture.status_o[79:0]);
        $display("AXI_ATTENTION_CHILD_START_HOLD_TRACE decoder_block_axi_attention_mlp_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", axi_attention_busy_cycles, axi_attention_done_seen_while_start_high_r, axi_attention_start_deasserted_after_done_r, axi_attention_release_seen_r);
        $display("MLP_CHILD_START_HOLD_TRACE decoder_block_axi_attention_mlp_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", mlp_busy_cycles, mlp_done_seen_while_start_high_r, mlp_start_deasserted_after_done_r, mlp_release_seen_r);
        $display("AXI_PROJECTION_AR_TRACE decoder_block_axi_attention_mlp_fixture addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=%0d", dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w, ar_ready_low_cycles);
        $display("AXI_PROJECTION_R_METADATA_TRACE decoder_block_axi_attention_mlp_fixture accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h", dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r, (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_AR_TRACE decoder_block_axi_attention_mlp_fixture projection=q addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=%0d instance=u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w, ar_ready_low_cycles);
        $display("AXI_PROJECTION_CHILD_AR_TRACE decoder_block_axi_attention_mlp_fixture projection=k addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_araddr_w, dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arlen_w, dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arsize_w, dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arburst_w, dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE decoder_block_axi_attention_mlp_fixture projection=v addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_araddr_w, dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arlen_w, dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arsize_w, dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arburst_w, dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE decoder_block_axi_attention_mlp_fixture projection=o addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_araddr_w, dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arlen_w, dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arsize_w, dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arburst_w, dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE decoder_block_axi_attention_mlp_fixture projection=q accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_trace_r, (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE decoder_block_axi_attention_mlp_fixture projection=k accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rlast_trace_r, (dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE decoder_block_axi_attention_mlp_fixture projection=v accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rlast_trace_r, (dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE decoder_block_axi_attention_mlp_fixture projection=o accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.accepted_beat_trace_r, dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rlast_trace_r, (dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rid_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rresp_error_trace_r == '0), (dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.rlast_error_trace_r == '0), dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("BLOCK_AXI_METADATA_PROPAGATION_TRACE decoder_block_axi_attention_mlp_fixture block_bits=0x%0h block_bit_lsb=80 block_bit_msb=83 attention_child_bits=0x%0h q_bits=0x%0h k_bits=0x%0h v_bits=0x%0h o_bits=0x%0h source=decoder_child_axi_attention_datapath.status_o[63:60]:q_projection_axi_stream_integration.integration_status_o[45:42]:k_projection_axi_stream_integration.integration_status_o[45:42]:v_projection_axi_stream_integration.integration_status_o[45:42]:o_projection_axi_stream_integration.integration_status_o[45:42]:aggregate_and status=0x%0h", status_o[80 +: 4], dut.captured_attention_status_r[63:60], dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42], dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42], dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42], status_o);
        $write("AXI_PROJECTION_EMITTED_PAYLOADS decoder_block_axi_attention_mlp_fixture");
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $write(" 0x%08h", observed_emitted_payload[idx]);
        end
        $write("\n");
        $write("AXI_PROJECTION_CONSUMED_PAYLOADS decoder_block_axi_attention_mlp_fixture");
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $write(" 0x%08h", observed_consumed_payload[idx]);
        end
        $write("\n");
        $display("AXI_PROJECTION_BACKPRESSURE_TRACE decoder_block_axi_attention_mlp_fixture ready_low_payload_idx=0,3,4 trace=0x%0h payload_hold_ok=%0d rvalid_while_projection_not_ready_cycles=%0d", observed_ready_low_trace, payload_hold_ok, rvalid_while_projection_not_ready_cycles);
        $display("AXI_PROJECTION_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture output=%0d,%0d status=0x%0h", $signed(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_ROUND_TRIP_TRACE decoder_block_axi_attention_mlp_fixture packed_bytes=32 unpacked_values=64 round_trip_passed=1");
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE decoder_block_axi_attention_mlp_fixture projection=q emitted=%0d consumed=%0d payload_match=%0d ready_low_trace=0x%0h instance=u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", emitted_count, consumed_count, (emitted_count == 8 && consumed_count == 8), dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE decoder_block_axi_attention_mlp_fixture projection=k emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[15:8], dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[23:16], dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE decoder_block_axi_attention_mlp_fixture projection=v emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[15:8], dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[23:16], dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE decoder_block_axi_attention_mlp_fixture projection=o emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[15:8], dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[23:16], dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture projection=q output=%0d,%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", $signed(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture projection=k output=%0d,%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", $signed(dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture projection=v output=%0d,%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", $signed(dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture projection=o output=%0d,%0d status=0x%0h instance=u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", $signed(dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE decoder_block_axi_attention_mlp_fixture projection=q packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE decoder_block_axi_attention_mlp_fixture projection=k packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE decoder_block_axi_attention_mlp_fixture projection=v packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE decoder_block_axi_attention_mlp_fixture projection=o packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration");
        $display("CACHE_WRITE_TRACE decoder_block_axi_attention_mlp_fixture count=1 slot=%0d key=%0d,%0d,%0d,%0d value=%0d,%0d,%0d,%0d stable=1", dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_slot_r, lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r, 0), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r, 1), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r, 2), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_key_r, 3), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r, 0), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r, 1), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r, 2), lane_s8(dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.cache_write_value_r, 3));
        $display("SCORE_TRACE decoder_block_axi_attention_mlp_fixture scores=%0d,%0d", dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score0_r, dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.score1_r);
        $display("SOFTMAX_CONTROL_TRACE decoder_block_axi_attention_mlp_fixture policy=bounded_fixture_static_weights weights=%0d,%0d exp=0", dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight0_r, dut.u_decoder_child_axi_attention_datapath.u_attention_kv_cache_fixture.weight1_r);
        $display("ATTENTION_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture output=%0d,%0d,%0d,%0d source=hierarchical_axi_attention_child_output", lane_s16(dut.captured_attention_r, 0), lane_s16(dut.captured_attention_r, 1), lane_s16(dut.captured_attention_r, 2), lane_s16(dut.captured_attention_r, 3));
        $display("MLP_INPUT_TRACE decoder_block_axi_attention_mlp_fixture hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d captured_match=%0d source=hierarchical_residual_mlp_child_inputs", lane_s16(dut.u_residual_mlp_fixture.hidden_r, 0), lane_s16(dut.u_residual_mlp_fixture.hidden_r, 1), lane_s16(dut.u_residual_mlp_fixture.hidden_r, 2), lane_s16(dut.u_residual_mlp_fixture.hidden_r, 3), lane_s16(dut.u_residual_mlp_fixture.attention_r, 0), lane_s16(dut.u_residual_mlp_fixture.attention_r, 1), lane_s16(dut.u_residual_mlp_fixture.attention_r, 2), lane_s16(dut.u_residual_mlp_fixture.attention_r, 3), (dut.u_residual_mlp_fixture.attention_r == dut.captured_attention_r));
        $display("RESIDUAL0_TRACE decoder_block_axi_attention_mlp_fixture residual0=%0d,%0d,%0d,%0d", lane_s16(dut.u_residual_mlp_fixture.residual0_r, 0), lane_s16(dut.u_residual_mlp_fixture.residual0_r, 1), lane_s16(dut.u_residual_mlp_fixture.residual0_r, 2), lane_s16(dut.u_residual_mlp_fixture.residual0_r, 3));
        $display("GATE_UP_TRACE decoder_block_axi_attention_mlp_fixture gate=%0d,%0d,%0d,%0d up=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_fixture_constant_matrices", lane_s16(dut.u_residual_mlp_fixture.gate_r, 0), lane_s16(dut.u_residual_mlp_fixture.gate_r, 1), lane_s16(dut.u_residual_mlp_fixture.gate_r, 2), lane_s16(dut.u_residual_mlp_fixture.gate_r, 3), lane_s16(dut.u_residual_mlp_fixture.up_r, 0), lane_s16(dut.u_residual_mlp_fixture.up_r, 1), lane_s16(dut.u_residual_mlp_fixture.up_r, 2), lane_s16(dut.u_residual_mlp_fixture.up_r, 3));
        $display("SWIGLU_TRACE decoder_block_axi_attention_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=%0d,%0d,%0d,%0d true_silu_exp=0", lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 0), lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 1), lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 2), lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 3));
        $display("DOWN_TRACE decoder_block_axi_attention_mlp_fixture down=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_fixture_constant_matrix", lane_s16(dut.u_residual_mlp_fixture.down_r, 0), lane_s16(dut.u_residual_mlp_fixture.down_r, 1), lane_s16(dut.u_residual_mlp_fixture.down_r, 2), lane_s16(dut.u_residual_mlp_fixture.down_r, 3));
        $display("MLP_FINAL_TRACE decoder_block_axi_attention_mlp_fixture output=%0d,%0d,%0d,%0d", lane_s16(dut.u_residual_mlp_fixture.final_output_o, 0), lane_s16(dut.u_residual_mlp_fixture.final_output_o, 1), lane_s16(dut.u_residual_mlp_fixture.final_output_o, 2), lane_s16(dut.u_residual_mlp_fixture.final_output_o, 3));
        $display("FINAL_OUTPUT_TRACE decoder_block_axi_attention_mlp_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("DECODER_BLOCK_STABILITY_TRACE decoder_block_axi_attention_mlp_fixture stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE decoder_block_axi_attention_mlp_fixture estimated_iob_bits=244 residual_standalone_iob_reference=292 axi_attention_child_iob_reference=164 exposed_128b=0 exposed_axi_debug=0 exposed_kv_arrays=0 exposed_hidden_ports=0 exposed_child_status_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL decoder_block_axi_attention_mlp_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS decoder_block_axi_attention_mlp_fixture");
        $finish;
    end
endmodule
