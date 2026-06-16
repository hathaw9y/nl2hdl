`timescale 1ns/1ps

module tb_layer_fsm_decoder_block_fixture;
    localparam int VALUES = 4;
    localparam int ELEM_WIDTH = 16;
    localparam int STATUS_WIDTH = 64;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_o;
    logic [STATUS_WIDTH-1:0] status_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    logic block_busy_seen_r;
    logic block_done_seen_while_start_high_r;
    logic block_start_deasserted_after_done_r;
    logic block_release_seen_r;
    logic block_done_prev_r;
    integer block_busy_cycles;
    integer observed;

    layer_fsm_decoder_block_fixture dut (
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

    task automatic check_lane(
        input string label_i,
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input integer idx_i,
        input integer expected_i
    );
        begin
            observed = lane_s16(vec_i, idx_i);
            if (observed != expected_i) begin
                $display("FAIL layer_fsm_decoder_block_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always @(posedge aclk) begin
        if (!aresetn) begin
            block_busy_seen_r <= 1'b0;
            block_done_seen_while_start_high_r <= 1'b0;
            block_start_deasserted_after_done_r <= 1'b0;
            block_release_seen_r <= 1'b0;
            block_done_prev_r <= 1'b0;
            block_busy_cycles <= 0;
        end else begin
            if (dut.block_start_r && !dut.block_done_w) begin
                block_busy_seen_r <= 1'b1;
                block_busy_cycles <= block_busy_cycles + 1;
            end
            if (block_busy_seen_r && !block_done_seen_while_start_high_r && !dut.block_done_w && !dut.block_start_r) begin
                $display("FAIL layer_fsm_decoder_block_fixture block child start was not held while busy");
                $fatal;
            end
            if (dut.block_start_r && dut.block_done_w) begin
                block_done_seen_while_start_high_r <= 1'b1;
            end
            if (block_done_prev_r && !dut.block_start_r) begin
                block_start_deasserted_after_done_r <= 1'b1;
            end
            if (block_done_prev_r && dut.block_start_r) begin
                $display("FAIL layer_fsm_decoder_block_fixture block child start was not deasserted after done_o");
                $fatal;
            end
            if (block_start_deasserted_after_done_r && !dut.block_done_w) begin
                block_release_seen_r <= 1'b1;
            end
            block_done_prev_r <= dut.block_done_w;
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
        #12000;
        if (!done_o) begin
            $display("FAIL layer_fsm_decoder_block_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[15:0] != 16'h4241) begin
            $display("FAIL layer_fsm_decoder_block_fixture layer trace observed=0x%0h expected=0x4241", status_o[15:0]);
            $fatal;
        end
        if (status_o[16 +: 32] != 32'hb2b1a2a1) begin
            $display("FAIL layer_fsm_decoder_block_fixture block trace observed=0x%0h expected=0xb2b1a2a1", status_o[16 +: 32]);
            $fatal;
        end
        if (status_o[48 +: 16] != 16'h4c01) begin
            $display("FAIL layer_fsm_decoder_block_fixture compact status marker observed=0x%0h", status_o[48 +: 16]);
            $fatal;
        end
        if (dut.u_decoder_block_attention_mlp_fixture.status_o[31:0] != 32'hb2b1a2a1) begin
            $display("FAIL layer_fsm_decoder_block_fixture hierarchical block trace observed=0x%0h", dut.u_decoder_block_attention_mlp_fixture.status_o[31:0]);
            $fatal;
        end
        if (dut.u_decoder_block_attention_mlp_fixture.status_o[32 +: 48] != 48'h323122211211) begin
            $display("FAIL layer_fsm_decoder_block_fixture attention trace observed=0x%0h", dut.u_decoder_block_attention_mlp_fixture.status_o[32 +: 48]);
            $fatal;
        end
        if (dut.u_decoder_block_attention_mlp_fixture.status_o[80 +: 80] != 80'h52514241323122211211) begin
            $display("FAIL layer_fsm_decoder_block_fixture mlp trace observed=0x%0h", dut.u_decoder_block_attention_mlp_fixture.status_o[80 +: 80]);
            $fatal;
        end
        if (dut.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.status_o[47:0] != 48'h323122211211) begin
            $display("FAIL layer_fsm_decoder_block_fixture nested attention child trace observed=0x%0h", dut.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.status_o[47:0]);
            $fatal;
        end
        if (dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0] != 80'h52514241323122211211) begin
            $display("FAIL layer_fsm_decoder_block_fixture nested MLP child trace observed=0x%0h", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0]);
            $fatal;
        end
        if (!block_done_seen_while_start_high_r || !block_start_deasserted_after_done_r || !block_release_seen_r || block_busy_cycles == 0) begin
            $display("FAIL layer_fsm_decoder_block_fixture block start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", block_busy_cycles, block_done_seen_while_start_high_r, block_start_deasserted_after_done_r, block_release_seen_r);
            $fatal;
        end

        check_lane("decoder_block_output", dut.u_decoder_block_attention_mlp_fixture.final_output_o, 0, 12);
        check_lane("decoder_block_output", dut.u_decoder_block_attention_mlp_fixture.final_output_o, 1, -6);
        check_lane("decoder_block_output", dut.u_decoder_block_attention_mlp_fixture.final_output_o, 2, 18);
        check_lane("decoder_block_output", dut.u_decoder_block_attention_mlp_fixture.final_output_o, 3, 6);
        check_lane("captured_attention", dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 0, 4);
        check_lane("captured_attention", dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 1, -2);
        check_lane("captured_attention", dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 2, 1);
        check_lane("captured_attention", dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 3, 2);
        if (dut.u_decoder_block_attention_mlp_fixture.captured_attention_r != dut.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.output_o) begin
            $display("FAIL layer_fsm_decoder_block_fixture captured attention did not match attention child output");
            $fatal;
        end
        if (dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r != dut.u_decoder_block_attention_mlp_fixture.captured_attention_r) begin
            $display("FAIL layer_fsm_decoder_block_fixture MLP child did not consume captured attention output");
            $fatal;
        end
        check_lane("mlp_hidden", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 0, 0);
        check_lane("mlp_hidden", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 1, 4);
        check_lane("mlp_hidden", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 2, 1);
        check_lane("mlp_hidden", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 3, 1);
        check_lane("mlp_attention", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 0, 4);
        check_lane("mlp_attention", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 1, -2);
        check_lane("mlp_attention", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 2, 1);
        check_lane("mlp_attention", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 3, 2);
        check_lane("residual0", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 0, 4);
        check_lane("residual0", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 1, 2);
        check_lane("residual0", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 2, 2);
        check_lane("residual0", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 3, 3);
        check_lane("mlp_final", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 0, 12);
        check_lane("mlp_final", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 1, -6);
        check_lane("mlp_final", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 2, 18);
        check_lane("mlp_final", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 3, 6);
        check_lane("final_layer_output", final_output_o, 0, 12);
        check_lane("final_layer_output", final_output_o, 1, -6);
        check_lane("final_layer_output", final_output_o, 2, 18);
        check_lane("final_layer_output", final_output_o, 3, 6);

        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o != stable_output_snapshot || status_o != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL layer_fsm_decoder_block_fixture output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("LAYER_TRACE layer_fsm_decoder_block_fixture layer_trace_hex=0x%0h block_trace_hex=0x%0h events=decoder_block_attention_mlp_fixture_start,decoder_block_attention_mlp_fixture_done", status_o[15:0], status_o[16 +: 32]);
        $display("BLOCK_TRACE layer_fsm_decoder_block_fixture block_trace_hex=0x%0h attention_trace_hex=0x%0h mlp_trace_hex=0x%0h events=attention_start,attention_done,mlp_start,mlp_done", dut.u_decoder_block_attention_mlp_fixture.status_o[31:0], dut.u_decoder_block_attention_mlp_fixture.status_o[32 +: 48], dut.u_decoder_block_attention_mlp_fixture.status_o[80 +: 80]);
        $display("ATTENTION_CHILD_TRACE layer_fsm_decoder_block_fixture trace_hex=0x%0h events=source_path_start,source_path_done,projection_shell_start,projection_shell_done,attention_kv_start,attention_kv_done", dut.u_decoder_block_attention_mlp_fixture.u_decoder_child_attention_datapath.status_o[47:0]);
        $display("MLP_CHILD_TRACE layer_fsm_decoder_block_fixture trace_hex=0x%0h events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done", dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0]);
        $display("LAYER_BLOCK_CHILD_START_HOLD_TRACE layer_fsm_decoder_block_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", block_busy_cycles, block_done_seen_while_start_high_r, block_start_deasserted_after_done_r, block_release_seen_r);
        $display("ATTENTION_OUTPUT_TRACE layer_fsm_decoder_block_fixture output=%0d,%0d,%0d,%0d source=hierarchical_attention_child_output", lane_s16(dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.captured_attention_r, 3));
        $display("MLP_INPUT_TRACE layer_fsm_decoder_block_fixture hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_child_inputs", lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 3), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 3));
        $display("RESIDUAL0_TRACE layer_fsm_decoder_block_fixture residual0=%0d,%0d,%0d,%0d", lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 3));
        $display("GATE_UP_TRACE layer_fsm_decoder_block_fixture gate=%0d,%0d,%0d,%0d up=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_fixture_constant_matrices", lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.gate_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.gate_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.gate_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.gate_r, 3), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.up_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.up_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.up_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.up_r, 3));
        $display("SWIGLU_TRACE layer_fsm_decoder_block_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=%0d,%0d,%0d,%0d true_silu_exp=0", lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.swiglu_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.swiglu_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.swiglu_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.swiglu_r, 3));
        $display("DOWN_TRACE layer_fsm_decoder_block_fixture down=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_fixture_constant_matrix", lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.down_r, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.down_r, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.down_r, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.down_r, 3));
        $display("MLP_FINAL_TRACE layer_fsm_decoder_block_fixture output=%0d,%0d,%0d,%0d", lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 3));
        $display("FINAL_DECODER_BLOCK_OUTPUT_TRACE layer_fsm_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(dut.u_decoder_block_attention_mlp_fixture.final_output_o, 0), lane_s16(dut.u_decoder_block_attention_mlp_fixture.final_output_o, 1), lane_s16(dut.u_decoder_block_attention_mlp_fixture.final_output_o, 2), lane_s16(dut.u_decoder_block_attention_mlp_fixture.final_output_o, 3), dut.u_decoder_block_attention_mlp_fixture.status_o);
        $display("FINAL_OUTPUT_TRACE layer_fsm_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("LAYER_STABILITY_TRACE layer_fsm_decoder_block_fixture stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE layer_fsm_decoder_block_fixture estimated_iob_bits=132 previous_decoder_block_iob=228 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b=0 exposed_kv_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL layer_fsm_decoder_block_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS layer_fsm_decoder_block_fixture");
        $finish;
    end
endmodule
