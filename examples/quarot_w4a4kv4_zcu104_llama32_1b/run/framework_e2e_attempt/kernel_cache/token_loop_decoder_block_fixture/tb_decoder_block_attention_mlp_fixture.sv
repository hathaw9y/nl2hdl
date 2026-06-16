`timescale 1ns/1ps

module tb_decoder_block_attention_mlp_fixture;
    localparam int VALUES = 4;
    localparam int ELEM_WIDTH = 16;
    localparam int STATUS_WIDTH = 160;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_o;
    logic [STATUS_WIDTH-1:0] status_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    logic attention_busy_seen_r;
    logic attention_done_seen_while_start_high_r;
    logic attention_start_deasserted_after_done_r;
    logic attention_release_seen_r;
    logic attention_done_prev_r;
    integer attention_busy_cycles;

    logic mlp_busy_seen_r;
    logic mlp_done_seen_while_start_high_r;
    logic mlp_start_deasserted_after_done_r;
    logic mlp_release_seen_r;
    logic mlp_done_prev_r;
    integer mlp_busy_cycles;

    integer observed;

    decoder_block_attention_mlp_fixture dut (
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

    function automatic integer lane_s8(input logic signed [31:0] vec_i, input integer idx_i);
        logic signed [7:0] tmp;
        begin
            tmp = vec_i[idx_i*8 +: 8];
            lane_s8 = int'($signed(tmp));
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
                $display("FAIL decoder_block_attention_mlp_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always @(posedge aclk) begin
        if (!aresetn) begin
            attention_busy_seen_r <= 1'b0;
            attention_done_seen_while_start_high_r <= 1'b0;
            attention_start_deasserted_after_done_r <= 1'b0;
            attention_release_seen_r <= 1'b0;
            attention_done_prev_r <= 1'b0;
            attention_busy_cycles <= 0;
        end else begin
            if (dut.attention_start_r && !dut.attention_done_w) begin
                attention_busy_seen_r <= 1'b1;
                attention_busy_cycles <= attention_busy_cycles + 1;
            end
            if (attention_busy_seen_r && !attention_done_seen_while_start_high_r && !dut.attention_done_w && !dut.attention_start_r) begin
                $display("FAIL decoder_block_attention_mlp_fixture attention child start was not held while busy");
                $fatal;
            end
            if (dut.attention_start_r && dut.attention_done_w) begin
                attention_done_seen_while_start_high_r <= 1'b1;
            end
            if (attention_done_prev_r && !dut.attention_start_r) begin
                attention_start_deasserted_after_done_r <= 1'b1;
            end
            if (attention_done_prev_r && dut.attention_start_r) begin
                $display("FAIL decoder_block_attention_mlp_fixture attention child start was not deasserted after done_o");
                $fatal;
            end
            if (attention_start_deasserted_after_done_r && !dut.attention_done_w) begin
                attention_release_seen_r <= 1'b1;
            end
            attention_done_prev_r <= dut.attention_done_w;
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
                $display("FAIL decoder_block_attention_mlp_fixture mlp child start was not held while busy");
                $fatal;
            end
            if (dut.mlp_start_r && dut.mlp_done_w) begin
                mlp_done_seen_while_start_high_r <= 1'b1;
            end
            if (mlp_done_prev_r && !dut.mlp_start_r) begin
                mlp_start_deasserted_after_done_r <= 1'b1;
            end
            if (mlp_done_prev_r && dut.mlp_start_r) begin
                $display("FAIL decoder_block_attention_mlp_fixture mlp child start was not deasserted after done_o");
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
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #10000;
        if (!done_o) begin
            $display("FAIL decoder_block_attention_mlp_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[31:0] != 32'hb2b1a2a1) begin
            $display("FAIL decoder_block_attention_mlp_fixture block trace observed=0x%0h expected=0xb2b1a2a1", status_o[31:0]);
            $fatal;
        end
        if (status_o[32 +: 48] != 48'h323122211211) begin
            $display("FAIL decoder_block_attention_mlp_fixture attention trace observed=0x%0h", status_o[32 +: 48]);
            $fatal;
        end
        if (status_o[80 +: 80] != 80'h52514241323122211211) begin
            $display("FAIL decoder_block_attention_mlp_fixture mlp trace observed=0x%0h", status_o[80 +: 80]);
            $fatal;
        end
        if (!attention_done_seen_while_start_high_r || !attention_start_deasserted_after_done_r || !attention_release_seen_r || attention_busy_cycles == 0) begin
            $display("FAIL decoder_block_attention_mlp_fixture attention start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", attention_busy_cycles, attention_done_seen_while_start_high_r, attention_start_deasserted_after_done_r, attention_release_seen_r);
            $fatal;
        end
        if (!mlp_done_seen_while_start_high_r || !mlp_start_deasserted_after_done_r || !mlp_release_seen_r || mlp_busy_cycles == 0) begin
            $display("FAIL decoder_block_attention_mlp_fixture mlp start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", mlp_busy_cycles, mlp_done_seen_while_start_high_r, mlp_start_deasserted_after_done_r, mlp_release_seen_r);
            $fatal;
        end

        check_lane("captured_attention", dut.captured_attention_r, 0, 4);
        check_lane("captured_attention", dut.captured_attention_r, 1, -2);
        check_lane("captured_attention", dut.captured_attention_r, 2, 1);
        check_lane("captured_attention", dut.captured_attention_r, 3, 2);
        if (dut.captured_attention_r != dut.u_decoder_child_attention_datapath.output_o) begin
            $display("FAIL decoder_block_attention_mlp_fixture captured attention did not match attention child output");
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
            $display("FAIL decoder_block_attention_mlp_fixture output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("BLOCK_TRACE decoder_block_attention_mlp_fixture block_trace_hex=0x%0h attention_trace_hex=0x%0h mlp_trace_hex=0x%0h events=attention_start,attention_done,mlp_start,mlp_done", status_o[31:0], status_o[32 +: 48], status_o[80 +: 80]);
        $display("ATTENTION_CHILD_TRACE decoder_block_attention_mlp_fixture trace_hex=0x%0h events=source_path_start,source_path_done,projection_shell_start,projection_shell_done,attention_kv_start,attention_kv_done", dut.u_decoder_child_attention_datapath.status_o[47:0]);
        $display("MLP_CHILD_TRACE decoder_block_attention_mlp_fixture trace_hex=0x%0h events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done", dut.u_residual_mlp_fixture.status_o[79:0]);
        $display("ATTENTION_CHILD_START_HOLD_TRACE decoder_block_attention_mlp_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", attention_busy_cycles, attention_done_seen_while_start_high_r, attention_start_deasserted_after_done_r, attention_release_seen_r);
        $display("MLP_CHILD_START_HOLD_TRACE decoder_block_attention_mlp_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", mlp_busy_cycles, mlp_done_seen_while_start_high_r, mlp_start_deasserted_after_done_r, mlp_release_seen_r);
        $display("ATTENTION_OUTPUT_TRACE decoder_block_attention_mlp_fixture output=%0d,%0d,%0d,%0d source=hierarchical_attention_child_output", lane_s16(dut.captured_attention_r, 0), lane_s16(dut.captured_attention_r, 1), lane_s16(dut.captured_attention_r, 2), lane_s16(dut.captured_attention_r, 3));
        $display("MLP_INPUT_TRACE decoder_block_attention_mlp_fixture hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_child_inputs", lane_s16(dut.u_residual_mlp_fixture.hidden_r, 0), lane_s16(dut.u_residual_mlp_fixture.hidden_r, 1), lane_s16(dut.u_residual_mlp_fixture.hidden_r, 2), lane_s16(dut.u_residual_mlp_fixture.hidden_r, 3), lane_s16(dut.u_residual_mlp_fixture.attention_r, 0), lane_s16(dut.u_residual_mlp_fixture.attention_r, 1), lane_s16(dut.u_residual_mlp_fixture.attention_r, 2), lane_s16(dut.u_residual_mlp_fixture.attention_r, 3));
        $display("RESIDUAL0_TRACE decoder_block_attention_mlp_fixture residual0=%0d,%0d,%0d,%0d", lane_s16(dut.u_residual_mlp_fixture.residual0_r, 0), lane_s16(dut.u_residual_mlp_fixture.residual0_r, 1), lane_s16(dut.u_residual_mlp_fixture.residual0_r, 2), lane_s16(dut.u_residual_mlp_fixture.residual0_r, 3));
        $display("GATE_UP_TRACE decoder_block_attention_mlp_fixture gate=%0d,%0d,%0d,%0d up=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_fixture_constant_matrices", lane_s16(dut.u_residual_mlp_fixture.gate_r, 0), lane_s16(dut.u_residual_mlp_fixture.gate_r, 1), lane_s16(dut.u_residual_mlp_fixture.gate_r, 2), lane_s16(dut.u_residual_mlp_fixture.gate_r, 3), lane_s16(dut.u_residual_mlp_fixture.up_r, 0), lane_s16(dut.u_residual_mlp_fixture.up_r, 1), lane_s16(dut.u_residual_mlp_fixture.up_r, 2), lane_s16(dut.u_residual_mlp_fixture.up_r, 3));
        $display("SWIGLU_TRACE decoder_block_attention_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=%0d,%0d,%0d,%0d true_silu_exp=0", lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 0), lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 1), lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 2), lane_s16(dut.u_residual_mlp_fixture.swiglu_r, 3));
        $display("DOWN_TRACE decoder_block_attention_mlp_fixture down=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_fixture_constant_matrix", lane_s16(dut.u_residual_mlp_fixture.down_r, 0), lane_s16(dut.u_residual_mlp_fixture.down_r, 1), lane_s16(dut.u_residual_mlp_fixture.down_r, 2), lane_s16(dut.u_residual_mlp_fixture.down_r, 3));
        $display("MLP_FINAL_TRACE decoder_block_attention_mlp_fixture output=%0d,%0d,%0d,%0d", lane_s16(dut.u_residual_mlp_fixture.final_output_o, 0), lane_s16(dut.u_residual_mlp_fixture.final_output_o, 1), lane_s16(dut.u_residual_mlp_fixture.final_output_o, 2), lane_s16(dut.u_residual_mlp_fixture.final_output_o, 3));
        $display("FINAL_OUTPUT_TRACE decoder_block_attention_mlp_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("DECODER_BLOCK_STABILITY_TRACE decoder_block_attention_mlp_fixture stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE decoder_block_attention_mlp_fixture estimated_iob_bits=228 residual_standalone_iob_reference=292 exposed_128b=0 exposed_kv_arrays=0 exposed_hidden_ports=0 exposed_child_status_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL decoder_block_attention_mlp_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS decoder_block_attention_mlp_fixture");
        $finish;
    end
endmodule
