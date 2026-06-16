`timescale 1ns/1ps

module tb_ddr_axi_board_shell_fixture;
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

    logic model_busy_seen_r;
    logic model_done_seen_while_start_high_r;
    logic model_start_deasserted_after_done_r;
    logic model_release_seen_r;
    logic model_done_prev_r;
    integer model_busy_cycles;
    integer observed;

    ddr_axi_board_shell_fixture dut (
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
                $display("FAIL ddr_axi_board_shell_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            model_busy_seen_r <= 1'b0;
            model_done_seen_while_start_high_r <= 1'b0;
            model_start_deasserted_after_done_r <= 1'b0;
            model_release_seen_r <= 1'b0;
            model_done_prev_r <= 1'b0;
            model_busy_cycles <= 0;
        end else begin
            if (dut.model_start_r && !dut.model_done_w) begin
                model_busy_seen_r <= 1'b1;
                model_busy_cycles <= model_busy_cycles + 1;
            end
            if (model_busy_seen_r && !model_done_seen_while_start_high_r && !dut.model_done_w && !dut.model_start_r) begin
                $display("FAIL ddr_axi_board_shell_fixture model child start was not held while busy");
                $fatal;
            end
            if (dut.model_start_r && dut.model_done_w) begin
                model_done_seen_while_start_high_r <= 1'b1;
            end
            if (model_done_prev_r && !dut.model_start_r) begin
                model_start_deasserted_after_done_r <= 1'b1;
            end
            if (model_done_prev_r && dut.model_start_r) begin
                $display("FAIL ddr_axi_board_shell_fixture model child start was not deasserted after done_o");
                $fatal;
            end
            if (model_start_deasserted_after_done_r && !dut.model_done_w) begin
                model_release_seen_r <= 1'b1;
            end
            model_done_prev_r <= dut.model_done_w;
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
        #20000;
        if (!done_o) begin
            $display("FAIL ddr_axi_board_shell_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[7:0] != 8'h81 || status_o[15:8] != 8'h82) begin
            $display("FAIL ddr_axi_board_shell_fixture shell trace observed=0x%0h", status_o[15:0]);
            $fatal;
        end
        if (status_o[55:50] != 6'd7 || status_o[45:39] != 7'h7f) begin
            $display("FAIL ddr_axi_board_shell_fixture projection compact fields count=%0d request_mask=0x%0h", status_o[55:50], status_o[45:39]);
            $fatal;
        end
        if (status_o[35:32] != 4'hf || status_o[38:36] != 3'h7) begin
            $display("FAIL ddr_axi_board_shell_fixture attention/mlp masks attention=0x%0h mlp=0x%0h", status_o[35:32], status_o[38:36]);
            $fatal;
        end
        if (status_o[49:46] != 4'hf || status_o[49:46] != dut.model_status_w[48 +: 4]) begin
            $display("FAIL ddr_axi_board_shell_fixture model AXI metadata propagation shell=0x%0h model=0x%0h", status_o[49:46], dut.model_status_w[48 +: 4]);
            $fatal;
        end
        if (dut.u_model_fsm_axi_decoder_block_fixture.status_o[31:0] != 32'h74737271) begin
            $display("FAIL ddr_axi_board_shell_fixture model trace observed=0x%0h", dut.u_model_fsm_axi_decoder_block_fixture.status_o[31:0]);
            $fatal;
        end
        if (dut.u_model_fsm_axi_decoder_block_fixture.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4] != 4'hf) begin
            $display("FAIL ddr_axi_board_shell_fixture nested q/k/v/o AXI metadata bits not propagated");
            $fatal;
        end
        if (!model_done_seen_while_start_high_r || !model_start_deasserted_after_done_r || !model_release_seen_r || model_busy_cycles == 0) begin
            $display("FAIL ddr_axi_board_shell_fixture model start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", model_busy_cycles, model_done_seen_while_start_high_r, model_start_deasserted_after_done_r, model_release_seen_r);
            $fatal;
        end

        check_lane("model_child_output", dut.u_model_fsm_axi_decoder_block_fixture.final_output_o, 0, 12);
        check_lane("model_child_output", dut.u_model_fsm_axi_decoder_block_fixture.final_output_o, 1, -6);
        check_lane("model_child_output", dut.u_model_fsm_axi_decoder_block_fixture.final_output_o, 2, 18);
        check_lane("model_child_output", dut.u_model_fsm_axi_decoder_block_fixture.final_output_o, 3, 6);
        check_lane("final_output", final_output_o, 0, 12);
        check_lane("final_output", final_output_o, 1, -6);
        check_lane("final_output", final_output_o, 2, 18);
        check_lane("final_output", final_output_o, 3, 6);

        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o != stable_output_snapshot || status_o != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL ddr_axi_board_shell_fixture output/status changed while shell done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("DDR_AXI_BOARD_SHELL_TRACE ddr_axi_board_shell_fixture shell_trace_hex=0x%0h model_trace_hex=0x%0h request_mask=0x%0h attention_mask=0x%0h mlp_mask=0x%0h events=model_fsm_axi_decoder_block_fixture_start,model_fsm_axi_decoder_block_fixture_done", status_o[15:0], dut.u_model_fsm_axi_decoder_block_fixture.status_o[31:0], status_o[45:39], status_o[35:32], status_o[38:36]);
        $display("DDR_AXI_MODEL_CHILD_START_HOLD_TRACE ddr_axi_board_shell_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", model_busy_cycles, model_done_seen_while_start_high_r, model_start_deasserted_after_done_r, model_release_seen_r);
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=q_proj decoder_use=attention rows=2048 cols=2048 packed_int4_bytes=2097152 memory_beats=131072 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=0 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=k_proj decoder_use=attention rows=512 cols=2048 packed_int4_bytes=524288 memory_beats=32768 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=1 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=v_proj decoder_use=attention rows=512 cols=2048 packed_int4_bytes=524288 memory_beats=32768 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=2 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=o_proj decoder_use=attention rows=2048 cols=2048 packed_int4_bytes=2097152 memory_beats=131072 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=3 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=gate_proj decoder_use=mlp rows=8192 cols=2048 packed_int4_bytes=8388608 memory_beats=524288 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=4 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=up_proj decoder_use=mlp rows=8192 cols=2048 packed_int4_bytes=8388608 memory_beats=524288 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=5 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_PROJECTION_REQUEST_TRACE ddr_axi_board_shell_fixture projection=down_proj decoder_use=mlp rows=2048 cols=8192 packed_int4_bytes=8388608 memory_beats=524288 layout_dependency=blocked_by_real_gptq_weight_layout_preflight payload_dependency=blocked_by_gptq_payload_probe request_bit=6 status_bit=1 source=projection_weight_stream_plan");
        $display("DDR_AXI_COMPACT_STATUS_TRACE ddr_axi_board_shell_fixture projection_count=%0d model_bits=0x%0h request_mask=0x%0h attention_mask=0x%0h mlp_mask=0x%0h status=0x%0h", status_o[55:50], status_o[49:46], status_o[45:39], status_o[35:32], status_o[38:36], status_o);
        $display("FINAL_OUTPUT_TRACE ddr_axi_board_shell_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("DDR_AXI_BOARD_SHELL_STABILITY_TRACE ddr_axi_board_shell_fixture stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE ddr_axi_board_shell_fixture estimated_iob_bits=132 prior_model_fsm_axi_bonded_iob=132 prior_model_fsm_axi_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0 board_io_constraints=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL ddr_axi_board_shell_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS ddr_axi_board_shell_fixture");
        $finish;
    end
endmodule
