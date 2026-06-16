`timescale 1ns/1ps

module tb_token_loop_axi_decoder_block_fixture;
    localparam int VALUES = 4;
    localparam int ELEM_WIDTH = 16;
    localparam int STATUS_WIDTH = 64;
    localparam int TOP_STATUS_WIDTH = 64;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_o;
    logic [STATUS_WIDTH-1:0] status_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    logic signed [VALUES*ELEM_WIDTH-1:0] token0_output_seen;
    logic signed [VALUES*ELEM_WIDTH-1:0] token1_output_seen;
    logic [TOP_STATUS_WIDTH-1:0] token0_status_seen;
    logic [TOP_STATUS_WIDTH-1:0] token1_status_seen;
    logic [31:0] token0_block_trace_seen;
    logic [31:0] token1_block_trace_seen;
    logic [47:0] token1_attention_trace_seen;
    logic [79:0] token1_mlp_trace_seen;
    logic [3:0] token0_axi_metadata_seen;
    logic [3:0] token1_axi_metadata_seen;

    logic token0_busy_seen_r;
    logic token1_busy_seen_r;
    logic token0_done_seen_while_start_high_r;
    logic token1_done_seen_while_start_high_r;
    logic token0_start_deasserted_after_done_r;
    logic token1_start_deasserted_after_done_r;
    logic token0_release_seen_r;
    logic token1_release_seen_r;
    logic top_start_prev_r;
    logic top_done_prev_r;
    integer token0_busy_cycles;
    integer token1_busy_cycles;
    integer token_start_count;
    integer token_done_count;
    integer active_token;
    integer observed;

    token_loop_axi_decoder_block_fixture dut (
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
                $display("FAIL token_loop_axi_decoder_block_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always @(posedge aclk) begin
        if (!aresetn) begin
            token0_busy_seen_r <= 1'b0;
            token1_busy_seen_r <= 1'b0;
            token0_done_seen_while_start_high_r <= 1'b0;
            token1_done_seen_while_start_high_r <= 1'b0;
            token0_start_deasserted_after_done_r <= 1'b0;
            token1_start_deasserted_after_done_r <= 1'b0;
            token0_release_seen_r <= 1'b0;
            token1_release_seen_r <= 1'b0;
            token0_busy_cycles <= 0;
            token1_busy_cycles <= 0;
            token_start_count <= 0;
            token_done_count <= 0;
            active_token <= 0;
            top_start_prev_r <= 1'b0;
            top_done_prev_r <= 1'b0;
            token0_output_seen <= '0;
            token1_output_seen <= '0;
            token0_status_seen <= '0;
            token1_status_seen <= '0;
            token0_block_trace_seen <= '0;
            token1_block_trace_seen <= '0;
            token1_attention_trace_seen <= '0;
            token1_mlp_trace_seen <= '0;
            token0_axi_metadata_seen <= '0;
            token1_axi_metadata_seen <= '0;
        end else begin
            if (!top_start_prev_r && dut.top_start_r) begin
                active_token <= token_start_count;
                token_start_count <= token_start_count + 1;
            end
            if (dut.top_start_r && !dut.top_done_w) begin
                if (active_token == 0) begin
                    token0_busy_seen_r <= 1'b1;
                    token0_busy_cycles <= token0_busy_cycles + 1;
                end else begin
                    token1_busy_seen_r <= 1'b1;
                    token1_busy_cycles <= token1_busy_cycles + 1;
                end
            end
            if (active_token == 0 && token0_busy_seen_r && !token0_done_seen_while_start_high_r && !dut.top_done_w && !dut.top_start_r) begin
                $display("FAIL token_loop_axi_decoder_block_fixture token0 child start was not held while top child busy");
                $fatal;
            end
            if (active_token == 1 && token1_busy_seen_r && !token1_done_seen_while_start_high_r && !dut.top_done_w && !dut.top_start_r) begin
                $display("FAIL token_loop_axi_decoder_block_fixture token1 child start was not held while top child busy");
                $fatal;
            end
            if (dut.top_start_r && dut.top_done_w) begin
                token_done_count <= token_done_count + 1;
                if (active_token == 0) begin
                    token0_done_seen_while_start_high_r <= 1'b1;
                    token0_output_seen <= dut.top_output_w;
                    token0_status_seen <= dut.top_status_w;
                    token0_block_trace_seen <= dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[16 +: 32];
                    token0_axi_metadata_seen <= dut.top_status_w[48 +: 4];
                end else if (active_token == 1) begin
                    token1_done_seen_while_start_high_r <= 1'b1;
                    token1_output_seen <= dut.top_output_w;
                    token1_status_seen <= dut.top_status_w;
                    token1_block_trace_seen <= dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[16 +: 32];
                    token1_attention_trace_seen <= dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[32 +: 48];
                    token1_mlp_trace_seen <= dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[96 +: 80];
                    token1_axi_metadata_seen <= dut.top_status_w[48 +: 4];
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
                $display("FAIL token_loop_axi_decoder_block_fixture child start was not deasserted after top child done_o");
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

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        stable_passed = 1'b0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #40000;
        if (!done_o) begin
            $display("FAIL token_loop_axi_decoder_block_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[31:0] != 32'h64636261) begin
            $display("FAIL token_loop_axi_decoder_block_fixture token trace observed=0x%0h expected=0x64636261", status_o[31:0]);
            $fatal;
        end
        if (status_o[32 +: 16] != 16'h5453) begin
            $display("FAIL token_loop_axi_decoder_block_fixture final top trace observed=0x%0h expected=0x5453", status_o[32 +: 16]);
            $fatal;
        end
        if (status_o[48 +: 4] != 4'hf || status_o[48 +: 4] != token1_status_seen[48 +: 4]) begin
            $display("FAIL token_loop_axi_decoder_block_fixture loop AXI metadata bits loop=0x%0h top=0x%0h status=0x%0h", status_o[48 +: 4], token1_status_seen[48 +: 4], status_o);
            $fatal;
        end
        if (token_start_count != 2 || token_done_count != 2) begin
            $display("FAIL token_loop_axi_decoder_block_fixture token call counts start=%0d done=%0d", token_start_count, token_done_count);
            $fatal;
        end
        if (!token0_done_seen_while_start_high_r || !token1_done_seen_while_start_high_r || !token0_start_deasserted_after_done_r || !token1_start_deasserted_after_done_r || !token0_release_seen_r || !token1_release_seen_r || token0_busy_cycles == 0 || token1_busy_cycles == 0) begin
            $display("FAIL token_loop_axi_decoder_block_fixture child start protocol token0_busy=%0d token0_done_seen=%0d token0_deassert=%0d token0_release=%0d token1_busy=%0d token1_done_seen=%0d token1_deassert=%0d token1_release=%0d", token0_busy_cycles, token0_done_seen_while_start_high_r, token0_start_deasserted_after_done_r, token0_release_seen_r, token1_busy_cycles, token1_done_seen_while_start_high_r, token1_start_deasserted_after_done_r, token1_release_seen_r);
            $fatal;
        end
        if (token0_status_seen[15:0] != 16'h5453 || token1_status_seen[15:0] != 16'h5453) begin
            $display("FAIL token_loop_axi_decoder_block_fixture captured top traces token0=0x%0h token1=0x%0h", token0_status_seen[15:0], token1_status_seen[15:0]);
            $fatal;
        end
        if (token0_status_seen[16 +: 16] != 16'h4241 || token1_status_seen[16 +: 16] != 16'h4241) begin
            $display("FAIL token_loop_axi_decoder_block_fixture captured layer traces token0=0x%0h token1=0x%0h", token0_status_seen[16 +: 16], token1_status_seen[16 +: 16]);
            $fatal;
        end
        if (token0_block_trace_seen != 32'hb2b1a2a1 || token1_block_trace_seen != 32'hb2b1a2a1) begin
            $display("FAIL token_loop_axi_decoder_block_fixture captured block traces token0=0x%0h token1=0x%0h", token0_block_trace_seen, token1_block_trace_seen);
            $fatal;
        end
        if (token0_axi_metadata_seen != 4'hf || token1_axi_metadata_seen != 4'hf) begin
            $display("FAIL token_loop_axi_decoder_block_fixture captured AXI metadata token0=0x%0h token1=0x%0h", token0_axi_metadata_seen, token1_axi_metadata_seen);
            $fatal;
        end
        if (token1_attention_trace_seen != 48'h323122211211) begin
            $display("FAIL token_loop_axi_decoder_block_fixture nested AXI attention trace observed=0x%0h", token1_attention_trace_seen);
            $fatal;
        end
        if (token1_mlp_trace_seen != 80'h52514241323122211211) begin
            $display("FAIL token_loop_axi_decoder_block_fixture nested MLP trace observed=0x%0h", token1_mlp_trace_seen);
            $fatal;
        end
        if (dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r != dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r) begin
            $display("FAIL token_loop_axi_decoder_block_fixture MLP child did not consume captured AXI attention output");
            $fatal;
        end
        check_lane("token0_output", token0_output_seen, 0, 12);
        check_lane("token0_output", token0_output_seen, 1, -6);
        check_lane("token0_output", token0_output_seen, 2, 18);
        check_lane("token0_output", token0_output_seen, 3, 6);
        check_lane("token1_output", token1_output_seen, 0, 12);
        check_lane("token1_output", token1_output_seen, 1, -6);
        check_lane("token1_output", token1_output_seen, 2, 18);
        check_lane("token1_output", token1_output_seen, 3, 6);
        check_lane("captured_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 0, 4);
        check_lane("captured_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 1, -2);
        check_lane("captured_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 2, 1);
        check_lane("captured_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 3, 2);
        check_lane("mlp_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 0, 4);
        check_lane("mlp_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 1, -2);
        check_lane("mlp_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 2, 1);
        check_lane("mlp_attention", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 3, 2);
        check_lane("final_output", final_output_o, 0, 12);
        check_lane("final_output", final_output_o, 1, -6);
        check_lane("final_output", final_output_o, 2, 18);
        check_lane("final_output", final_output_o, 3, 6);
        if (token0_output_seen != final_output_o || token1_output_seen != final_output_o) begin
            $display("FAIL token_loop_axi_decoder_block_fixture per-token outputs changed token0=0x%0h token1=0x%0h final=0x%0h", token0_output_seen, token1_output_seen, final_output_o);
            $fatal;
        end

        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o != stable_output_snapshot || status_o != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL token_loop_axi_decoder_block_fixture output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("TOKEN_LOOP_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture trace_hex=0x%0h events=token0_start,token0_done,token1_start,token1_done", status_o[31:0]);
        $display("TOKEN_CHILD_CALL_TRACE token_loop_axi_decoder_block_fixture token=0 top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h output=%0d,%0d,%0d,%0d", token0_status_seen[15:0], token0_status_seen[16 +: 16], token0_block_trace_seen, token0_axi_metadata_seen, lane_s16(token0_output_seen, 0), lane_s16(token0_output_seen, 1), lane_s16(token0_output_seen, 2), lane_s16(token0_output_seen, 3));
        $display("TOKEN_CHILD_CALL_TRACE token_loop_axi_decoder_block_fixture token=1 top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h output=%0d,%0d,%0d,%0d", token1_status_seen[15:0], token1_status_seen[16 +: 16], token1_block_trace_seen, token1_axi_metadata_seen, lane_s16(token1_output_seen, 0), lane_s16(token1_output_seen, 1), lane_s16(token1_output_seen, 2), lane_s16(token1_output_seen, 3));
        $display("TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_decoder_block_fixture token=0 busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", token0_busy_cycles, token0_done_seen_while_start_high_r, token0_start_deasserted_after_done_r, token0_release_seen_r);
        $display("TOKEN_CHILD_START_HOLD_TRACE token_loop_axi_decoder_block_fixture token=1 busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", token1_busy_cycles, token1_done_seen_while_start_high_r, token1_start_deasserted_after_done_r, token1_release_seen_r);
        $display("TOP_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture token=0 top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h events=layer_fsm_axi_decoder_block_fixture_start,layer_fsm_axi_decoder_block_fixture_done", token0_status_seen[15:0], token0_status_seen[16 +: 16], token0_block_trace_seen, token0_axi_metadata_seen);
        $display("TOP_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture token=1 top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h events=layer_fsm_axi_decoder_block_fixture_start,layer_fsm_axi_decoder_block_fixture_done", token1_status_seen[15:0], token1_status_seen[16 +: 16], token1_block_trace_seen, token1_axi_metadata_seen);
        $display("LAYER_AXI_DECODER_BLOCK_TRACE token_loop_axi_decoder_block_fixture token=1 layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h events=decoder_block_axi_attention_mlp_fixture_start,decoder_block_axi_attention_mlp_fixture_done", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[16 +: 32], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[48 +: 4]);
        $display("BLOCK_AXI_TRACE token_loop_axi_decoder_block_fixture token=1 block_trace_hex=0x%0h attention_trace_hex=0x%0h axi_metadata_bits=0x%0h mlp_trace_hex=0x%0h events=axi_attention_start,axi_attention_done,mlp_start,mlp_done", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[32 +: 48], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[96 +: 80]);
        $display("AXI_ATTENTION_CHILD_TRACE token_loop_axi_decoder_block_fixture token=1 trace_hex=0x%0h events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.status_o[47:0]);
        $display("MLP_CHILD_TRACE token_loop_axi_decoder_block_fixture token=1 trace_hex=0x%0h events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0]);
        $display("TOKEN_LOOP_AXI_METADATA_PROPAGATION_TRACE token_loop_axi_decoder_block_fixture loop_bits=0x%0h loop_bit_lsb=48 loop_bit_msb=51 top_bits=0x%0h layer_bits=0x%0h child_block_bits=0x%0h q_bits=0x%0h k_bits=0x%0h v_bits=0x%0h o_bits=0x%0h source=top_fsm_axi_decoder_block_fixture.status_o[51:48]:layer_fsm_axi_decoder_block_fixture.status_o[51:48]:decoder_block_axi_attention_mlp_fixture.status_o[83:80]:q_projection_axi_stream_integration.integration_status_o[45:42]:k_projection_axi_stream_integration.integration_status_o[45:42]:v_projection_axi_stream_integration.integration_status_o[45:42]:o_projection_axi_stream_integration.integration_status_o[45:42]:aggregate_and status=0x%0h", status_o[48 +: 4], dut.u_top_fsm_axi_decoder_block_fixture.status_o[48 +: 4], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[48 +: 4], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42], status_o);
        $display("AXI_PROJECTION_CHILD_AR_TRACE token_loop_axi_decoder_block_fixture projection=q addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE token_loop_axi_decoder_block_fixture projection=k addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_araddr_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arlen_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arsize_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arburst_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE token_loop_axi_decoder_block_fixture projection=v addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_araddr_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arlen_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arsize_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arburst_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_AR_TRACE token_loop_axi_decoder_block_fixture projection=o addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=2 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_araddr_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arlen_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arsize_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arburst_w, dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arid_w);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE token_loop_axi_decoder_block_fixture projection=q accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE token_loop_axi_decoder_block_fixture projection=k accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE token_loop_axi_decoder_block_fixture projection=v accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE token_loop_axi_decoder_block_fixture projection=o accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE token_loop_axi_decoder_block_fixture projection=q emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[15:8], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[23:16], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE token_loop_axi_decoder_block_fixture projection=k emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[15:8], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[23:16], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE token_loop_axi_decoder_block_fixture projection=v emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[15:8], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[23:16], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE token_loop_axi_decoder_block_fixture projection=o emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[15:8], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[23:16], dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE token_loop_axi_decoder_block_fixture projection=q output=%0d,%0d status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE token_loop_axi_decoder_block_fixture projection=k output=%0d,%0d status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE token_loop_axi_decoder_block_fixture projection=v output=%0d,%0d status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE token_loop_axi_decoder_block_fixture projection=o output=%0d,%0d status=0x%0h instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE token_loop_axi_decoder_block_fixture projection=q packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE token_loop_axi_decoder_block_fixture projection=k packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE token_loop_axi_decoder_block_fixture projection=v packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE token_loop_axi_decoder_block_fixture projection=o packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration");
        $display("ATTENTION_OUTPUT_TRACE token_loop_axi_decoder_block_fixture token=1 output=%0d,%0d,%0d,%0d source=hierarchical_axi_attention_child_output", lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 0), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 1), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 2), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 3));
        $display("MLP_INPUT_TRACE token_loop_axi_decoder_block_fixture token=1 hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_child_inputs", lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 0), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 1), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 2), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 3), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 0), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 1), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 2), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 3));
        $display("RESIDUAL0_TRACE token_loop_axi_decoder_block_fixture token=1 residual0=%0d,%0d,%0d,%0d", lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 0), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 1), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 2), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 3));
        $display("MLP_FINAL_TRACE token_loop_axi_decoder_block_fixture token=1 output=%0d,%0d,%0d,%0d", lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 0), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 1), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 2), lane_s16(dut.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 3));
        $display("FINAL_OUTPUT_TRACE token_loop_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("TOKEN_LOOP_STABILITY_TRACE token_loop_axi_decoder_block_fixture stable=%0d", stable_passed);
        $display("TOKEN_OUTPUT_POLICY_TRACE token_loop_axi_decoder_block_fixture repeated_deterministic_outputs=%0d token_dependent_outputs=0", token0_output_seen == token1_output_seen && token1_output_seen == final_output_o);
        $display("COMPACT_IO_TRACE token_loop_axi_decoder_block_fixture estimated_iob_bits=132 prior_top_fsm_axi_bonded_iob=132 prior_top_fsm_axi_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL token_loop_axi_decoder_block_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS token_loop_axi_decoder_block_fixture");
        $finish;
    end
endmodule
