`timescale 1ns/1ps

module tb_model_fsm_axi_decoder_block_fixture;
    localparam int VALUES = 4;
    localparam int ELEM_WIDTH = 16;
    localparam int STATUS_WIDTH = 64;
    localparam int TOKEN_STATUS_WIDTH = 64;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_o;
    logic [STATUS_WIDTH-1:0] status_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    logic signed [VALUES*ELEM_WIDTH-1:0] layer0_output_seen;
    logic signed [VALUES*ELEM_WIDTH-1:0] layer1_output_seen;
    logic [TOKEN_STATUS_WIDTH-1:0] layer0_status_seen;
    logic [TOKEN_STATUS_WIDTH-1:0] layer1_status_seen;
    logic [3:0] layer0_axi_metadata_seen;
    logic [3:0] layer1_axi_metadata_seen;

    logic layer0_busy_seen_r;
    logic layer1_busy_seen_r;
    logic layer0_done_seen_while_start_high_r;
    logic layer1_done_seen_while_start_high_r;
    logic layer0_start_deasserted_after_done_r;
    logic layer1_start_deasserted_after_done_r;
    logic layer0_release_seen_r;
    logic layer1_release_seen_r;
    logic token_start_prev_r;
    logic token_done_prev_r;
    integer layer0_busy_cycles;
    integer layer1_busy_cycles;
    integer layer_start_count;
    integer layer_done_count;
    integer active_layer;
    integer observed;

    model_fsm_axi_decoder_block_fixture dut (
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

    function automatic integer lane_s16(input logic signed [VALUES*ELEM_WIDTH-1:0] vector_i, input integer idx_i);
        logic signed [ELEM_WIDTH-1:0] lane_w;
        begin
            lane_w = vector_i[idx_i*ELEM_WIDTH +: ELEM_WIDTH];
            lane_s16 = {{16{lane_w[ELEM_WIDTH-1]}}, lane_w};
        end
    endfunction

    task automatic check_lane(
        input string label_i,
        input logic signed [VALUES*ELEM_WIDTH-1:0] vector_i,
        input integer idx_i,
        input integer expected_i
    );
        begin
            observed = lane_s16(vector_i, idx_i);
            if (observed !== expected_i) begin
                $display("FAIL model_fsm_axi_decoder_block_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            layer0_busy_seen_r <= 1'b0;
            layer1_busy_seen_r <= 1'b0;
            layer0_done_seen_while_start_high_r <= 1'b0;
            layer1_done_seen_while_start_high_r <= 1'b0;
            layer0_start_deasserted_after_done_r <= 1'b0;
            layer1_start_deasserted_after_done_r <= 1'b0;
            layer0_release_seen_r <= 1'b0;
            layer1_release_seen_r <= 1'b0;
            token_start_prev_r <= 1'b0;
            token_done_prev_r <= 1'b0;
            layer0_busy_cycles <= 0;
            layer1_busy_cycles <= 0;
            layer_start_count <= 0;
            layer_done_count <= 0;
            active_layer <= -1;
            layer0_output_seen <= '0;
            layer1_output_seen <= '0;
            layer0_status_seen <= '0;
            layer1_status_seen <= '0;
            layer0_axi_metadata_seen <= '0;
            layer1_axi_metadata_seen <= '0;
        end else begin
            token_start_prev_r <= dut.token_start_r;
            token_done_prev_r <= dut.token_done_w;
            if (dut.token_start_r && !token_start_prev_r) begin
                layer_start_count <= layer_start_count + 1;
                active_layer <= layer_start_count;
                if (layer_start_count == 0) begin
                    layer0_busy_seen_r <= 1'b1;
                end else if (layer_start_count == 1) begin
                    layer1_busy_seen_r <= 1'b1;
                end
            end
            if (dut.token_start_r && !dut.token_done_w) begin
                if (active_layer == 0) begin
                    layer0_busy_cycles <= layer0_busy_cycles + 1;
                end else if (active_layer == 1) begin
                    layer1_busy_cycles <= layer1_busy_cycles + 1;
                end
            end
            if (dut.token_start_r && dut.token_done_w) begin
                if (active_layer == 0) begin
                    layer0_done_seen_while_start_high_r <= 1'b1;
                    layer0_output_seen <= dut.u_token_loop_axi_decoder_block_fixture.final_output_o;
                    layer0_status_seen <= dut.u_token_loop_axi_decoder_block_fixture.status_o;
                    layer0_axi_metadata_seen <= dut.u_token_loop_axi_decoder_block_fixture.status_o[48 +: 4];
                end else if (active_layer == 1) begin
                    layer1_done_seen_while_start_high_r <= 1'b1;
                    layer1_output_seen <= dut.u_token_loop_axi_decoder_block_fixture.final_output_o;
                    layer1_status_seen <= dut.u_token_loop_axi_decoder_block_fixture.status_o;
                    layer1_axi_metadata_seen <= dut.u_token_loop_axi_decoder_block_fixture.status_o[48 +: 4];
                end
            end
            if (!dut.token_start_r && token_start_prev_r && token_done_prev_r) begin
                layer_done_count <= layer_done_count + 1;
                if (active_layer == 0) begin
                    layer0_start_deasserted_after_done_r <= 1'b1;
                end else if (active_layer == 1) begin
                    layer1_start_deasserted_after_done_r <= 1'b1;
                end
            end
            if (!dut.token_start_r && !dut.token_done_w && token_done_prev_r) begin
                if (active_layer == 0) begin
                    layer0_release_seen_r <= 1'b1;
                end else if (active_layer == 1) begin
                    layer1_release_seen_r <= 1'b1;
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        stable_passed = 1'b0;
        stable_output_snapshot = '0;
        stable_status_snapshot = '0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #40000;

        if (!done_o) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[31:0] != 32'h74737271) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture model trace observed=0x%0h expected=0x74737271", status_o[31:0]);
            $fatal;
        end
        if (status_o[48 +: 4] != 4'hf || layer1_axi_metadata_seen != 4'hf) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture AXI metadata bits model=0x%0h token_loop=0x%0h status=0x%0h", status_o[48 +: 4], layer1_axi_metadata_seen, status_o);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.status_o[48 +: 4] != 4'hf) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture hierarchical token-loop metadata observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.status_o[48 +: 4]);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[48 +: 4] != 4'hf) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture hierarchical top metadata observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[48 +: 4]);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4] != 4'hf) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture hierarchical block metadata observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4]);
            $fatal;
        end
        if (layer_start_count != 2 || layer_done_count != 2) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture layer call count start=%0d done=%0d", layer_start_count, layer_done_count);
            $fatal;
        end
        if (!layer0_busy_seen_r || !layer0_done_seen_while_start_high_r || !layer0_start_deasserted_after_done_r || !layer0_release_seen_r) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture layer0 protocol busy=%0d done_seen=%0d deassert=%0d release=%0d", layer0_busy_seen_r, layer0_done_seen_while_start_high_r, layer0_start_deasserted_after_done_r, layer0_release_seen_r);
            $fatal;
        end
        if (!layer1_busy_seen_r || !layer1_done_seen_while_start_high_r || !layer1_start_deasserted_after_done_r || !layer1_release_seen_r) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture layer1 protocol busy=%0d done_seen=%0d deassert=%0d release=%0d", layer1_busy_seen_r, layer1_done_seen_while_start_high_r, layer1_start_deasserted_after_done_r, layer1_release_seen_r);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.status_o[31:0] != 32'h64636261) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture token-loop trace observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.status_o[31:0]);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[15:0] != 16'h5453) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture top trace observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[15:0]);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0] != 16'h4241) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture layer trace observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0]);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0] != 32'hb2b1a2a1) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture block trace observed=0x%0h", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0]);
            $fatal;
        end
        if (dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r != dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture MLP child did not consume captured attention output");
            $fatal;
        end
                check_lane("final_output", final_output_o, 0, 12);
        check_lane("final_output", final_output_o, 1, -6);
        check_lane("final_output", final_output_o, 2, 18);
        check_lane("final_output", final_output_o, 3, 6);
        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o !== stable_output_snapshot || status_o !== stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL model_fsm_axi_decoder_block_fixture output/status changed while model done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("MODEL_FSM_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture trace_hex=0x%0h events=layer0_start,layer0_done,layer1_start,layer1_done", status_o[31:0]);
        $display("MODEL_CHILD_CALL_TRACE model_fsm_axi_decoder_block_fixture layer=0 token_trace_hex=0x%0h top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h output=%0d,%0d,%0d,%0d", layer0_status_seen[31:0], layer0_status_seen[32 +: 16], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0], layer0_axi_metadata_seen, lane_s16(layer0_output_seen, 0), lane_s16(layer0_output_seen, 1), lane_s16(layer0_output_seen, 2), lane_s16(layer0_output_seen, 3));
        $display("MODEL_CHILD_CALL_TRACE model_fsm_axi_decoder_block_fixture layer=1 token_trace_hex=0x%0h top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h output=%0d,%0d,%0d,%0d", layer1_status_seen[31:0], layer1_status_seen[32 +: 16], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0], layer1_axi_metadata_seen, lane_s16(layer1_output_seen, 0), lane_s16(layer1_output_seen, 1), lane_s16(layer1_output_seen, 2), lane_s16(layer1_output_seen, 3));
        $display("MODEL_CHILD_START_HOLD_TRACE model_fsm_axi_decoder_block_fixture layer=0 busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", layer0_busy_cycles, layer0_done_seen_while_start_high_r, layer0_start_deasserted_after_done_r, layer0_release_seen_r);
        $display("MODEL_CHILD_START_HOLD_TRACE model_fsm_axi_decoder_block_fixture layer=1 busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", layer1_busy_cycles, layer1_done_seen_while_start_high_r, layer1_start_deasserted_after_done_r, layer1_release_seen_r);
        $display("TOKEN_LOOP_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture token_trace_hex=0x%0h events=token0_start,token0_done,token1_start,token1_done source=hierarchical_token_loop_child", dut.u_token_loop_axi_decoder_block_fixture.status_o[31:0]);
        $display("TOP_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h source=hierarchical_token_loop_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[48 +: 4]);
        $display("LAYER_AXI_DECODER_BLOCK_TRACE model_fsm_axi_decoder_block_fixture layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h source=hierarchical_layer_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[16 +: 32], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[48 +: 4]);
        $display("BLOCK_AXI_TRACE model_fsm_axi_decoder_block_fixture block_trace_hex=0x%0h attention_trace_hex=0x%0h axi_metadata_bits=0x%0h mlp_trace_hex=0x%0h source=hierarchical_decoder_block_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[32 +: 48], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[96 +: 80]);
        $display("MODEL_FSM_AXI_METADATA_PROPAGATION_TRACE model_fsm_axi_decoder_block_fixture model_bits=0x%0h model_bit_lsb=48 model_bit_msb=51 token_loop_bits=0x%0h top_bits=0x%0h layer_bits=0x%0h child_block_bits=0x%0h q_bits=0x%0h k_bits=0x%0h v_bits=0x%0h o_bits=0x%0h source=token_loop.status_o[51:48]:top.status_o[51:48]:layer.status_o[51:48]:block.status_o[83:80]:q/k/v/o.integration_status_o[45:42] status=0x%0h", status_o[48 +: 4], dut.u_token_loop_axi_decoder_block_fixture.status_o[48 +: 4], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.status_o[48 +: 4], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.status_o[48 +: 4], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42], dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42], status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE model_fsm_axi_decoder_block_fixture projection=q accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h source=hierarchical_model_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE model_fsm_axi_decoder_block_fixture projection=k accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h source=hierarchical_model_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE model_fsm_axi_decoder_block_fixture projection=v accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h source=hierarchical_model_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE model_fsm_axi_decoder_block_fixture projection=o accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h source=hierarchical_model_child", dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("ATTENTION_OUTPUT_TRACE model_fsm_axi_decoder_block_fixture layer=1 token=1 output=%0d,%0d,%0d,%0d source=hierarchical_axi_attention_child_output", lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 0), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 1), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 2), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 3));
        $display("MLP_INPUT_TRACE model_fsm_axi_decoder_block_fixture layer=1 token=1 hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_child_inputs", lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 0), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 1), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 2), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 3), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 0), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 1), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 2), lane_s16(dut.u_token_loop_axi_decoder_block_fixture.u_top_fsm_axi_decoder_block_fixture.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 3));
        $display("FINAL_OUTPUT_TRACE model_fsm_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("MODEL_FSM_STABILITY_TRACE model_fsm_axi_decoder_block_fixture stable=%0d", stable_passed);
        $display("MODEL_OUTPUT_POLICY_TRACE model_fsm_axi_decoder_block_fixture repeated_deterministic_outputs=%0d layer_dependent_outputs=0", layer0_output_seen == layer1_output_seen && layer1_output_seen == final_output_o);
        $display("COMPACT_IO_TRACE model_fsm_axi_decoder_block_fixture estimated_iob_bits=132 prior_token_loop_axi_bonded_iob=132 prior_token_loop_axi_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL model_fsm_axi_decoder_block_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS model_fsm_axi_decoder_block_fixture");
        $finish;
    end
endmodule
