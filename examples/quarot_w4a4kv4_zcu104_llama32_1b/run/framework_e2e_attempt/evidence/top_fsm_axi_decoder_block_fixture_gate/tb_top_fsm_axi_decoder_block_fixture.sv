`timescale 1ns/1ps

module tb_top_fsm_axi_decoder_block_fixture;
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

    logic layer_busy_seen_r;
    logic layer_done_seen_while_start_high_r;
    logic layer_start_deasserted_after_done_r;
    logic layer_release_seen_r;
    logic layer_done_prev_r;
    integer layer_busy_cycles;
    integer observed;

    top_fsm_axi_decoder_block_fixture dut (
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
                $display("FAIL top_fsm_axi_decoder_block_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    always @(posedge aclk) begin
        if (!aresetn) begin
            layer_busy_seen_r <= 1'b0;
            layer_done_seen_while_start_high_r <= 1'b0;
            layer_start_deasserted_after_done_r <= 1'b0;
            layer_release_seen_r <= 1'b0;
            layer_done_prev_r <= 1'b0;
            layer_busy_cycles <= 0;
        end else begin
            if (dut.layer_start_r && !dut.layer_done_w) begin
                layer_busy_seen_r <= 1'b1;
                layer_busy_cycles <= layer_busy_cycles + 1;
            end
            if (layer_busy_seen_r && !layer_done_seen_while_start_high_r && !dut.layer_done_w && !dut.layer_start_r) begin
                $display("FAIL top_fsm_axi_decoder_block_fixture layer start was not held while layer busy");
                $fatal;
            end
            if (dut.layer_start_r && dut.layer_done_w) begin
                layer_done_seen_while_start_high_r <= 1'b1;
            end
            if (layer_done_prev_r && !dut.layer_start_r) begin
                layer_start_deasserted_after_done_r <= 1'b1;
            end
            if (layer_done_prev_r && dut.layer_start_r) begin
                $display("FAIL top_fsm_axi_decoder_block_fixture layer start was not deasserted after layer done_o");
                $fatal;
            end
            if (layer_start_deasserted_after_done_r && !dut.layer_done_w) begin
                layer_release_seen_r <= 1'b1;
            end
            layer_done_prev_r <= dut.layer_done_w;
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
        #14000;
        if (!done_o) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture done_o was not asserted");
            $fatal;
        end
        if (status_o[15:0] != 16'h5453) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture top trace observed=0x%0h expected=0x5453", status_o[15:0]);
            $fatal;
        end
        if (status_o[16 +: 16] != 16'h4241) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture layer trace observed=0x%0h expected=0x4241", status_o[16 +: 16]);
            $fatal;
        end
        if (status_o[48 +: 4] != 4'hf || status_o[48 +: 4] != dut.layer_status_w[48 +: 4]) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture top AXI metadata bits top=0x%0h layer=0x%0h status=0x%0h", status_o[48 +: 4], dut.layer_status_w[48 +: 4], status_o);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0] != 16'h4241) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture hierarchical layer trace observed=0x%0h", dut.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0]);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0] != 32'hb2b1a2a1) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture hierarchical block trace observed=0x%0h", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0]);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4] != 4'hf) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture hierarchical block AXI metadata bits observed=0x%0h", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4]);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.status_o[47:0] != 48'h323122211211) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture nested AXI attention trace observed=0x%0h", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.status_o[47:0]);
            $fatal;
        end
        if (dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0] != 80'h52514241323122211211) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture nested MLP trace observed=0x%0h", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0]);
            $fatal;
        end
        if (!layer_done_seen_while_start_high_r || !layer_start_deasserted_after_done_r || !layer_release_seen_r || layer_busy_cycles == 0) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture layer start protocol busy_cycles=%0d done_seen=%0d deassert=%0d release=%0d", layer_busy_cycles, layer_done_seen_while_start_high_r, layer_start_deasserted_after_done_r, layer_release_seen_r);
            $fatal;
        end

        check_lane("layer_child_output", dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 0, 12);
        check_lane("layer_child_output", dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 1, -6);
        check_lane("layer_child_output", dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 2, 18);
        check_lane("layer_child_output", dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 3, 6);
        check_lane("decoder_block_output", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 0, 12);
        check_lane("decoder_block_output", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 1, -6);
        check_lane("decoder_block_output", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 2, 18);
        check_lane("decoder_block_output", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 3, 6);
        if (dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r != dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture MLP child did not consume captured attention output");
            $fatal;
        end
        check_lane("final_top_output", final_output_o, 0, 12);
        check_lane("final_top_output", final_output_o, 1, -6);
        check_lane("final_top_output", final_output_o, 2, 18);
        check_lane("final_top_output", final_output_o, 3, 6);

        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o != stable_output_snapshot || status_o != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL top_fsm_axi_decoder_block_fixture output/status changed while top done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("TOP_AXI_DECODER_BLOCK_TRACE top_fsm_axi_decoder_block_fixture top_trace_hex=0x%0h layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h events=layer_fsm_axi_decoder_block_fixture_start,layer_fsm_axi_decoder_block_fixture_done", status_o[15:0], status_o[16 +: 16], dut.u_layer_fsm_axi_decoder_block_fixture.status_o[16 +: 32], status_o[48 +: 4]);
        $display("LAYER_AXI_DECODER_BLOCK_TRACE top_fsm_axi_decoder_block_fixture layer_trace_hex=0x%0h block_trace_hex=0x%0h axi_metadata_bits=0x%0h events=decoder_block_axi_attention_mlp_fixture_start,decoder_block_axi_attention_mlp_fixture_done", dut.u_layer_fsm_axi_decoder_block_fixture.status_o[15:0], dut.u_layer_fsm_axi_decoder_block_fixture.status_o[16 +: 32], dut.u_layer_fsm_axi_decoder_block_fixture.status_o[48 +: 4]);
        $display("BLOCK_AXI_TRACE top_fsm_axi_decoder_block_fixture block_trace_hex=0x%0h attention_trace_hex=0x%0h axi_metadata_bits=0x%0h mlp_trace_hex=0x%0h events=axi_attention_start,axi_attention_done,mlp_start,mlp_done", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[31:0], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[32 +: 48], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[96 +: 80]);
        $display("AXI_ATTENTION_CHILD_TRACE top_fsm_axi_decoder_block_fixture trace_hex=0x%0h events=source_path_start,source_path_done,projection_axi_start,projection_axi_done,attention_kv_start,attention_kv_done", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.status_o[47:0]);
        $display("TOP_AXI_METADATA_PROPAGATION_TRACE top_fsm_axi_decoder_block_fixture top_bits=0x%0h top_bit_lsb=48 top_bit_msb=51 layer_bits=0x%0h child_block_bits=0x%0h q_bits=0x%0h k_bits=0x%0h v_bits=0x%0h o_bits=0x%0h source=layer_fsm_axi_decoder_block_fixture.status_o[51:48]:decoder_block_axi_attention_mlp_fixture.status_o[83:80]:q_projection_axi_stream_integration.integration_status_o[45:42]:k_projection_axi_stream_integration.integration_status_o[45:42]:v_projection_axi_stream_integration.integration_status_o[45:42]:o_projection_axi_stream_integration.integration_status_o[45:42]:aggregate_and status=0x%0h", status_o[48 +: 4], dut.u_layer_fsm_axi_decoder_block_fixture.status_o[48 +: 4], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o[80 +: 4], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[45:42], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[45:42], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[45:42], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[45:42], status_o);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_decoder_block_fixture projection=q addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=%0d ready_low_cycles=%0d instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.axi_arid_w, 2, 2);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_decoder_block_fixture projection=k addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=%0d ready_low_cycles=%0d instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.axi_arid_w, 2, 2);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_decoder_block_fixture projection=v addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=%0d ready_low_cycles=%0d instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.axi_arid_w, 2, 2);
        $display("AXI_PROJECTION_CHILD_AR_TRACE top_fsm_axi_decoder_block_fixture projection=o addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=%0d ready_low_cycles=%0d instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_araddr_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arlen_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arsize_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arburst_w, dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.axi_arid_w, 2, 2);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_decoder_block_fixture projection=q accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_decoder_block_fixture projection=k accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_decoder_block_fixture projection=v accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_R_METADATA_TRACE top_fsm_axi_decoder_block_fixture projection=o accepted=0x3 last=0x2 rid_ok=1 rresp_ok=1 rlast_ok=1 status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_decoder_block_fixture projection=q emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_decoder_block_fixture projection=k emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_decoder_block_fixture projection=v emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_PAYLOAD_TRACE top_fsm_axi_decoder_block_fixture projection=o emitted=%0d consumed=%0d payload_match=1 ready_low_trace=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[15:8], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o[23:16], dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.ready_low_trace_r);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture projection=q output=%0d,%0d status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture projection=k output=%0d,%0d status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture projection=v output=%0d,%0d status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture projection=o output=%0d,%0d status=0x%0h instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration", $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[0*32 +: 32]), $signed(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.output_o[1*32 +: 32]), dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration.integration_status_o);
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_decoder_block_fixture projection=q packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_decoder_block_fixture projection=k packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_k_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_decoder_block_fixture projection=v packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_v_projection_axi_stream_integration");
        $display("AXI_PROJECTION_CHILD_ROUND_TRIP_TRACE top_fsm_axi_decoder_block_fixture projection=o packed_bytes=32 unpacked_values=64 round_trip_passed=1 instance=u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_decoder_child_axi_attention_datapath.u_o_projection_axi_stream_integration");
        $display("MLP_CHILD_TRACE top_fsm_axi_decoder_block_fixture trace_hex=0x%0h events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done", dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.status_o[79:0]);
        $display("TOP_LAYER_CHILD_START_HOLD_TRACE top_fsm_axi_decoder_block_fixture busy_cycles=%0d done_seen_while_start_high=%0d deasserted_after_done=%0d release_seen=%0d", layer_busy_cycles, layer_done_seen_while_start_high_r, layer_start_deasserted_after_done_r, layer_release_seen_r);
        $display("ATTENTION_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d source=hierarchical_attention_child_output", lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.captured_attention_r, 3));
        $display("MLP_INPUT_TRACE top_fsm_axi_decoder_block_fixture hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d source=hierarchical_residual_mlp_child_inputs", lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.hidden_r, 3), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.attention_r, 3));
        $display("RESIDUAL0_TRACE top_fsm_axi_decoder_block_fixture residual0=%0d,%0d,%0d,%0d", lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.residual0_r, 3));
        $display("MLP_FINAL_TRACE top_fsm_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d", lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.u_residual_mlp_fixture.final_output_o, 3));
        $display("FINAL_DECODER_BLOCK_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.final_output_o, 3), dut.u_layer_fsm_axi_decoder_block_fixture.u_decoder_block_axi_attention_mlp_fixture.status_o);
        $display("LAYER_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 0), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 1), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 2), lane_s16(dut.u_layer_fsm_axi_decoder_block_fixture.final_output_o, 3), dut.u_layer_fsm_axi_decoder_block_fixture.status_o);
        $display("FINAL_OUTPUT_TRACE top_fsm_axi_decoder_block_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("TOP_STABILITY_TRACE top_fsm_axi_decoder_block_fixture stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE top_fsm_axi_decoder_block_fixture estimated_iob_bits=132 prior_layer_bonded_iob=132 prior_layer_status_bits=64 exposed_child_vectors=0 exposed_wide_status=0 exposed_128b_axi_data=0 exposed_axi_debug=0 exposed_kv_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL top_fsm_axi_decoder_block_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS top_fsm_axi_decoder_block_fixture");
        $finish;
    end
endmodule
