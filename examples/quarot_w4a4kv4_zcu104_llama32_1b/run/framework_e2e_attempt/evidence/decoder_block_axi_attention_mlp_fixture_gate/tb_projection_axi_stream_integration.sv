`timescale 1ns/1ps

module tb_projection_axi_stream_integration;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [63:0] output_o;
    logic [63:0] integration_status_o;
    logic [127:0] expected_rdata [0:1];
    logic [31:0] expected_payload [0:7];
    logic signed [7:0] expected_weight [0:63];
    logic [31:0] observed_emitted_payload [0:7];
    logic [31:0] observed_consumed_payload [0:7];
    logic [7:0] observed_ready_low_trace;
    logic signed [63:0] stable_output;
    logic [63:0] stable_status;
    logic stable_done;
    logic payload_seen_pending;
    logic payload_hold_ok;
    integer emitted_count;
    integer consumed_count;
    integer accepted_r_count;
    integer ar_ready_low_cycles;
    integer rvalid_while_projection_not_ready_cycles;
    integer idx;
    integer observed_output;

    projection_axi_stream_integration dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
        .output_o(output_o),
        .integration_status_o(integration_status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    function automatic logic signed [7:0] sign_extend_int4(input logic [3:0] nibble_i);
        begin
            sign_extend_int4 = { {4{nibble_i[3]}}, nibble_i };
        end
    endfunction

    always @(negedge aclk) begin
        if (!aresetn || !start_i) begin
            payload_seen_pending = 1'b0;
        end else begin
            if (dut.axi_arvalid_w && !dut.axi_arready_w) begin
                ar_ready_low_cycles = ar_ready_low_cycles + 1;
            end
            if (dut.axi_rvalid_w && !dut.axi_rready_w && dut.payload_link_valid_r) begin
                rvalid_while_projection_not_ready_cycles = rvalid_while_projection_not_ready_cycles + 1;
            end
            if (dut.payload_link_valid_r) begin
                if (!dut.payload_link_ready_w && consumed_count < 8) begin
                    observed_ready_low_trace[consumed_count] = 1'b1;
                    if (dut.payload_link_word_r != expected_payload[consumed_count]) begin
                        payload_hold_ok = 1'b0;
                    end
                end
                if (!payload_seen_pending) begin
                    if (emitted_count >= 8) begin
                        $display("FAIL projection_axi_stream_integration too many emitted payloads");
                        $fatal;
                    end
                    observed_emitted_payload[emitted_count] = dut.payload_link_word_r;
                    if (dut.payload_link_word_r != expected_payload[emitted_count]) begin
                        $display("FAIL projection_axi_stream_integration emitted[%0d]=0x%0h expected=0x%0h",
                                 emitted_count, dut.payload_link_word_r, expected_payload[emitted_count]);
                        $fatal;
                    end
                    emitted_count = emitted_count + 1;
                    payload_seen_pending = 1'b1;
                end
                if (dut.payload_link_ready_w) begin
                    if (consumed_count >= 8) begin
                        $display("FAIL projection_axi_stream_integration too many consumed payloads");
                        $fatal;
                    end
                    observed_consumed_payload[consumed_count] = dut.payload_link_word_r;
                    if (dut.payload_link_word_r != expected_payload[consumed_count]) begin
                        $display("FAIL projection_axi_stream_integration consumed[%0d]=0x%0h expected=0x%0h",
                                 consumed_count, dut.payload_link_word_r, expected_payload[consumed_count]);
                        $fatal;
                    end
                    consumed_count = consumed_count + 1;
                    payload_seen_pending = 1'b0;
                end
            end else begin
                payload_seen_pending = 1'b0;
            end
        end
    end

    always @(posedge aclk) begin
        if (aresetn && start_i && dut.r_fire_w) begin
            if (accepted_r_count >= 2) begin
                $display("FAIL projection_axi_stream_integration too many R beats");
                $fatal;
            end
            if (dut.axi_rdata_w != expected_rdata[accepted_r_count]) begin
                $display("FAIL projection_axi_stream_integration rdata[%0d]=0x%0h expected=0x%0h",
                         accepted_r_count, dut.axi_rdata_w, expected_rdata[accepted_r_count]);
                $fatal;
            end
            if (dut.axi_rid_w != 8'h02 || dut.axi_rresp_w != 2'b00 ||
                dut.axi_rlast_w != (accepted_r_count == (2 - 1))) begin
                $display("FAIL projection_axi_stream_integration R metadata beat=%0d rid=0x%0h resp=0x%0h last=%0b",
                         accepted_r_count, dut.axi_rid_w, dut.axi_rresp_w, dut.axi_rlast_w);
                $fatal;
            end
            accepted_r_count = accepted_r_count + 1;
        end
    end

    initial begin
        expected_rdata[0] = 128'h61c72d83e94fa50b61c72d83e94fa50b;
        expected_rdata[1] = 128'h61c72d83e94fa50b61c72d83e94fa50b;
        expected_payload[0] = 32'he94fa50b;
        expected_payload[1] = 32'h61c72d83;
        expected_payload[2] = 32'he94fa50b;
        expected_payload[3] = 32'h61c72d83;
        expected_payload[4] = 32'he94fa50b;
        expected_payload[5] = 32'h61c72d83;
        expected_payload[6] = 32'he94fa50b;
        expected_payload[7] = 32'h61c72d83;
        expected_weight[0] = -8'sd5;
        expected_weight[1] = 8'sd0;
        expected_weight[2] = 8'sd5;
        expected_weight[3] = -8'sd6;
        expected_weight[4] = -8'sd1;
        expected_weight[5] = 8'sd4;
        expected_weight[6] = -8'sd7;
        expected_weight[7] = -8'sd2;
        expected_weight[8] = 8'sd3;
        expected_weight[9] = -8'sd8;
        expected_weight[10] = -8'sd3;
        expected_weight[11] = 8'sd2;
        expected_weight[12] = 8'sd7;
        expected_weight[13] = -8'sd4;
        expected_weight[14] = 8'sd1;
        expected_weight[15] = 8'sd6;
        expected_weight[16] = -8'sd5;
        expected_weight[17] = 8'sd0;
        expected_weight[18] = 8'sd5;
        expected_weight[19] = -8'sd6;
        expected_weight[20] = -8'sd1;
        expected_weight[21] = 8'sd4;
        expected_weight[22] = -8'sd7;
        expected_weight[23] = -8'sd2;
        expected_weight[24] = 8'sd3;
        expected_weight[25] = -8'sd8;
        expected_weight[26] = -8'sd3;
        expected_weight[27] = 8'sd2;
        expected_weight[28] = 8'sd7;
        expected_weight[29] = -8'sd4;
        expected_weight[30] = 8'sd1;
        expected_weight[31] = 8'sd6;
        expected_weight[32] = -8'sd5;
        expected_weight[33] = 8'sd0;
        expected_weight[34] = 8'sd5;
        expected_weight[35] = -8'sd6;
        expected_weight[36] = -8'sd1;
        expected_weight[37] = 8'sd4;
        expected_weight[38] = -8'sd7;
        expected_weight[39] = -8'sd2;
        expected_weight[40] = 8'sd3;
        expected_weight[41] = -8'sd8;
        expected_weight[42] = -8'sd3;
        expected_weight[43] = 8'sd2;
        expected_weight[44] = 8'sd7;
        expected_weight[45] = -8'sd4;
        expected_weight[46] = 8'sd1;
        expected_weight[47] = 8'sd6;
        expected_weight[48] = -8'sd5;
        expected_weight[49] = 8'sd0;
        expected_weight[50] = 8'sd5;
        expected_weight[51] = -8'sd6;
        expected_weight[52] = -8'sd1;
        expected_weight[53] = 8'sd4;
        expected_weight[54] = -8'sd7;
        expected_weight[55] = -8'sd2;
        expected_weight[56] = 8'sd3;
        expected_weight[57] = -8'sd8;
        expected_weight[58] = -8'sd3;
        expected_weight[59] = 8'sd2;
        expected_weight[60] = 8'sd7;
        expected_weight[61] = -8'sd4;
        expected_weight[62] = 8'sd1;
        expected_weight[63] = 8'sd6;
        aresetn = 1'b0;
        start_i = 1'b0;
        emitted_count = 0;
        consumed_count = 0;
        accepted_r_count = 0;
        ar_ready_low_cycles = 0;
        rvalid_while_projection_not_ready_cycles = 0;
        observed_ready_low_trace = '0;
        payload_seen_pending = 1'b0;
        payload_hold_ok = 1'b1;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #20;
        if (!dut.axi_arvalid_w) begin
            $display("FAIL projection_axi_stream_integration AR valid missing");
            $fatal;
        end
        if (dut.axi_araddr_w != 32'h00120000 ||
            dut.axi_arlen_w != 8'h01 ||
            dut.axi_arsize_w != 3'h4 ||
            dut.axi_arburst_w != 2'b01 ||
            dut.axi_arid_w != 8'h02) begin
            $display("FAIL projection_axi_stream_integration AR fields addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h",
                     dut.axi_araddr_w, dut.axi_arlen_w, dut.axi_arsize_w, dut.axi_arburst_w, dut.axi_arid_w);
            $fatal;
        end
        #5000;
        if (!done_o) begin
            $display("FAIL projection_axi_stream_integration done_o did not assert");
            $fatal;
        end
        if (accepted_r_count != 2 || emitted_count != 8 || consumed_count != 8) begin
            $display("FAIL projection_axi_stream_integration counts r=%0d emitted=%0d consumed=%0d",
                     accepted_r_count, emitted_count, consumed_count);
            $fatal;
        end
        if (dut.accepted_beat_trace_r != 2'h3 ||
            dut.rlast_trace_r != 2'h2 ||
            dut.rid_error_trace_r != '0 || dut.rresp_error_trace_r != '0 || dut.rlast_error_trace_r != '0) begin
            $display("FAIL projection_axi_stream_integration metadata trace accepted=0x%0h last=0x%0h rid_err=0x%0h resp_err=0x%0h last_err=0x%0h status=0x%0h",
                     dut.accepted_beat_trace_r, dut.rlast_trace_r, dut.rid_error_trace_r,
                     dut.rresp_error_trace_r, dut.rlast_error_trace_r, integration_status_o);
            $fatal;
        end
        if (observed_ready_low_trace != 8'h19 || !payload_hold_ok ||
            dut.ready_low_trace_r != 8'h19) begin
            $display("FAIL projection_axi_stream_integration ready-low trace observed=0x%0h dut=0x%0h hold_ok=%0d expected=0x19",
                     observed_ready_low_trace, dut.ready_low_trace_r, payload_hold_ok);
            $fatal;
        end
        if (rvalid_while_projection_not_ready_cycles < 1) begin
            $display("FAIL projection_axi_stream_integration R valid did not observe projection-side stall");
            $fatal;
        end
        for (idx = 0; idx < 8; idx = idx + 1) begin
            if (observed_emitted_payload[idx] != observed_consumed_payload[idx]) begin
                $display("FAIL projection_axi_stream_integration payload mismatch[%0d] emitted=0x%0h consumed=0x%0h",
                         idx, observed_emitted_payload[idx], observed_consumed_payload[idx]);
                $fatal;
            end
        end
        for (idx = 0; idx < 64; idx = idx + 1) begin
            if (sign_extend_int4(dut.packed_weight_r[idx*4 +: 4]) != expected_weight[idx]) begin
                $display("FAIL projection_axi_stream_integration round-trip[%0d]=%0d expected=%0d",
                         idx, sign_extend_int4(dut.packed_weight_r[idx*4 +: 4]), expected_weight[idx]);
                $fatal;
            end
        end
        observed_output = $signed(output_o[0*32 +: 32]);
        if (observed_output != 484) begin
            $display("FAIL projection_axi_stream_integration output[%0d]=%0d expected=484", 0, observed_output);
            $fatal;
        end
        observed_output = $signed(output_o[1*32 +: 32]);
        if (observed_output != 1904) begin
            $display("FAIL projection_axi_stream_integration output[%0d]=%0d expected=1904", 1, observed_output);
            $fatal;
        end
        stable_output = output_o;
        stable_status = integration_status_o;
        stable_done = done_o;
        #30;
        if (!done_o || done_o != stable_done || output_o != stable_output || integration_status_o != stable_status) begin
            $display("FAIL projection_axi_stream_integration outputs changed while done_o was high");
            $fatal;
        end

        $display("AXI_STREAM_AR_TRACE projection_axi_stream_integration addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2 ready_low_cycles=%0d",
                 dut.axi_araddr_w, dut.axi_arlen_w, dut.axi_arsize_w, dut.axi_arburst_w, dut.axi_arid_w, ar_ready_low_cycles);
        $display("AXI_STREAM_R_METADATA_TRACE projection_axi_stream_integration accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d status=0x%0h words=0x61c72d83e94fa50b61c72d83e94fa50b 0x61c72d83e94fa50b61c72d83e94fa50b",
                 dut.accepted_beat_trace_r, dut.rlast_trace_r, (dut.rid_error_trace_r == '0),
                 (dut.rresp_error_trace_r == '0), (dut.rlast_error_trace_r == '0), integration_status_o);
        $write("OBSERVED_EMITTED_PAYLOADS projection_axi_stream_integration");
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $write(" 0x%08h", observed_emitted_payload[idx]);
        end
        $write("\n");
        $write("OBSERVED_CONSUMED_PAYLOADS projection_axi_stream_integration");
        for (idx = 0; idx < 8; idx = idx + 1) begin
            $write(" 0x%08h", observed_consumed_payload[idx]);
        end
        $write("\n");
        $display("EXPECTED_PAYLOAD_TRACE projection_axi_stream_integration payload_words=0xe94fa50b 0x61c72d83 0xe94fa50b 0x61c72d83 0xe94fa50b 0x61c72d83 0xe94fa50b 0x61c72d83");
        $display("BACKPRESSURE_TRACE projection_axi_stream_integration ready_low_payload_idx=0,3,4 trace=0x%0h inside_beat_idx=0 boundary_idx=3 post_boundary_idx=4 payload_hold_ok=%0d rvalid_while_projection_not_ready_cycles=%0d",
                 observed_ready_low_trace, payload_hold_ok, rvalid_while_projection_not_ready_cycles);
        $display("PROJECTION_OUTPUT_TRACE projection_axi_stream_integration output=484,1904 golden=484,1904");
        $display("ROUND_TRIP_TRACE projection_axi_stream_integration packed_bytes=32 unpacked_values=64 round_trip_passed=1");
        $display("VALIDATION_TRACE projection_axi_stream_integration payload_match=1 metadata_ok=1 done_stability_ok=1 emitted_count=8 consumed_count=8");
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL projection_axi_stream_integration done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS projection_axi_stream_integration");
        $finish;
    end
endmodule
