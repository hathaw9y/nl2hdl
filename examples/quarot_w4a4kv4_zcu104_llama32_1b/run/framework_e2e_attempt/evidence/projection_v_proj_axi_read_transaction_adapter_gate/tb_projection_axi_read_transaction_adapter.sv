`timescale 1ns/1ps

module tb_projection_axi_read_transaction_adapter;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic axi_arvalid_o;
    logic axi_arready_i;
    logic [31:0] axi_araddr_o;
    logic [7:0] axi_arlen_o;
    logic [2:0] axi_arsize_o;
    logic [1:0] axi_arburst_o;
    logic [7:0] axi_arid_o;
    logic axi_rvalid_i;
    logic axi_rready_o;
    logic [127:0] axi_rdata_i;
    logic [7:0] axi_rid_i;
    logic [1:0] axi_rresp_i;
    logic axi_rlast_i;
    logic payload_valid_o;
    logic payload_ready_i;
    logic [31:0] payload_word_o;
    logic payload_last_o;
    logic [1:0] accepted_beat_trace_o;
    logic [1:0] rlast_trace_o;
    logic [1:0] rid_error_trace_o;
    logic [1:0] rresp_error_trace_o;
    logic [1:0] rlast_error_trace_o;
    logic [15:0] status_o;
    logic [127:0] expected_rdata [0:1];
    logic [31:0] expected_payload [0:7];
    logic [31:0] stable_araddr;
    logic [7:0] stable_arlen;
    logic [2:0] stable_arsize;
    logic [1:0] stable_arburst;
    logic [7:0] stable_arid;
    logic [31:0] stable_payload;
    logic stable_payload_last;
    logic stable_done;
    logic [15:0] stable_status;
    integer idx;
    integer ar_ready_low_cycles;
    integer rvalid_while_rready_low_cycles;
    integer payload_ready_low_events;
    integer accepted_r_beats;
    integer accepted_last_trace;
    integer rid_error_count;
    integer rresp_error_count;
    integer rlast_error_count;
    integer payload_count;

    projection_axi_read_transaction_adapter dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
        .axi_arvalid_o(axi_arvalid_o),
        .axi_arready_i(axi_arready_i),
        .axi_araddr_o(axi_araddr_o),
        .axi_arlen_o(axi_arlen_o),
        .axi_arsize_o(axi_arsize_o),
        .axi_arburst_o(axi_arburst_o),
        .axi_arid_o(axi_arid_o),
        .axi_rvalid_i(axi_rvalid_i),
        .axi_rready_o(axi_rready_o),
        .axi_rdata_i(axi_rdata_i),
        .axi_rid_i(axi_rid_i),
        .axi_rresp_i(axi_rresp_i),
        .axi_rlast_i(axi_rlast_i),
        .payload_valid_o(payload_valid_o),
        .payload_ready_i(payload_ready_i),
        .payload_word_o(payload_word_o),
        .payload_last_o(payload_last_o),
        .accepted_beat_trace_o(accepted_beat_trace_o),
        .rlast_trace_o(rlast_trace_o),
        .rid_error_trace_o(rid_error_trace_o),
        .rresp_error_trace_o(rresp_error_trace_o),
        .rlast_error_trace_o(rlast_error_trace_o),
        .status_o(status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    task automatic check_ar_fields_stable;
        begin
            if (axi_araddr_o !== stable_araddr || axi_arlen_o !== stable_arlen ||
                axi_arsize_o !== stable_arsize || axi_arburst_o !== stable_arburst ||
                axi_arid_o !== stable_arid) begin
                $display("FAIL projection_axi_read_transaction_adapter AR field changed during ready-low stall");
                $fatal;
            end
        end
    endtask

    task automatic check_payload_hold;
        input [31:0] expected_word;
        input expected_last;
        begin
            stable_payload = payload_word_o;
            stable_payload_last = payload_last_o;
            #10;
            if (!payload_valid_o || payload_word_o != expected_word || payload_word_o != stable_payload ||
                payload_last_o != expected_last || payload_last_o != stable_payload_last || payload_ready_i) begin
                $display("FAIL projection_axi_read_transaction_adapter payload did not hold under ready-low backpressure word=0x%0h expected=0x%0h",
                         payload_word_o, expected_word);
                $fatal;
            end
            payload_ready_low_events = payload_ready_low_events + 1;
        end
    endtask

    task automatic accept_r_beat;
        input integer beat_idx;
        begin
            if (!axi_rready_o) begin
                $display("FAIL projection_axi_read_transaction_adapter axi_rready_o low before beat %0d", beat_idx);
                $fatal;
            end
            axi_rdata_i = expected_rdata[beat_idx];
            axi_rid_i = 8'h02;
            axi_rresp_i = 2'b00;
            axi_rlast_i = (beat_idx == (2 - 1));
            axi_rvalid_i = 1'b1;
            #10;
            if (axi_rid_i != 8'h02) begin
                rid_error_count = rid_error_count + 1;
            end
            if (axi_rresp_i != 2'b00) begin
                rresp_error_count = rresp_error_count + 1;
            end
            if (axi_rlast_i != (beat_idx == (2 - 1))) begin
                rlast_error_count = rlast_error_count + 1;
            end
            accepted_r_beats = accepted_r_beats | (1 << beat_idx);
            if (axi_rlast_i) begin
                accepted_last_trace = accepted_last_trace | (1 << beat_idx);
            end
        end
    endtask

    task automatic drain_payloads_for_current_beat;
        integer drain_idx;
        begin
            payload_ready_i = 1'b1;
            for (drain_idx = 0; drain_idx < 4; drain_idx = drain_idx + 1) begin
                #10;
            end
            payload_ready_i = 1'b0;
        end
    endtask

    task automatic run_bad_metadata_regression;
        begin
            start_i = 1'b1;
            axi_arready_i = 1'b0;
            axi_rvalid_i = 1'b0;
            axi_rdata_i = '0;
            axi_rid_i = 8'h00;
            axi_rresp_i = 2'b00;
            axi_rlast_i = 1'b0;
            payload_ready_i = 1'b0;
            #10;
            if (!axi_arvalid_o) begin
                $display("FAIL projection_axi_read_transaction_adapter negative AR valid missing");
                $fatal;
            end
            axi_arready_i = 1'b1;
            #10;
            axi_arready_i = 1'b0;
            if (!axi_rready_o) begin
                $display("FAIL projection_axi_read_transaction_adapter negative R ready missing after AR");
                $fatal;
            end

            axi_rdata_i = expected_rdata[0];
            axi_rid_i = 8'h03;
            axi_rresp_i = 2'b10;
            axi_rlast_i = 1'b1;
            axi_rvalid_i = 1'b1;
            #10;
            axi_rvalid_i = 1'b0;
            drain_payloads_for_current_beat();
            if (!axi_rready_o) begin
                $display("FAIL projection_axi_read_transaction_adapter negative R ready missing before final beat");
                $fatal;
            end

            axi_rdata_i = expected_rdata[1];
            axi_rid_i = 8'h02;
            axi_rresp_i = 2'b00;
            axi_rlast_i = 1'b0;
            axi_rvalid_i = 1'b1;
            #10;
            axi_rvalid_i = 1'b0;
            drain_payloads_for_current_beat();
            if (!done_o) begin
                $display("FAIL projection_axi_read_transaction_adapter negative did not reach done");
                $fatal;
            end
            if (accepted_beat_trace_o != 3 ||
                rlast_trace_o != 1 ||
                rid_error_trace_o != 1 ||
                rresp_error_trace_o != 1 ||
                rlast_error_trace_o != 3 ||
                ((axi_arlen_o + 1) != 2)) begin
                $display("FAIL projection_axi_read_transaction_adapter negative metadata errors not recorded accepted=0x%0h last=0x%0h rid_err=0x%0h resp_err=0x%0h last_err=0x%0h status=0x%0h",
                         accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o, rlast_error_trace_o, status_o);
                $fatal;
            end
            $display("NEGATIVE_METADATA_TRACE projection_axi_read_transaction_adapter accepted=0x%0h last=0x%0h rid_error=0x%0h rresp_error=0x%0h rlast_error=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d arlen_matches_r_beats=%0d status=0x%0h",
                     accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o, rlast_error_trace_o,
                     (rid_error_trace_o == '0), (rresp_error_trace_o == '0), (rlast_error_trace_o == '0),
                     ((axi_arlen_o + 1) == 2), status_o);
            start_i = 1'b0;
            payload_ready_i = 1'b0;
            #20;
            if (done_o) begin
                $display("FAIL projection_axi_read_transaction_adapter negative done_o did not clear");
                $fatal;
            end
        end
    endtask

    initial begin
        expected_rdata[0] = 128'h00112233445566778899aabbccddeeff;
        expected_rdata[1] = 128'h102132435465768798a9babbdcddfeff;
        expected_payload[0] = 32'hccddeeff;
        expected_payload[1] = 32'h8899aabb;
        expected_payload[2] = 32'h44556677;
        expected_payload[3] = 32'h00112233;
        expected_payload[4] = 32'hdcddfeff;
        expected_payload[5] = 32'h98a9babb;
        expected_payload[6] = 32'h54657687;
        expected_payload[7] = 32'h10213243;
        aresetn = 1'b0;
        start_i = 1'b0;
        axi_arready_i = 1'b0;
        axi_rvalid_i = 1'b0;
        axi_rdata_i = '0;
        axi_rid_i = 8'h00;
        axi_rresp_i = 2'b00;
        axi_rlast_i = 1'b0;
        payload_ready_i = 1'b0;
        ar_ready_low_cycles = 0;
        rvalid_while_rready_low_cycles = 0;
        payload_ready_low_events = 0;
        accepted_r_beats = 0;
        accepted_last_trace = 0;
        rid_error_count = 0;
        rresp_error_count = 0;
        rlast_error_count = 0;
        payload_count = 0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #10;
        if (!axi_arvalid_o) begin
            $display("FAIL projection_axi_read_transaction_adapter axi_arvalid_o not asserted");
            $fatal;
        end
        if (axi_araddr_o != 32'h00120000 ||
            axi_arlen_o != 8'h01 ||
            axi_arsize_o != 3'h4 ||
            axi_arburst_o != 2'b01 ||
            axi_arid_o != 8'h02) begin
            $display("FAIL projection_axi_read_transaction_adapter unexpected AR command addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h",
                     axi_araddr_o, axi_arlen_o, axi_arsize_o, axi_arburst_o, axi_arid_o);
            $fatal;
        end
        stable_araddr = axi_araddr_o;
        stable_arlen = axi_arlen_o;
        stable_arsize = axi_arsize_o;
        stable_arburst = axi_arburst_o;
        stable_arid = axi_arid_o;
        repeat (2) begin
            #10;
            if (!axi_arvalid_o || axi_arready_i) begin
                $display("FAIL projection_axi_read_transaction_adapter AR ready-low stall was not maintained");
                $fatal;
            end
            check_ar_fields_stable();
            ar_ready_low_cycles = ar_ready_low_cycles + 1;
        end
        axi_arready_i = 1'b1;
        #10;
        if (axi_arvalid_o || !axi_rready_o) begin
            $display("FAIL projection_axi_read_transaction_adapter did not transition from AR to R channel arvalid=%0b rready=%0b",
                     axi_arvalid_o, axi_rready_o);
            $fatal;
        end
        axi_arready_i = 1'b0;

        accept_r_beat(0);
        axi_rdata_i = expected_rdata[1];
        axi_rlast_i = 1'b1;
        if (axi_rready_o) begin
            $display("FAIL projection_axi_read_transaction_adapter did not backpressure early second R beat");
            $fatal;
        end
        rvalid_while_rready_low_cycles = rvalid_while_rready_low_cycles + 1;
        if (!payload_valid_o || payload_word_o != expected_payload[0] || payload_last_o) begin
            $display("FAIL projection_axi_read_transaction_adapter first payload word=0x%0h last=%0b", payload_word_o, payload_last_o);
            $fatal;
        end
        check_payload_hold(expected_payload[0], 1'b0);

        payload_ready_i = 1'b1;
        for (idx = 1; idx < 3; idx = idx + 1) begin
            #10;
            if (!payload_valid_o || payload_word_o != expected_payload[idx] || payload_last_o) begin
                $display("FAIL projection_axi_read_transaction_adapter payload[%0d] observed=0x%0h expected=0x%0h last=%0b",
                         idx, payload_word_o, expected_payload[idx], payload_last_o);
                $fatal;
            end
            payload_count = payload_count + 1;
            if (!axi_rready_o) begin
                rvalid_while_rready_low_cycles = rvalid_while_rready_low_cycles + 1;
            end
        end
        #10;
        if (!payload_valid_o || payload_word_o != expected_payload[3] || payload_last_o) begin
            $display("FAIL projection_axi_read_transaction_adapter boundary payload observed=0x%0h expected=0x%0h last=%0b",
                     payload_word_o, expected_payload[3], payload_last_o);
            $fatal;
        end
        payload_count = payload_count + 1;
        if (!axi_rready_o) begin
            rvalid_while_rready_low_cycles = rvalid_while_rready_low_cycles + 1;
        end
        payload_ready_i = 1'b0;
        check_payload_hold(expected_payload[3], 1'b0);

        payload_ready_i = 1'b1;
        #10;
        if (!axi_rready_o || payload_valid_o) begin
            $display("FAIL projection_axi_read_transaction_adapter did not return to R channel for second beat ready=%0b valid=%0b",
                     axi_rready_o, payload_valid_o);
            $fatal;
        end
        accept_r_beat(1);
        axi_rvalid_i = 1'b0;
        axi_rdata_i = '0;
        axi_rlast_i = 1'b0;
        payload_ready_i = 1'b0;
        #10;
        if (!payload_valid_o || payload_word_o != expected_payload[4] || payload_last_o) begin
            $display("FAIL projection_axi_read_transaction_adapter post-boundary payload observed=0x%0h expected=0x%0h last=%0b",
                     payload_word_o, expected_payload[4], payload_last_o);
            $fatal;
        end
        check_payload_hold(expected_payload[4], 1'b0);

        payload_ready_i = 1'b1;
        for (idx = 5; idx < 8; idx = idx + 1) begin
            #10;
            if (!payload_valid_o || payload_word_o != expected_payload[idx]) begin
                $display("FAIL projection_axi_read_transaction_adapter payload[%0d] observed=0x%0h expected=0x%0h",
                         idx, payload_word_o, expected_payload[idx]);
                $fatal;
            end
            if (payload_last_o != (idx == (8 - 1))) begin
                $display("FAIL projection_axi_read_transaction_adapter payload_last_o[%0d]=%0b", idx, payload_last_o);
                $fatal;
            end
            payload_count = payload_count + 1;
        end
        #10;
        if (!done_o || payload_valid_o) begin
            $display("FAIL projection_axi_read_transaction_adapter did not assert done cleanly done=%0b payload_valid=%0b",
                     done_o, payload_valid_o);
            $fatal;
        end
        if (payload_word_o != expected_payload[7] || !payload_last_o) begin
            $display("FAIL projection_axi_read_transaction_adapter final payload not stable at done word=0x%0h last=%0b",
                     payload_word_o, payload_last_o);
            $fatal;
        end
        if (accepted_r_beats != 3 ||
            accepted_last_trace != 2 ||
            rid_error_count != 0 || rresp_error_count != 0 || rlast_error_count != 0 ||
            accepted_beat_trace_o != 3 ||
            rlast_trace_o != 2 ||
            rid_error_trace_o != '0 ||
            rresp_error_trace_o != '0 ||
            rlast_error_trace_o != '0 ||
            ((stable_arlen + 1) != 2)) begin
            $display("FAIL projection_axi_read_transaction_adapter transaction mismatch accepted=0x%0h last=0x%0h dut_accepted=0x%0h dut_last=0x%0h dut_rid_err=0x%0h dut_resp_err=0x%0h dut_last_err=0x%0h arlen=%0d",
                     accepted_r_beats, accepted_last_trace, accepted_beat_trace_o, rlast_trace_o,
                     rid_error_trace_o, rresp_error_trace_o, rlast_error_trace_o, stable_arlen);
            $fatal;
        end

        stable_payload = payload_word_o;
        stable_payload_last = payload_last_o;
        stable_done = done_o;
        stable_status = status_o;
        #20;
        if (!done_o || done_o != stable_done || payload_word_o != stable_payload ||
            payload_last_o != stable_payload_last || status_o != stable_status) begin
            $display("FAIL projection_axi_read_transaction_adapter outputs changed while done_o was high");
            $fatal;
        end

        $display("AXI_TRANSACTION_AR_TRACE projection_axi_read_transaction_adapter addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=2",
                 stable_araddr, stable_arlen, stable_arsize, stable_arburst, stable_arid);
        $display("AXI_TRANSACTION_R_TRACE projection_axi_read_transaction_adapter accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d arlen_matches_r_beats=%0d words=0x00112233445566778899aabbccddeeff 0x102132435465768798a9babbdcddfeff",
                 accepted_beat_trace_o, rlast_trace_o, (rid_error_trace_o == '0), (rresp_error_trace_o == '0),
                 (rlast_error_trace_o == '0), ((stable_arlen + 1) == 2));
        $display("PAYLOAD_TRACE projection_axi_read_transaction_adapter payload_words=0xccddeeff 0x8899aabb 0x44556677 0x00112233 0xdcddfeff 0x98a9babb 0x54657687 0x10213243 count=8 order_ok=1 stability_at_done=1");
        $display("BACKPRESSURE_TRACE projection_axi_read_transaction_adapter ar_ready_low_cycles=%0d ar_fields_stable=1 rvalid_while_rready_low_cycles=%0d payload_ready_low_indices=0,3,4 payload_ready_low_events=%0d",
                 ar_ready_low_cycles, rvalid_while_rready_low_cycles, payload_ready_low_events);
        $display("DUT_METADATA_TRACE projection_axi_read_transaction_adapter accepted=0x%0h last=0x%0h rid_error=0x%0h rresp_error=0x%0h rlast_error=0x%0h status=0x%0h",
                 accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o, rlast_error_trace_o, status_o);
        $display("VALIDATION_TRACE projection_axi_read_transaction_adapter id_ok=%0d resp_ok=%0d last_ok=%0d arlen_matches_r_beats=1 payload_order_ok=1 done_stability_ok=1",
                 (rid_error_trace_o == '0), (rresp_error_trace_o == '0), (rlast_error_trace_o == '0));
        start_i = 1'b0;
        payload_ready_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL projection_axi_read_transaction_adapter done_o did not clear after start_i deasserted");
            $fatal;
        end
        run_bad_metadata_regression();
        $display("PASS projection_axi_read_transaction_adapter");
        $finish;
    end
endmodule
