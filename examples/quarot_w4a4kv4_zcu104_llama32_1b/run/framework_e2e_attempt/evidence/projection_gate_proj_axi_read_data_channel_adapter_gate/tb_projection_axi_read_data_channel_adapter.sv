`timescale 1ns/1ps

module tb_projection_axi_read_data_channel_adapter;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
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
    logic [7:0] payload_trace_o;
    logic [7:0] payload_ready_low_trace_o;
    logic [15:0] status_o;
    logic [127:0] expected_rdata [0:1];
    logic [31:0] expected_payload [0:7];
    logic [31:0] stable_payload;
    logic stable_payload_last;
    logic [1:0] stable_accepted_trace;
    logic [1:0] stable_last_trace;
    logic [7:0] stable_payload_trace;
    logic [15:0] stable_status;
    integer idx;
    integer rready_low_cycles;
    integer payload_ready_low_events;
    integer negative_pass_count;

    projection_axi_read_data_channel_adapter dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
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
        .payload_trace_o(payload_trace_o),
        .payload_ready_low_trace_o(payload_ready_low_trace_o),
        .status_o(status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    task automatic check_payload_hold;
        input [31:0] expected_word;
        input expected_last;
        begin
            stable_payload = payload_word_o;
            stable_payload_last = payload_last_o;
            #10;
            if (!payload_valid_o || payload_word_o != expected_word || payload_word_o != stable_payload ||
                payload_last_o != expected_last || payload_last_o != stable_payload_last || payload_ready_i) begin
                $display("FAIL projection_axi_read_data_channel_adapter payload did not hold under ready-low backpressure word=0x%0h expected=0x%0h",
                         payload_word_o, expected_word);
                $fatal;
            end
            payload_ready_low_events = payload_ready_low_events + 1;
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

    task automatic start_negative_case;
        begin
            start_i = 1'b1;
            axi_rvalid_i = 1'b0;
            axi_rdata_i = '0;
            axi_rid_i = 8'h00;
            axi_rresp_i = 2'b00;
            axi_rlast_i = 1'b0;
            payload_ready_i = 1'b0;
            #10;
            if (!axi_rready_o) begin
                $display("FAIL projection_axi_read_data_channel_adapter negative R ready missing");
                $fatal;
            end
        end
    endtask

    task automatic drive_negative_beat;
        input integer beat_idx;
        input [7:0] rid_value;
        input [1:0] rresp_value;
        input rlast_value;
        begin
            if (!axi_rready_o) begin
                $display("FAIL projection_axi_read_data_channel_adapter negative R ready missing before beat %0d", beat_idx);
                $fatal;
            end
            axi_rdata_i = expected_rdata[beat_idx];
            axi_rid_i = rid_value;
            axi_rresp_i = rresp_value;
            axi_rlast_i = rlast_value;
            axi_rvalid_i = 1'b1;
            #10;
            axi_rvalid_i = 1'b0;
            drain_payloads_for_current_beat();
        end
    endtask

    task automatic finish_negative_case;
        input [8*32-1:0] case_name;
        input [1:0] expected_last_trace;
        input [1:0] expected_rid_error_trace;
        input [1:0] expected_rresp_error_trace;
        input [1:0] expected_rlast_error_trace;
        begin
            if (!done_o) begin
                $display("FAIL projection_axi_read_data_channel_adapter negative case %0s did not reach done", case_name);
                $fatal;
            end
            if (accepted_beat_trace_o != 2'h3 ||
                rlast_trace_o != expected_last_trace ||
                rid_error_trace_o != expected_rid_error_trace ||
                rresp_error_trace_o != expected_rresp_error_trace ||
                rlast_error_trace_o != expected_rlast_error_trace) begin
                $display("FAIL projection_axi_read_data_channel_adapter negative case %0s errors not recorded accepted=0x%0h last=0x%0h rid_err=0x%0h resp_err=0x%0h last_err=0x%0h status=0x%0h",
                         case_name, accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o,
                         rlast_error_trace_o, status_o);
                $fatal;
            end
            $display("NEGATIVE_METADATA_CASE projection_axi_read_data_channel_adapter case=%0s accepted=0x%0h last=0x%0h rid_error=0x%0h rresp_error=0x%0h rlast_error=0x%0h status=0x%0h",
                     case_name, accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o,
                     rlast_error_trace_o, status_o);
            negative_pass_count = negative_pass_count + 1;
            start_i = 1'b0;
            payload_ready_i = 1'b0;
            #20;
            if (done_o) begin
                $display("FAIL projection_axi_read_data_channel_adapter negative case %0s done_o did not clear", case_name);
                $fatal;
            end
        end
    endtask

    task automatic run_bad_metadata_regression;
        begin
            start_negative_case();
            drive_negative_beat(0, 8'h03, 2'b00, 1'b0);
            drive_negative_beat(1, 8'h02, 2'b00, 1'b1);
            finish_negative_case("bad_rid", 2'h2, 2'h1, '0, '0);

            start_negative_case();
            drive_negative_beat(0, 8'h02, 2'b10, 1'b0);
            drive_negative_beat(1, 8'h02, 2'b00, 1'b1);
            finish_negative_case("bad_rresp", 2'h2, '0, 2'h1, '0);

            start_negative_case();
            drive_negative_beat(0, 8'h02, 2'b00, 1'b1);
            drive_negative_beat(1, 8'h02, 2'b00, 1'b1);
            finish_negative_case("early_rlast", 2'h3, '0, '0, 2'h1);

            start_negative_case();
            drive_negative_beat(0, 8'h02, 2'b00, 1'b0);
            drive_negative_beat(1, 8'h02, 2'b00, 1'b0);
            finish_negative_case("missing_final_rlast", '0, '0, '0, 2'h2);

            if (negative_pass_count != 4) begin
                $display("FAIL projection_axi_read_data_channel_adapter negative metadata case count=%0d", negative_pass_count);
                $fatal;
            end
            $display("NEGATIVE_METADATA_SUMMARY projection_axi_read_data_channel_adapter bad_rid=1 bad_rresp=1 early_rlast=1 missing_final_rlast=1 dut_recorded_expected_errors=1 cases=%0d",
                     negative_pass_count);
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
        axi_rvalid_i = 1'b0;
        axi_rdata_i = '0;
        axi_rid_i = 8'h00;
        axi_rresp_i = 2'b00;
        axi_rlast_i = 1'b0;
        payload_ready_i = 1'b0;
        rready_low_cycles = 0;
        payload_ready_low_events = 0;
        negative_pass_count = 0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #10;
        if (!axi_rready_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter axi_rready_o not asserted after start");
            $fatal;
        end

        axi_rdata_i = expected_rdata[0];
        axi_rid_i = 8'h02;
        axi_rresp_i = 2'b00;
        axi_rlast_i = 1'b0;
        axi_rvalid_i = 1'b1;
        #10;
        axi_rdata_i = expected_rdata[1];
        axi_rlast_i = 1'b1;
        if (axi_rready_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter did not backpressure early second R beat");
            $fatal;
        end
        if (!payload_valid_o || payload_word_o != expected_payload[0] || payload_last_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter first payload word=0x%0h last=%0b", payload_word_o, payload_last_o);
            $fatal;
        end
        rready_low_cycles = rready_low_cycles + 1;
        check_payload_hold(expected_payload[0], 1'b0);

        payload_ready_i = 1'b1;
        for (idx = 1; idx < 3; idx = idx + 1) begin
            #10;
            if (!payload_valid_o || payload_word_o != expected_payload[idx] || payload_last_o) begin
                $display("FAIL projection_axi_read_data_channel_adapter payload[%0d] observed=0x%0h expected=0x%0h last=%0b",
                         idx, payload_word_o, expected_payload[idx], payload_last_o);
                $fatal;
            end
            if (!axi_rready_o) begin
                rready_low_cycles = rready_low_cycles + 1;
            end
        end
        #10;
        if (!payload_valid_o || payload_word_o != expected_payload[3] || payload_last_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter boundary payload observed=0x%0h expected=0x%0h last=%0b",
                     payload_word_o, expected_payload[3], payload_last_o);
            $fatal;
        end
        if (!axi_rready_o) begin
            rready_low_cycles = rready_low_cycles + 1;
        end
        payload_ready_i = 1'b0;
        check_payload_hold(expected_payload[3], 1'b0);

        payload_ready_i = 1'b1;
        #10;
        if (!axi_rready_o || payload_valid_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter did not return to R channel for second beat ready=%0b valid=%0b",
                     axi_rready_o, payload_valid_o);
            $fatal;
        end
        #10;
        axi_rvalid_i = 1'b0;
        axi_rdata_i = '0;
        axi_rlast_i = 1'b0;
        payload_ready_i = 1'b0;
        #10;
        if (!payload_valid_o || payload_word_o != expected_payload[4] || payload_last_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter post-boundary payload observed=0x%0h expected=0x%0h last=%0b",
                     payload_word_o, expected_payload[4], payload_last_o);
            $fatal;
        end
        check_payload_hold(expected_payload[4], 1'b0);

        payload_ready_i = 1'b1;
        for (idx = 5; idx < 8; idx = idx + 1) begin
            #10;
            if (!payload_valid_o || payload_word_o != expected_payload[idx]) begin
                $display("FAIL projection_axi_read_data_channel_adapter payload[%0d] observed=0x%0h expected=0x%0h",
                         idx, payload_word_o, expected_payload[idx]);
                $fatal;
            end
            if (payload_last_o != (idx == (8 - 1))) begin
                $display("FAIL projection_axi_read_data_channel_adapter payload_last_o[%0d]=%0b", idx, payload_last_o);
                $fatal;
            end
        end
        #10;
        if (!done_o || payload_valid_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter did not assert done cleanly done=%0b payload_valid=%0b",
                     done_o, payload_valid_o);
            $fatal;
        end
        if (payload_word_o != expected_payload[7] || !payload_last_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter final payload not stable at done word=0x%0h last=%0b",
                     payload_word_o, payload_last_o);
            $fatal;
        end
        if (accepted_beat_trace_o != 2'h3 ||
            rlast_trace_o != 2'h2 ||
            rid_error_trace_o != '0 || rresp_error_trace_o != '0 || rlast_error_trace_o != '0 ||
            payload_trace_o != 8'hff) begin
            $display("FAIL projection_axi_read_data_channel_adapter trace/error mismatch accepted=0x%0h last=0x%0h rid_err=0x%0h resp_err=0x%0h last_err=0x%0h payload=0x%0h",
                     accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o, rlast_error_trace_o, payload_trace_o);
            $fatal;
        end

        stable_payload = payload_word_o;
        stable_payload_last = payload_last_o;
        stable_accepted_trace = accepted_beat_trace_o;
        stable_last_trace = rlast_trace_o;
        stable_payload_trace = payload_trace_o;
        stable_status = status_o;
        #20;
        if (!done_o || payload_word_o != stable_payload || payload_last_o != stable_payload_last ||
            accepted_beat_trace_o != stable_accepted_trace || rlast_trace_o != stable_last_trace ||
            payload_trace_o != stable_payload_trace || status_o != stable_status) begin
            $display("FAIL projection_axi_read_data_channel_adapter outputs changed while done_o was high");
            $fatal;
        end

        $display("AXI_R_ACCEPT_TRACE projection_axi_read_data_channel_adapter accepted=0x%0h last=0x%0h rid_ok=%0d rresp_ok=%0d rlast_ok=%0d order_ok=1 beats=2 words=0x00112233445566778899aabbccddeeff 0x102132435465768798a9babbdcddfeff",
                 accepted_beat_trace_o, rlast_trace_o, (rid_error_trace_o == '0), (rresp_error_trace_o == '0), (rlast_error_trace_o == '0));
        $display("AXI_R_BACKPRESSURE_TRACE projection_axi_read_data_channel_adapter rvalid_while_rready_low_cycles=%0d payload_ready_low_indices=0,3,4 payload_ready_low_events=%0d",
                 rready_low_cycles, payload_ready_low_events);
        $display("PAYLOAD_TRACE projection_axi_read_data_channel_adapter payload_words=0xccddeeff 0x8899aabb 0x44556677 0x00112233 0xdcddfeff 0x98a9babb 0x54657687 0x10213243 count=8 order_ok=1 stability_at_done=1");
        $display("DUT_METADATA_TRACE projection_axi_read_data_channel_adapter accepted=0x%0h last=0x%0h rid_error=0x%0h rresp_error=0x%0h rlast_error=0x%0h status=0x%0h",
                 accepted_beat_trace_o, rlast_trace_o, rid_error_trace_o, rresp_error_trace_o, rlast_error_trace_o, status_o);
        $display("VALIDATION_TRACE projection_axi_read_data_channel_adapter id_ok=1 resp_ok=1 last_ok=1 beat_order_ok=1 done_stability_ok=1");
        start_i = 1'b0;
        payload_ready_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL projection_axi_read_data_channel_adapter done_o did not clear after start_i deasserted");
            $fatal;
        end
        run_bad_metadata_regression();
        $display("PASS projection_axi_read_data_channel_adapter");
        $finish;
    end
endmodule
