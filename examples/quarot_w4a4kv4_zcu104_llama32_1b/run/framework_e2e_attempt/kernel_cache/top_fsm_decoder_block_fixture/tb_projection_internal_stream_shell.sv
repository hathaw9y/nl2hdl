`timescale 1ns/1ps

module tb_projection_internal_stream_shell;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [63:0] output_vec;
    logic [63:0] shell_status_o;
    logic [127:0] expected_mem [0:3];
    logic [31:0] expected_payload [0:15];
    logic [31:0] observed_adapter_payload [0:15];
    logic [31:0] observed_projection_payload [0:15];
    logic signed [63:0] stable_output;
    logic [63:0] stable_shell_status;
    logic [15:0] observed_ready_low_trace;
    logic [23:0] held_req_addr;
    logic [15:0] held_req_beats;
    logic [7:0] held_req_tag;
    logic payload_seen_pending;
    integer observed;
    integer idx;
    integer wait_cycles;
    integer request_count;
    integer response_count;
    integer adapter_observed_count;
    integer projection_observed_count;

    projection_internal_stream_shell dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
        .output_o(output_vec),
        .shell_status_o(shell_status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    always @(negedge aclk) begin
        if (!aresetn || !start_i) begin
            payload_seen_pending = 1'b0;
        end else if (dut.payload_link_valid_w) begin
            if (!dut.payload_link_ready_w && projection_observed_count < 16) begin
                observed_ready_low_trace[projection_observed_count] = 1'b1;
            end
            if (!payload_seen_pending) begin
                if (adapter_observed_count >= 16) begin
                    $display("FAIL projection_internal_stream_shell too many adapter payloads");
                    $fatal;
                end
                observed_adapter_payload[adapter_observed_count] = dut.payload_link_word_w;
                if (dut.payload_link_word_w != expected_payload[adapter_observed_count]) begin
                    $display("FAIL projection_internal_stream_shell adapter payload[%0d] observed=0x%0h expected=0x%0h", adapter_observed_count, dut.payload_link_word_w, expected_payload[adapter_observed_count]);
                    $fatal;
                end
                adapter_observed_count = adapter_observed_count + 1;
                payload_seen_pending = 1'b1;
            end
            if (dut.payload_link_ready_w) begin
                if (projection_observed_count >= 16) begin
                    $display("FAIL projection_internal_stream_shell too many projection payloads");
                    $fatal;
                end
                observed_projection_payload[projection_observed_count] = dut.payload_link_word_w;
                if (dut.payload_link_word_w != expected_payload[projection_observed_count]) begin
                    $display("FAIL projection_internal_stream_shell projection payload[%0d] observed=0x%0h expected=0x%0h", projection_observed_count, dut.payload_link_word_w, expected_payload[projection_observed_count]);
                    $fatal;
                end
                projection_observed_count = projection_observed_count + 1;
                payload_seen_pending = 1'b0;
            end
        end else begin
            payload_seen_pending = 1'b0;
        end
    end

    always @(posedge aclk) begin
        if (aresetn && start_i && dut.mem_rsp_valid_w && dut.mem_rsp_ready_w) begin
            if (response_count >= 4) begin
                $display("FAIL projection_internal_stream_shell too many internal responses");
                $fatal;
            end
            if (dut.mem_rsp_word_w != expected_mem[response_count]) begin
                $display("FAIL projection_internal_stream_shell response[%0d] word=0x%0h expected=0x%0h", response_count, dut.mem_rsp_word_w, expected_mem[response_count]);
                $fatal;
            end
            if (dut.mem_rsp_last_w != (response_count == (4 - 1))) begin
                $display("FAIL projection_internal_stream_shell response[%0d] last=%0b", response_count, dut.mem_rsp_last_w);
                $fatal;
            end
            if (dut.mem_rsp_tag_w != 8'h2a) begin
                $display("FAIL projection_internal_stream_shell response[%0d] tag=0x%0h", response_count, dut.mem_rsp_tag_w);
                $fatal;
            end
            response_count = response_count + 1;
        end
    end

    initial begin
        expected_mem[0] = 128'h61c72d83e94fa50b61c72d83e94fa50b;
        expected_mem[1] = 128'h61c72d83e94fa50b61c72d83e94fa50b;
        expected_mem[2] = 128'h61c72d83e94fa50b61c72d83e94fa50b;
        expected_mem[3] = 128'h61c72d83e94fa50b61c72d83e94fa50b;
        expected_payload[0] = 32'he94fa50b;
        expected_payload[1] = 32'h61c72d83;
        expected_payload[2] = 32'he94fa50b;
        expected_payload[3] = 32'h61c72d83;
        expected_payload[4] = 32'he94fa50b;
        expected_payload[5] = 32'h61c72d83;
        expected_payload[6] = 32'he94fa50b;
        expected_payload[7] = 32'h61c72d83;
        expected_payload[8] = 32'he94fa50b;
        expected_payload[9] = 32'h61c72d83;
        expected_payload[10] = 32'he94fa50b;
        expected_payload[11] = 32'h61c72d83;
        expected_payload[12] = 32'he94fa50b;
        expected_payload[13] = 32'h61c72d83;
        expected_payload[14] = 32'he94fa50b;
        expected_payload[15] = 32'h61c72d83;
        aresetn = 1'b0;
        start_i = 1'b0;
        adapter_observed_count = 0;
        projection_observed_count = 0;
        request_count = 0;
        response_count = 0;
        observed_ready_low_trace = '0;
        payload_seen_pending = 1'b0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;

        wait_cycles = 0;
        while (!dut.mem_req_valid_w && wait_cycles < 20) begin
            #10;
            wait_cycles = wait_cycles + 1;
        end
        if (!dut.mem_req_valid_w) begin
            $display("FAIL projection_internal_stream_shell internal request valid did not assert");
            $fatal;
        end
        held_req_addr = dut.mem_req_addr_w;
        held_req_beats = dut.mem_req_beats_w;
        held_req_tag = dut.mem_req_tag_w;
        repeat (2) begin
            #10;
            if (!dut.mem_req_valid_w || dut.mem_req_addr_w != held_req_addr ||
                dut.mem_req_beats_w != held_req_beats || dut.mem_req_tag_w != held_req_tag) begin
                $display("FAIL projection_internal_stream_shell internal request fields changed under backpressure");
                $fatal;
            end
        end
        if (dut.mem_req_addr_w != 24'h120000 ||
            dut.mem_req_beats_w != 16'd4 || dut.mem_req_tag_w != 8'h2a) begin
            $display("FAIL projection_internal_stream_shell request fields addr=0x%0h beats=%0d tag=0x%0h", dut.mem_req_addr_w, dut.mem_req_beats_w, dut.mem_req_tag_w);
            $fatal;
        end
        wait_cycles = 0;
        while (!dut.request_accepted_r && wait_cycles < 20) begin
            #10;
            wait_cycles = wait_cycles + 1;
        end
        if (!(dut.request_accepted_r)) begin
            $display("FAIL projection_internal_stream_shell internal request was not accepted");
            $fatal;
        end
        request_count = 1;
        #20;
        if (dut.mem_req_valid_w) begin
            $display("FAIL projection_internal_stream_shell issued more than one internal request");
            $fatal;
        end

        #5000;
        if (!done_o) begin
            $display("FAIL projection_internal_stream_shell done_o was not asserted");
            $fatal;
        end
        if (request_count != 1 || response_count != 4) begin
            $display("FAIL projection_internal_stream_shell counts request=%0d response=%0d", request_count, response_count);
            $fatal;
        end
        if (dut.response_stall_trace_r != 4'ha) begin
            $display("FAIL projection_internal_stream_shell response stall trace=0x%0h expected=0xa", dut.response_stall_trace_r);
            $fatal;
        end
        if (adapter_observed_count != 16 || projection_observed_count != 16) begin
            $display("FAIL projection_internal_stream_shell observed payload counts adapter=%0d projection=%0d", adapter_observed_count, projection_observed_count);
            $fatal;
        end
        for (idx = 0; idx < 16; idx = idx + 1) begin
            if (observed_adapter_payload[idx] != observed_projection_payload[idx]) begin
                $display("FAIL projection_internal_stream_shell observed link mismatch[%0d] adapter=0x%0h projection=0x%0h", idx, observed_adapter_payload[idx], observed_projection_payload[idx]);
                $fatal;
            end
        end
        if (dut.child_debug_trace_w[3:0] != 4'hf || dut.child_debug_trace_w[7:4] != 4'h8) begin
            $display("FAIL projection_internal_stream_shell compact response trace debug=0x%0h", dut.child_debug_trace_w);
            $fatal;
        end
        if (dut.child_debug_trace_w[28:24] != 16 || dut.child_debug_trace_w[33:29] != 16) begin
            $display("FAIL projection_internal_stream_shell compact payload counts debug=0x%0h", dut.child_debug_trace_w);
            $fatal;
        end
        if (observed_ready_low_trace != 16'h19 || dut.child_debug_trace_w[23:8] != 16'h19) begin
            $display("FAIL projection_internal_stream_shell ready-low trace observed=0x%0h debug=0x%0h expected=0x19", observed_ready_low_trace, dut.child_debug_trace_w[23:8]);
            $fatal;
        end
        if (dut.child_debug_trace_w[40:34] != 7'd64) begin
            $display("FAIL projection_internal_stream_shell compact parallel pair count=%0d expected=64", dut.child_debug_trace_w[40:34]);
            $fatal;
        end
        if (!dut.child_debug_trace_w[41] || !dut.child_debug_trace_w[42] || dut.child_debug_trace_w[43] || dut.child_debug_trace_w[44] || dut.child_debug_trace_w[45]) begin
            $display("FAIL projection_internal_stream_shell request/error debug=0x%0h", dut.child_debug_trace_w);
            $fatal;
        end
        observed = $signed(output_vec[0*32 +: 32]);
        if (observed != 976) begin
            $display("FAIL projection_internal_stream_shell[%0d] observed=%0d expected=976", 0, observed);
            $fatal;
        end
        observed = $signed(output_vec[1*32 +: 32]);
        if (observed != 2360) begin
            $display("FAIL projection_internal_stream_shell[%0d] observed=%0d expected=2360", 1, observed);
            $fatal;
        end
        stable_output = output_vec;
        stable_shell_status = shell_status_o;
        #20;
        if (!done_o || output_vec != stable_output || shell_status_o != stable_shell_status) begin
            $display("FAIL projection_internal_stream_shell output/status changed while done_o was high");
            $fatal;
        end

        $write("OBSERVED_ADAPTER_PAYLOADS projection_internal_stream_shell");
        for (idx = 0; idx < 16; idx = idx + 1) begin
            $write(" 0x%08h", observed_adapter_payload[idx]);
        end
        $write("\n");
        $write("OBSERVED_CONSUMED_PAYLOADS projection_internal_stream_shell");
        for (idx = 0; idx < 16; idx = idx + 1) begin
            $write(" 0x%08h", observed_projection_payload[idx]);
        end
        $write("\n");
        $display("REQUEST_TRACE projection_internal_stream_shell count=%0d addr=0x%0h beats=%0d tag=0x%0h backpressure_seen=%0d", request_count, held_req_addr, held_req_beats, held_req_tag, dut.child_debug_trace_w[42]);
        $display("RESPONSE_TRACE projection_internal_stream_shell accepted=0x%0h last=0x%0h words=4", dut.child_debug_trace_w[3:0], dut.child_debug_trace_w[7:4]);
        $display("RESPONSE_STALL_TRACE projection_internal_stream_shell stall_before_response_idx=1,3 trace=0x%0h", dut.response_stall_trace_r);
        $display("BACKPRESSURE_TRACE projection_internal_stream_shell ready_low_payload_idx=0,3,4 trace=0x%0h", observed_ready_low_trace);
        $display("PAYLOAD_LINK_TRACE projection_internal_stream_shell payloads=%0d", projection_observed_count);
        $display("PARALLEL_TRACE projection_internal_stream_shell true_lanes=2 pair_count=%0d", dut.child_debug_trace_w[40:34]);
        $display("TOP_IO_TRACE projection_internal_stream_shell exposed_mem_response=0 exposed_request_boundary=0 exposed_payload_data=0 output_bits=64 status_bits=64");
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL projection_internal_stream_shell done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS projection_internal_stream_shell");
        $finish;
    end
endmodule
