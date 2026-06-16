`timescale 1ns/1ps

module tb_projection_target_stream_plan;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic [127:0] mem_word_i;
    logic mem_valid_i;
    logic mem_ready_o;
    logic mem_last_i;
    logic done_o;
    logic signed [63:0] output_vec;
    logic [31:0] payload_link_word_o;
    logic payload_link_valid_o;
    logic payload_link_ready_o;
    logic payload_link_last_o;
    logic [47:0] debug_trace_o;
    logic [127:0] expected_mem [0:3];
    logic [31:0] expected_payload [0:15];
    logic [31:0] observed_adapter_payload [0:15];
    logic [31:0] observed_projection_payload [0:15];
    logic signed [63:0] stable_output;
    logic [47:0] stable_debug_trace;
    logic [15:0] observed_ready_low_trace;
    logic payload_seen_pending;
    integer observed;
    integer idx;
    integer adapter_observed_count;
    integer projection_observed_count;

    projection_target_stream_plan dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .mem_word_i(mem_word_i),
        .mem_valid_i(mem_valid_i),
        .mem_ready_o(mem_ready_o),
        .mem_last_i(mem_last_i),
        .done_o(done_o),
        .output_o(output_vec),
        .payload_link_word_o(payload_link_word_o),
        .payload_link_valid_o(payload_link_valid_o),
        .payload_link_ready_o(payload_link_ready_o),
        .payload_link_last_o(payload_link_last_o),
        .debug_trace_o(debug_trace_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    always @(negedge aclk) begin
        if (!aresetn || !start_i) begin
            payload_seen_pending = 1'b0;
        end else if (payload_link_valid_o) begin
            if (!payload_link_ready_o && projection_observed_count < 16) begin
                observed_ready_low_trace[projection_observed_count] = 1'b1;
            end
            if (!payload_seen_pending) begin
                if (adapter_observed_count >= 16) begin
                    $display("FAIL projection_target_stream_plan too many adapter payloads");
                    $fatal;
                end
                observed_adapter_payload[adapter_observed_count] = payload_link_word_o;
                if (payload_link_word_o != expected_payload[adapter_observed_count]) begin
                    $display("FAIL projection_target_stream_plan adapter payload[%0d] observed=0x%0h expected=0x%0h", adapter_observed_count, payload_link_word_o, expected_payload[adapter_observed_count]);
                    $fatal;
                end
                adapter_observed_count = adapter_observed_count + 1;
                payload_seen_pending = 1'b1;
            end
            if (payload_link_ready_o) begin
                if (projection_observed_count >= 16) begin
                    $display("FAIL projection_target_stream_plan too many projection payloads");
                    $fatal;
                end
                observed_projection_payload[projection_observed_count] = payload_link_word_o;
                if (payload_link_word_o != expected_payload[projection_observed_count]) begin
                    $display("FAIL projection_target_stream_plan projection payload[%0d] observed=0x%0h expected=0x%0h", projection_observed_count, payload_link_word_o, expected_payload[projection_observed_count]);
                    $fatal;
                end
                projection_observed_count = projection_observed_count + 1;
                payload_seen_pending = 1'b0;
            end
        end else begin
            payload_seen_pending = 1'b0;
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
        mem_word_i = '0;
        mem_valid_i = 1'b0;
        mem_last_i = 1'b0;
        adapter_observed_count = 0;
        projection_observed_count = 0;
        observed_ready_low_trace = '0;
        payload_seen_pending = 1'b0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #10;
        if (!mem_ready_o) begin
            $display("FAIL projection_target_stream_plan mem_ready_o not asserted before first input word");
            $fatal;
        end

        for (idx = 0; idx < 4; idx = idx + 1) begin
            while (!mem_ready_o) begin
                #10;
            end
            mem_word_i = expected_mem[idx];
            mem_last_i = (idx == (4 - 1));
            mem_valid_i = 1'b1;
            #10;
            mem_valid_i = 1'b0;
            mem_last_i = 1'b0;
            mem_word_i = '0;
        end

        #5000;
        if (!done_o) begin
            $display("FAIL projection_target_stream_plan done_o was not asserted");
            $fatal;
        end
        if (adapter_observed_count != 16 || projection_observed_count != 16) begin
            $display("FAIL projection_target_stream_plan observed payload counts adapter=%0d projection=%0d", adapter_observed_count, projection_observed_count);
            $fatal;
        end
        for (idx = 0; idx < 16; idx = idx + 1) begin
            if (observed_adapter_payload[idx] != observed_projection_payload[idx]) begin
                $display("FAIL projection_target_stream_plan observed link mismatch[%0d] adapter=0x%0h projection=0x%0h", idx, observed_adapter_payload[idx], observed_projection_payload[idx]);
                $fatal;
            end
        end
        if (debug_trace_o[3:0] != 4'hf || debug_trace_o[7:4] != 4'h8) begin
            $display("FAIL projection_target_stream_plan compact input trace debug=0x%0h", debug_trace_o);
            $fatal;
        end
        if (debug_trace_o[28:24] != 16 || debug_trace_o[33:29] != 16) begin
            $display("FAIL projection_target_stream_plan compact payload counts debug=0x%0h", debug_trace_o);
            $fatal;
        end
        if (observed_ready_low_trace != 16'h19 || debug_trace_o[23:8] != 16'h19) begin
            $display("FAIL projection_target_stream_plan ready-low trace observed=0x%0h debug=0x%0h expected=0x19", observed_ready_low_trace, debug_trace_o[23:8]);
            $fatal;
        end
        if (debug_trace_o[40:34] != 7'd64) begin
            $display("FAIL projection_target_stream_plan compact parallel pair count=%0d expected=64", debug_trace_o[40:34]);
            $fatal;
        end
        observed = $signed(output_vec[0*32 +: 32]);
        if (observed != 976) begin
            $display("FAIL projection_target_stream_plan[%0d] observed=%0d expected=976", 0, observed);
            $fatal;
        end
        observed = $signed(output_vec[1*32 +: 32]);
        if (observed != 2360) begin
            $display("FAIL projection_target_stream_plan[%0d] observed=%0d expected=2360", 1, observed);
            $fatal;
        end
        stable_output = output_vec;
        stable_debug_trace = debug_trace_o;
        #20;
        if (!done_o || output_vec != stable_output || debug_trace_o != stable_debug_trace) begin
            $display("FAIL projection_target_stream_plan output/debug changed while done_o was high");
            $fatal;
        end

        $write("OBSERVED_ADAPTER_PAYLOADS projection_target_stream_plan");
        for (idx = 0; idx < 16; idx = idx + 1) begin
            $write(" 0x%08h", observed_adapter_payload[idx]);
        end
        $write("\n");
        $write("OBSERVED_CONSUMED_PAYLOADS projection_target_stream_plan");
        for (idx = 0; idx < 16; idx = idx + 1) begin
            $write(" 0x%08h", observed_projection_payload[idx]);
        end
        $write("\n");
        $display("INPUT_HANDSHAKE_TRACE projection_target_stream_plan accepted=0x%0h last=0x%0h words=4", debug_trace_o[3:0], debug_trace_o[7:4]);
        $display("BACKPRESSURE_TRACE projection_target_stream_plan ready_low_payload_idx=0,3,4 trace=0x%0h", observed_ready_low_trace);
        $display("PAYLOAD_LINK_TRACE projection_target_stream_plan payloads=%0d", projection_observed_count);
        $display("PARALLEL_TRACE projection_target_stream_plan true_lanes=2 pair_count=%0d", debug_trace_o[40:34]);
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL projection_target_stream_plan done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS projection_target_stream_plan");
        $finish;
    end
endmodule
