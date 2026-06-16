`timescale 1ns/1ps

module tb_attention_kv_cache_fixture;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [4*16-1:0] output_vec;
    logic [63:0] status_vec;
    logic signed [4*16-1:0] stable_output_snapshot;
    logic [63:0] stable_status_snapshot;
    logic held_write_seen;
    logic accepted_write_stable;
    logic [0:0] held_write_slot;
    logic signed [31:0] held_write_key;
    logic signed [31:0] held_write_value;
    logic signed [31:0] observed_key0;
    logic signed [31:0] observed_key1;
    logic signed [31:0] observed_value0;
    logic signed [31:0] observed_value1;
    integer write_count;
    integer key_read_count;
    integer value_read_count;
    integer observed;

    attention_kv_cache_fixture dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .done_o(done_o),
        .output_o(output_vec),
        .status_o(status_vec)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    function automatic logic signed [7:0] lane_s8(input logic signed [31:0] packed_i, input int idx_i);
        begin
            lane_s8 = packed_i[idx_i*8 +: 8];
        end
    endfunction

    task automatic check_packed_vec(
        input logic signed [31:0] packed_i,
        input logic signed [7:0] exp0_i,
        input logic signed [7:0] exp1_i,
        input logic signed [7:0] exp2_i,
        input logic signed [7:0] exp3_i,
        input [160*8-1:0] label_i
    );
        begin
            if (lane_s8(packed_i, 0) != exp0_i || lane_s8(packed_i, 1) != exp1_i || lane_s8(packed_i, 2) != exp2_i || lane_s8(packed_i, 3) != exp3_i) begin
                $display("FAIL attention_kv_cache_fixture %0s packed=0x%0h values=%0d,%0d,%0d,%0d", label_i, packed_i, lane_s8(packed_i, 0), lane_s8(packed_i, 1), lane_s8(packed_i, 2), lane_s8(packed_i, 3));
                $fatal;
            end
        end
    endtask

    always @(posedge aclk) begin
        if (aresetn) begin
            if (dut.cache_write_valid_r && !dut.cache_write_accept_r) begin
                held_write_seen <= 1'b1;
                held_write_slot <= dut.cache_write_slot_r;
                held_write_key <= dut.cache_write_key_r;
                held_write_value <= dut.cache_write_value_r;
            end
            if (dut.cache_write_valid_r && dut.cache_write_accept_r) begin
                write_count <= write_count + 1;
                if (!held_write_seen || held_write_slot != dut.cache_write_slot_r || held_write_key != dut.cache_write_key_r || held_write_value != dut.cache_write_value_r) begin
                    $display("FAIL attention_kv_cache_fixture write fields changed before accepted write");
                    $fatal;
                end
                accepted_write_stable <= 1'b1;
                if (dut.cache_write_slot_r != 1'b1) begin
                    $display("FAIL attention_kv_cache_fixture write slot=%0d", dut.cache_write_slot_r);
                    $fatal;
                end
                check_packed_vec(dut.cache_write_key_r, -8'sd3, 8'sd2, 8'sd1, 8'sd6, "write_key");
                check_packed_vec(dut.cache_write_value_r, -8'sd2, 8'sd6, -8'sd5, 8'sd4, "write_value");
            end
            if (dut.key_read_valid_r) begin
                key_read_count <= key_read_count + 1;
                if (dut.key_read_slot_r == 1'b0) begin
                    observed_key0 <= dut.key_read_data_r;
                    check_packed_vec(dut.key_read_data_r, 8'sd2, -8'sd1, 8'sd4, 8'sd3, "key_read_slot0");
                end else begin
                    observed_key1 <= dut.key_read_data_r;
                    check_packed_vec(dut.key_read_data_r, -8'sd3, 8'sd2, 8'sd1, 8'sd6, "key_read_slot1");
                end
            end
            if (dut.value_read_valid_r) begin
                value_read_count <= value_read_count + 1;
                if (dut.value_read_slot_r == 1'b0) begin
                    observed_value0 <= dut.value_read_data_r;
                    check_packed_vec(dut.value_read_data_r, 8'sd7, -8'sd4, 8'sd3, 8'sd2, "value_read_slot0");
                end else begin
                    observed_value1 <= dut.value_read_data_r;
                    check_packed_vec(dut.value_read_data_r, -8'sd2, 8'sd6, -8'sd5, 8'sd4, "value_read_slot1");
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        held_write_seen = 1'b0;
        accepted_write_stable = 1'b0;
        held_write_slot = '0;
        held_write_key = '0;
        held_write_value = '0;
        observed_key0 = '0;
        observed_key1 = '0;
        observed_value0 = '0;
        observed_value1 = '0;
        write_count = 0;
        key_read_count = 0;
        value_read_count = 0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #240;
        if (!done_o) begin
            $display("FAIL attention_kv_cache_fixture done_o was not asserted");
            $fatal;
        end
        if (write_count != 1 || !accepted_write_stable) begin
            $display("FAIL attention_kv_cache_fixture write trace count=%0d stable=%0d", write_count, accepted_write_stable);
            $fatal;
        end
        if (key_read_count != 2 || value_read_count != 2) begin
            $display("FAIL attention_kv_cache_fixture read counts key=%0d value=%0d", key_read_count, value_read_count);
            $fatal;
        end
        if (!dut.score0_valid_r || !dut.score1_valid_r || dut.score0_r != 25 || dut.score1_r != -14) begin
            $display("FAIL attention_kv_cache_fixture scores observed=%0d,%0d", dut.score0_r, dut.score1_r);
            $fatal;
        end
        if (!dut.control_valid_r || dut.weight0_r != 8'd12 || dut.weight1_r != 8'd4) begin
            $display("FAIL attention_kv_cache_fixture weights observed=%0d,%0d", dut.weight0_r, dut.weight1_r);
            $fatal;
        end
        if (!dut.output_valid_r || status_vec[0] != 1'b1 || status_vec[4:3] != 2'b11 || status_vec[6:5] != 2'b11 || status_vec[10] != 1'b1 || status_vec[63] != 1'b1) begin
            $display("FAIL attention_kv_cache_fixture compact status=0x%0h", status_vec);
            $fatal;
        end
        observed = $signed({ {16{output_vec[0*16 + 15]}}, output_vec[0*16 +: 16] });
        if (observed != 4) begin
            $display("FAIL attention_kv_cache_fixture output[%0d] observed=%0d expected=4", 0, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[1*16 + 15]}}, output_vec[1*16 +: 16] });
        if (observed != -2) begin
            $display("FAIL attention_kv_cache_fixture output[%0d] observed=%0d expected=-2", 1, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[2*16 + 15]}}, output_vec[2*16 +: 16] });
        if (observed != 1) begin
            $display("FAIL attention_kv_cache_fixture output[%0d] observed=%0d expected=1", 2, observed);
            $fatal;
        end
        observed = $signed({ {16{output_vec[3*16 + 15]}}, output_vec[3*16 +: 16] });
        if (observed != 2) begin
            $display("FAIL attention_kv_cache_fixture output[%0d] observed=%0d expected=2", 3, observed);
            $fatal;
        end
        stable_output_snapshot = output_vec;
        stable_status_snapshot = status_vec;
        #20;
        if (output_vec != stable_output_snapshot || status_vec != stable_status_snapshot || !done_o) begin
            $display("FAIL attention_kv_cache_fixture output/status/done changed while done_o was high");
            $fatal;
        end
        $display("CACHE_WRITE_TRACE attention_kv_cache_fixture count=%0d slot=%0d key=-3,2,1,6 value=-2,6,-5,4 stable=%0d", write_count, dut.cache_write_slot_r, accepted_write_stable);
        $display("KEY_READ_TRACE attention_kv_cache_fixture slots=0,1 keys=2,-1,4,3|-3,2,1,6");
        $display("VALUE_READ_TRACE attention_kv_cache_fixture slots=0,1 values=7,-4,3,2|-2,6,-5,4");
        $display("SCORE_TRACE attention_kv_cache_fixture scores=%0d,%0d", dut.score0_r, dut.score1_r);
        $display("SOFTMAX_CONTROL_TRACE attention_kv_cache_fixture policy=two_score_winner_loser_q0_4 weights=%0d,%0d exp=0", dut.weight0_r, dut.weight1_r);
        $display("OUTPUT_TRACE attention_kv_cache_fixture output=4,-2,1,2");
        $display("NUMERIC_POLICY_TRACE attention_kv_cache_fixture query=3,-2,5,-1 output_shift=4");
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL attention_kv_cache_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS attention_kv_cache_fixture");
        $finish;
    end
endmodule
