`timescale 1ns/1ps

module tb_residual_mlp_fixture;
    localparam int VALUES = 4;
    localparam int ELEM_WIDTH = 16;
    localparam int STATUS_WIDTH = 96;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic done_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] hidden_input_i;
    logic signed [VALUES*ELEM_WIDTH-1:0] attention_output_i;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_o;
    logic [STATUS_WIDTH-1:0] status_o;
    logic signed [VALUES*ELEM_WIDTH-1:0] stable_output_snapshot;
    logic [STATUS_WIDTH-1:0] stable_status_snapshot;
    logic stable_passed;

    residual_mlp_fixture dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .hidden_input_i(hidden_input_i),
        .attention_output_i(attention_output_i),
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

    task automatic set_lane(
        inout logic signed [VALUES*ELEM_WIDTH-1:0] vec_io,
        input integer idx_i,
        input integer value_i
    );
        begin
            vec_io[idx_i*ELEM_WIDTH +: ELEM_WIDTH] = value_i[ELEM_WIDTH-1:0];
        end
    endtask

    task automatic check_lane(
        input string label_i,
        input logic signed [VALUES*ELEM_WIDTH-1:0] vec_i,
        input integer idx_i,
        input integer expected_i
    );
        integer observed;
        begin
            observed = lane_s16(vec_i, idx_i);
            if (observed != expected_i) begin
                $display("FAIL residual_mlp_fixture %s[%0d] observed=%0d expected=%0d", label_i, idx_i, observed, expected_i);
                $fatal;
            end
        end
    endtask

    task automatic check_trace_byte(input integer idx_i, input logic [7:0] expected_i);
        begin
            if (status_o[idx_i*8 +: 8] != expected_i) begin
                $display("FAIL residual_mlp_fixture trace[%0d] observed=0x%0h expected=0x%0h", idx_i, status_o[idx_i*8 +: 8], expected_i);
                $fatal;
            end
        end
    endtask

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        hidden_input_i = '0;
        attention_output_i = '0;
        stable_passed = 1'b0;
        set_lane(hidden_input_i, 0, 3);
        set_lane(hidden_input_i, 1, -2);
        set_lane(hidden_input_i, 2, 5);
        set_lane(hidden_input_i, 3, 1);
        set_lane(attention_output_i, 0, 1);
        set_lane(attention_output_i, 1, 4);
        set_lane(attention_output_i, 2, -3);
        set_lane(attention_output_i, 3, 2);
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #400;
        if (!done_o) begin
            $display("FAIL residual_mlp_fixture done_o was not asserted");
            $fatal;
        end

        check_trace_byte(0, 8'h11);
        check_trace_byte(1, 8'h12);
        check_trace_byte(2, 8'h21);
        check_trace_byte(3, 8'h22);
        check_trace_byte(4, 8'h31);
        check_trace_byte(5, 8'h32);
        check_trace_byte(6, 8'h41);
        check_trace_byte(7, 8'h42);
        check_trace_byte(8, 8'h51);
        check_trace_byte(9, 8'h52);

        check_lane("hidden_input", dut.hidden_r, 0, 3);
        check_lane("hidden_input", dut.hidden_r, 1, -2);
        check_lane("hidden_input", dut.hidden_r, 2, 5);
        check_lane("hidden_input", dut.hidden_r, 3, 1);
        check_lane("attention_output", dut.attention_r, 0, 1);
        check_lane("attention_output", dut.attention_r, 1, 4);
        check_lane("attention_output", dut.attention_r, 2, -3);
        check_lane("attention_output", dut.attention_r, 3, 2);
        check_lane("residual0", dut.residual0_r, 0, 4);
        check_lane("residual0", dut.residual0_r, 1, 2);
        check_lane("residual0", dut.residual0_r, 2, 2);
        check_lane("residual0", dut.residual0_r, 3, 3);
        check_lane("gate", dut.gate_r, 0, 8);
        check_lane("gate", dut.gate_r, 1, 5);
        check_lane("gate", dut.gate_r, 2, -3);
        check_lane("gate", dut.gate_r, 3, 5);
        check_lane("up", dut.up_r, 0, 9);
        check_lane("up", dut.up_r, 1, 2);
        check_lane("up", dut.up_r, 2, 1);
        check_lane("up", dut.up_r, 3, 0);
        check_lane("swiglu", dut.swiglu_r, 0, 9);
        check_lane("swiglu", dut.swiglu_r, 1, 1);
        check_lane("swiglu", dut.swiglu_r, 2, -1);
        check_lane("swiglu", dut.swiglu_r, 3, 0);
        check_lane("down", dut.down_r, 0, 8);
        check_lane("down", dut.down_r, 1, -8);
        check_lane("down", dut.down_r, 2, 16);
        check_lane("down", dut.down_r, 3, 3);
        check_lane("final_output", final_output_o, 0, 12);
        check_lane("final_output", final_output_o, 1, -6);
        check_lane("final_output", final_output_o, 2, 18);
        check_lane("final_output", final_output_o, 3, 6);

        stable_output_snapshot = final_output_o;
        stable_status_snapshot = status_o;
        #20;
        if (!done_o || final_output_o != stable_output_snapshot || status_o != stable_status_snapshot) begin
            stable_passed = 1'b0;
            $display("FAIL residual_mlp_fixture output/status changed while done_o was high");
            $fatal;
        end
        stable_passed = 1'b1;

        $display("RESIDUAL_MLP_TRACE residual_mlp_fixture trace_hex=0x%0h events=residual0_start,residual0_done,gate_up_start,gate_up_done,swiglu_start,swiglu_done,down_start,down_done,residual1_start,residual1_done", status_o[79:0]);
        $display("HIDDEN_INPUT_TRACE residual_mlp_fixture hidden=%0d,%0d,%0d,%0d attention=%0d,%0d,%0d,%0d", lane_s16(dut.hidden_r, 0), lane_s16(dut.hidden_r, 1), lane_s16(dut.hidden_r, 2), lane_s16(dut.hidden_r, 3), lane_s16(dut.attention_r, 0), lane_s16(dut.attention_r, 1), lane_s16(dut.attention_r, 2), lane_s16(dut.attention_r, 3));
        $display("RESIDUAL0_TRACE residual_mlp_fixture residual0=%0d,%0d,%0d,%0d", lane_s16(dut.residual0_r, 0), lane_s16(dut.residual0_r, 1), lane_s16(dut.residual0_r, 2), lane_s16(dut.residual0_r, 3));
        $display("GATE_UP_TRACE residual_mlp_fixture gate=%0d,%0d,%0d,%0d up=%0d,%0d,%0d,%0d source=fixture_constant_projection_matrices", lane_s16(dut.gate_r, 0), lane_s16(dut.gate_r, 1), lane_s16(dut.gate_r, 2), lane_s16(dut.gate_r, 3), lane_s16(dut.up_r, 0), lane_s16(dut.up_r, 1), lane_s16(dut.up_r, 2), lane_s16(dut.up_r, 3));
        $display("SWIGLU_TRACE residual_mlp_fixture policy=bounded_silu_linear_gate_times_up_shift swiglu=%0d,%0d,%0d,%0d true_silu_exp=0", lane_s16(dut.swiglu_r, 0), lane_s16(dut.swiglu_r, 1), lane_s16(dut.swiglu_r, 2), lane_s16(dut.swiglu_r, 3));
        $display("DOWN_TRACE residual_mlp_fixture down=%0d,%0d,%0d,%0d source=fixture_constant_projection_matrix", lane_s16(dut.down_r, 0), lane_s16(dut.down_r, 1), lane_s16(dut.down_r, 2), lane_s16(dut.down_r, 3));
        $display("FINAL_OUTPUT_TRACE residual_mlp_fixture output=%0d,%0d,%0d,%0d status=0x%0h", lane_s16(final_output_o, 0), lane_s16(final_output_o, 1), lane_s16(final_output_o, 2), lane_s16(final_output_o, 3), status_o);
        $display("RESIDUAL_MLP_STABILITY_TRACE residual_mlp_fixture stable=%0d", stable_passed);
        $display("COMPACT_IO_TRACE residual_mlp_fixture estimated_iob_bits=292 exposed_128b_memory_response=0 exposed_full_hidden_vectors=0 exposed_intermediate_tensors=0 exposed_matrices=0 exposed_debug_arrays=0");

        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL residual_mlp_fixture done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS residual_mlp_fixture");
        $finish;
    end
endmodule
