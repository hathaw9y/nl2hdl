`timescale 1ns/1ps

module tb_w4a4_ws_systolic_tile;
    localparam int ROWS = 2;
    localparam int COLS = 3;
    localparam int K = 4;
    localparam int W_BITS = 4;
    localparam int A_BITS = 4;
    localparam int ACC_BITS = 32;
    localparam int ACT_PACK_W = ROWS * K * A_BITS;
    localparam int WGT_PACK_W = K * COLS * W_BITS;
    localparam int OUT_PACK_W = ROWS * COLS * ACC_BITS;

    logic aclk;
    logic aresetn;
    logic start_i;
    logic [ACT_PACK_W-1:0] activation_tile_i;
    logic [WGT_PACK_W-1:0] weight_tile_i;
    logic done_o;
    logic signed [OUT_PACK_W-1:0] acc_tile_o;
    logic [31:0] status_o;

    w4a4_ws_systolic_tile #(
        .ROWS(ROWS),
        .COLS(COLS),
        .K(K),
        .W_BITS(W_BITS),
        .A_BITS(A_BITS),
        .ACC_BITS(ACC_BITS)
    ) dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .activation_tile_i(activation_tile_i),
        .weight_tile_i(weight_tile_i),
        .done_o(done_o),
        .acc_tile_o(acc_tile_o),
        .status_o(status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    function automatic logic [3:0] pack_int4(input int value_i);
        begin
            pack_int4 = value_i[3:0];
        end
    endfunction

    function automatic int decode_int4(input logic [3:0] nibble_i);
        begin
            decode_int4 = int'($signed(nibble_i));
        end
    endfunction

    function automatic int acc_at(input logic signed [OUT_PACK_W-1:0] packed_i, input int row_idx_i, input int col_idx_i);
        logic signed [ACC_BITS-1:0] element;
        begin
            element = packed_i[((row_idx_i*COLS)+col_idx_i)*ACC_BITS +: ACC_BITS];
            acc_at = int'(element);
        end
    endfunction

    task automatic set_activation(input int row_idx_i, input int k_idx_i, input int value_i);
        int flat_idx;
        begin
            flat_idx = (row_idx_i * K) + k_idx_i;
            activation_tile_i[flat_idx*A_BITS +: A_BITS] = pack_int4(value_i);
        end
    endtask

    task automatic set_weight(input int k_idx_i, input int col_idx_i, input int value_i);
        int flat_idx;
        begin
            flat_idx = (k_idx_i * COLS) + col_idx_i;
            weight_tile_i[flat_idx*W_BITS +: W_BITS] = pack_int4(value_i);
        end
    endtask

    task automatic expect_int(input string label_i, input int got_i, input int expected_i);
        begin
            if (got_i !== expected_i) begin
                $fatal(1, "%s got=%0d expected=%0d", label_i, got_i, expected_i);
            end
        end
    endtask

    task automatic expect_bit(input string label_i, input logic got_i, input logic expected_i);
        begin
            if (got_i !== expected_i) begin
                $fatal(1, "%s got=%0b expected=%0b", label_i, got_i, expected_i);
            end
        end
    endtask

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        activation_tile_i = '0;
        weight_tile_i = '0;

        set_activation(0, 0, -8);
        set_activation(0, 1, -1);
        set_activation(0, 2, 0);
        set_activation(0, 3, 7);
        set_activation(1, 0, 3);
        set_activation(1, 1, -4);
        set_activation(1, 2, 5);
        set_activation(1, 3, 2);

        set_weight(0, 0, 1);
        set_weight(0, 1, -2);
        set_weight(0, 2, 7);
        set_weight(1, 0, -8);
        set_weight(1, 1, 3);
        set_weight(1, 2, 0);
        set_weight(2, 0, 4);
        set_weight(2, 1, -1);
        set_weight(2, 2, -3);
        set_weight(3, 0, 2);
        set_weight(3, 1, 5);
        set_weight(3, 2, -6);

        repeat (4) @(posedge aclk);
        aresetn = 1'b1;
        @(posedge aclk);

        expect_int("activation decode row0 k0", decode_int4(activation_tile_i[0 +: 4]), -8);
        expect_int("activation decode row0 k1", decode_int4(activation_tile_i[4 +: 4]), -1);
        expect_int("weight decode k3 col2", decode_int4(weight_tile_i[((3*COLS)+2)*4 +: 4]), -6);

        start_i = 1'b1;
        @(posedge aclk);
        #1;
        expect_bit("done not immediate", done_o, 1'b0);
        expect_bit("weight loaded status", status_o[2], 1'b1);

        @(posedge aclk);
        #1;
        expect_bit("products valid after first k slice", status_o[1], 1'b1);
        if (dut.weight_hold_r !== weight_tile_i) begin
            $fatal(1, "weight stationary hold register changed after load");
        end

        wait (done_o === 1'b1);
        #1;
        expect_int("acc row0 col0", acc_at(acc_tile_o, 0, 0), 14);
        expect_int("acc row0 col1", acc_at(acc_tile_o, 0, 1), 48);
        expect_int("acc row0 col2", acc_at(acc_tile_o, 0, 2), -98);
        expect_int("acc row1 col0", acc_at(acc_tile_o, 1, 0), 59);
        expect_int("acc row1 col1", acc_at(acc_tile_o, 1, 1), -13);
        expect_int("acc row1 col2", acc_at(acc_tile_o, 1, 2), -6);
        expect_int("true products per cycle status", int'(status_o[23:16]), ROWS * COLS);

        begin
            logic signed [OUT_PACK_W-1:0] stable_output;
            logic [31:0] stable_status;
            stable_output = acc_tile_o;
            stable_status = status_o;
            repeat (3) begin
                @(posedge aclk);
                #1;
                expect_bit("done hold while start asserted", done_o, 1'b1);
                if (acc_tile_o !== stable_output) begin
                    $fatal(1, "acc_tile_o changed while done_o was high");
                end
                if (status_o !== stable_status) begin
                    $fatal(1, "status_o changed while done_o was high");
                end
            end
        end

        start_i = 1'b0;
        @(posedge aclk);
        #1;
        expect_bit("done release after start deassert", done_o, 1'b0);
        expect_bit("weight loaded status release", status_o[2], 1'b0);

        $display("PASS w4a4_ws_systolic_tile");
        $finish;
    end
endmodule
