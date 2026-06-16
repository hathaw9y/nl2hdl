`timescale 1ns/1ps

module tb_quarot_w4a4kv4_support;

    localparam int N         = 4;
    localparam int I8_WIDTH  = 8;
    localparam int A4_WIDTH  = 4;
    localparam int ROT_WIDTH = 10;

    logic                         aclk;
    logic                         aresetn;
    logic                         start_i;
    logic signed [N*I8_WIDTH-1:0] input_vec_i;
    logic        [N*A4_WIDTH-1:0] kv_vec0_i;
    logic        [N*A4_WIDTH-1:0] kv_vec1_i;
    logic                         done_o;
    logic signed [N*ROT_WIDTH-1:0] rotated_vec_o;
    logic        [N*A4_WIDTH-1:0] a4_packed_o;
    logic        [N*A4_WIDTH-1:0] kv_vec0_roundtrip_o;
    logic        [N*A4_WIDTH-1:0] kv_vec1_roundtrip_o;
    logic        [7:0]            status_o;

    quarot_w4a4kv4_support #(
        .N(N),
        .I8_WIDTH(I8_WIDTH),
        .A4_WIDTH(A4_WIDTH),
        .ROT_WIDTH(ROT_WIDTH)
    ) dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .input_vec_i(input_vec_i),
        .kv_vec0_i(kv_vec0_i),
        .kv_vec1_i(kv_vec1_i),
        .done_o(done_o),
        .rotated_vec_o(rotated_vec_o),
        .a4_packed_o(a4_packed_o),
        .kv_vec0_roundtrip_o(kv_vec0_roundtrip_o),
        .kv_vec1_roundtrip_o(kv_vec1_roundtrip_o),
        .status_o(status_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    function automatic logic [N*I8_WIDTH-1:0] pack_i8(
        input logic signed [I8_WIDTH-1:0] x0_i,
        input logic signed [I8_WIDTH-1:0] x1_i,
        input logic signed [I8_WIDTH-1:0] x2_i,
        input logic signed [I8_WIDTH-1:0] x3_i
    );
        begin
            pack_i8[0*I8_WIDTH +: I8_WIDTH] = x0_i;
            pack_i8[1*I8_WIDTH +: I8_WIDTH] = x1_i;
            pack_i8[2*I8_WIDTH +: I8_WIDTH] = x2_i;
            pack_i8[3*I8_WIDTH +: I8_WIDTH] = x3_i;
        end
    endfunction

    function automatic logic [N*A4_WIDTH-1:0] pack_kv4(
        input logic [A4_WIDTH-1:0] x0_i,
        input logic [A4_WIDTH-1:0] x1_i,
        input logic [A4_WIDTH-1:0] x2_i,
        input logic [A4_WIDTH-1:0] x3_i
    );
        begin
            pack_kv4[0*A4_WIDTH +: A4_WIDTH] = x0_i;
            pack_kv4[1*A4_WIDTH +: A4_WIDTH] = x1_i;
            pack_kv4[2*A4_WIDTH +: A4_WIDTH] = x2_i;
            pack_kv4[3*A4_WIDTH +: A4_WIDTH] = x3_i;
        end
    endfunction

    function automatic logic signed [ROT_WIDTH-1:0] rot_elem(
        input logic signed [N*ROT_WIDTH-1:0] vec_i,
        input int idx_i
    );
        begin
            rot_elem = $signed(vec_i[idx_i*ROT_WIDTH +: ROT_WIDTH]);
        end
    endfunction

    task automatic assert_equal_logic(
        input string label_i,
        input logic [63:0] got_i,
        input logic [63:0] expected_i
    );
        begin
            if (got_i !== expected_i) begin
                $fatal(1, "%s got 0x%0h expected 0x%0h", label_i, got_i, expected_i);
            end
        end
    endtask

    task automatic assert_equal_signed(
        input string label_i,
        input logic signed [ROT_WIDTH-1:0] got_i,
        input logic signed [ROT_WIDTH-1:0] expected_i
    );
        begin
            if (got_i !== expected_i) begin
                $fatal(1, "%s got %0d expected %0d", label_i, got_i, expected_i);
            end
        end
    endtask

    task automatic check_first_vector;
        begin
            assert_equal_logic("done_o first vector", {63'b0, done_o}, 64'd1);
            assert_equal_signed("rot[0] first vector", rot_elem(rotated_vec_o, 0), -10'sd40);
            assert_equal_signed("rot[1] first vector", rot_elem(rotated_vec_o, 1),  10'sd100);
            assert_equal_signed("rot[2] first vector", rot_elem(rotated_vec_o, 2),  10'sd20);
            assert_equal_signed("rot[3] first vector", rot_elem(rotated_vec_o, 3),  10'sd0);
            assert_equal_logic("a4_packed first vector", {48'b0, a4_packed_o}, 64'h0000_0000_0000_0778);
            assert_equal_logic("kv_vec0 roundtrip first vector", {48'b0, kv_vec0_roundtrip_o}, 64'h0000_0000_0000_78f1);
            assert_equal_logic("kv_vec1 roundtrip first vector", {48'b0, kv_vec1_roundtrip_o}, 64'h0000_0000_0000_9e20);
            assert_equal_logic("status first vector", {56'b0, status_o}, 64'h0000_0000_0000_001f);
        end
    endtask

    task automatic check_second_vector;
        begin
            assert_equal_logic("done_o second vector", {63'b0, done_o}, 64'd1);
            assert_equal_signed("rot[0] second vector", rot_elem(rotated_vec_o, 0),  10'sd10);
            assert_equal_signed("rot[1] second vector", rot_elem(rotated_vec_o, 1), -10'sd2);
            assert_equal_signed("rot[2] second vector", rot_elem(rotated_vec_o, 2), -10'sd4);
            assert_equal_signed("rot[3] second vector", rot_elem(rotated_vec_o, 3),  10'sd0);
            assert_equal_logic("a4_packed second vector", {48'b0, a4_packed_o}, 64'h0000_0000_0000_0ce7);
            assert_equal_logic("kv_vec0 roundtrip second vector", {48'b0, kv_vec0_roundtrip_o}, 64'h0000_0000_0000_4321);
            assert_equal_logic("kv_vec1 roundtrip second vector", {48'b0, kv_vec1_roundtrip_o}, 64'h0000_0000_0000_dcba);
            assert_equal_logic("status second vector", {56'b0, status_o}, 64'h0000_0000_0000_000b);
        end
    endtask

    initial begin
        logic signed [N*ROT_WIDTH-1:0] stable_rotated;
        logic [N*A4_WIDTH-1:0] stable_a4;
        logic [N*A4_WIDTH-1:0] stable_kv0;
        logic [N*A4_WIDTH-1:0] stable_kv1;
        logic [7:0] stable_status;

        aresetn = 1'b0;
        start_i = 1'b0;
        input_vec_i = '0;
        kv_vec0_i = '0;
        kv_vec1_i = '0;

        repeat (4) @(posedge aclk);
        aresetn = 1'b1;
        @(posedge aclk);
        #1;
        assert_equal_logic("done_o after reset", {63'b0, done_o}, 64'd0);

        input_vec_i = pack_i8(8'sd20, -8'sd30, 8'sd10, -8'sd40);
        kv_vec0_i = pack_kv4(4'h1, 4'hf, 4'h8, 4'h7);
        kv_vec1_i = pack_kv4(4'h0, 4'h2, 4'he, 4'h9);
        start_i = 1'b1;
        @(posedge aclk);
        #1;
        check_first_vector();

        stable_rotated = rotated_vec_o;
        stable_a4 = a4_packed_o;
        stable_kv0 = kv_vec0_roundtrip_o;
        stable_kv1 = kv_vec1_roundtrip_o;
        stable_status = status_o;

        input_vec_i = pack_i8(-8'sd1, -8'sd2, -8'sd3, -8'sd4);
        kv_vec0_i = pack_kv4(4'ha, 4'hb, 4'hc, 4'hd);
        kv_vec1_i = pack_kv4(4'h5, 4'h6, 4'h7, 4'h8);
        repeat (3) begin
            @(posedge aclk);
            #1;
            assert_equal_logic("done_o held while start_i high", {63'b0, done_o}, 64'd1);
            assert_equal_logic("rotated stable while done_o high", rotated_vec_o, stable_rotated);
            assert_equal_logic("a4 stable while done_o high", {48'b0, a4_packed_o}, {48'b0, stable_a4});
            assert_equal_logic("kv0 stable while done_o high", {48'b0, kv_vec0_roundtrip_o}, {48'b0, stable_kv0});
            assert_equal_logic("kv1 stable while done_o high", {48'b0, kv_vec1_roundtrip_o}, {48'b0, stable_kv1});
            assert_equal_logic("status stable while done_o high", {56'b0, status_o}, {56'b0, stable_status});
        end

        start_i = 1'b0;
        @(posedge aclk);
        #1;
        assert_equal_logic("done_o release after start_i deassert", {63'b0, done_o}, 64'd0);

        input_vec_i = pack_i8(8'sd1, 8'sd2, 8'sd3, 8'sd4);
        kv_vec0_i = pack_kv4(4'h1, 4'h2, 4'h3, 4'h4);
        kv_vec1_i = pack_kv4(4'ha, 4'hb, 4'hc, 4'hd);
        start_i = 1'b1;
        @(posedge aclk);
        #1;
        check_second_vector();

        start_i = 1'b0;
        @(posedge aclk);
        #1;
        assert_equal_logic("done_o final release", {63'b0, done_o}, 64'd0);

        $display("PASS tb_quarot_w4a4kv4_support");
        $finish;
    end

endmodule
