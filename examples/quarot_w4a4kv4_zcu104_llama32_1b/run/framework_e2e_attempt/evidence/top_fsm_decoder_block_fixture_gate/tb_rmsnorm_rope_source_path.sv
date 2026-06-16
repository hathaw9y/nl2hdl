`timescale 1ns/1ps

module tb_rmsnorm_rope_source_path;
    logic aclk;
    logic aresetn;
    logic start_i;
    logic [1:0] norm_token_i;
    logic [3:0] rope_position_i;
    logic done_o;
    logic signed [4*18-1:0] rms_output_vec;
    logic signed [4*18-1:0] rope_output_vec;
    logic [79:0] status_vec;
    logic signed [4*18-1:0] stable_rms_snapshot;
    logic signed [4*18-1:0] stable_rope_snapshot;
    logic [79:0] stable_status_snapshot;
    integer observed;

    rmsnorm_rope_source_path dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(start_i),
        .norm_token_i(norm_token_i),
        .rope_position_i(rope_position_i),
        .done_o(done_o),
        .rms_output_o(rms_output_vec),
        .rope_output_o(rope_output_vec),
        .status_o(status_vec)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        norm_token_i = 2'd0;
        rope_position_i = 4'd7;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #300;
        if (!done_o) begin
            $display("FAIL rmsnorm_rope_source_path done_o was not asserted");
            $fatal;
        end
        if (status_vec[0] != 1'b1 || status_vec[2:1] != 2'd0 || status_vec[18:3] != 16'd7024 || status_vec[34:19] != 16'd5568) begin
            $display("FAIL rmsnorm_rope_source_path rms lookup status=0x%0h", status_vec);
            $fatal;
        end
        if (status_vec[35] != 1'b1 || status_vec[39:36] != 4'd7 || status_vec[40] != 1'b0 || status_vec[48:41] != 8'd13 || status_vec[56:49] != 8'd9) begin
            $display("FAIL rmsnorm_rope_source_path rope pair0 status=0x%0h", status_vec);
            $fatal;
        end
        if (status_vec[57] != 1'b1 || status_vec[61:58] != 4'd7 || status_vec[62] != 1'b1 || status_vec[70:63] != 8'd7 || status_vec[78:71] != 8'hf2) begin
            $display("FAIL rmsnorm_rope_source_path rope pair1 status=0x%0h", status_vec);
            $fatal;
        end
        observed = $signed({ {14{rms_output_vec[0*18 + 17]}}, rms_output_vec[0*18 +: 18] });
        if (observed != 658) begin
            $display("FAIL rmsnorm_rope_source_path rms[%0d] observed=%0d expected=658", 0, observed);
            $fatal;
        end
        observed = $signed({ {14{rms_output_vec[1*18 + 17]}}, rms_output_vec[1*18 +: 18] });
        if (observed != -1372) begin
            $display("FAIL rmsnorm_rope_source_path rms[%0d] observed=%0d expected=-1372", 1, observed);
            $fatal;
        end
        observed = $signed({ {14{rms_output_vec[2*18 + 17]}}, rms_output_vec[2*18 +: 18] });
        if (observed != 1152) begin
            $display("FAIL rmsnorm_rope_source_path rms[%0d] observed=%0d expected=1152", 2, observed);
            $fatal;
        end
        observed = $signed({ {14{rms_output_vec[3*18 + 17]}}, rms_output_vec[3*18 +: 18] });
        if (observed != -659) begin
            $display("FAIL rmsnorm_rope_source_path rms[%0d] observed=%0d expected=-659", 3, observed);
            $fatal;
        end
        observed = $signed({ {14{rope_output_vec[0*18 + 17]}}, rope_output_vec[0*18 +: 18] });
        if (observed != 37) begin
            $display("FAIL rmsnorm_rope_source_path rope[%0d] observed=%0d expected=37", 0, observed);
            $fatal;
        end
        observed = $signed({ {14{rope_output_vec[1*18 + 17]}}, rope_output_vec[1*18 +: 18] });
        if (observed != -13) begin
            $display("FAIL rmsnorm_rope_source_path rope[%0d] observed=%0d expected=-13", 1, observed);
            $fatal;
        end
        observed = $signed({ {14{rope_output_vec[2*18 + 17]}}, rope_output_vec[2*18 +: 18] });
        if (observed != 3) begin
            $display("FAIL rmsnorm_rope_source_path rope[%0d] observed=%0d expected=3", 2, observed);
            $fatal;
        end
        observed = $signed({ {14{rope_output_vec[3*18 + 17]}}, rope_output_vec[3*18 +: 18] });
        if (observed != -42) begin
            $display("FAIL rmsnorm_rope_source_path rope[%0d] observed=%0d expected=-42", 3, observed);
            $fatal;
        end
        stable_rms_snapshot = rms_output_vec;
        stable_rope_snapshot = rope_output_vec;
        stable_status_snapshot = status_vec;
        #20;
        if (rms_output_vec != stable_rms_snapshot || rope_output_vec != stable_rope_snapshot || status_vec != stable_status_snapshot) begin
            $display("FAIL rmsnorm_rope_source_path output/status changed while done_o was high");
            $fatal;
        end
        $display("RMS_LOOKUP_TRACE rmsnorm_rope_source_path selector=%0d valid=%0d inv_rms=%0d sumsq=%0d", status_vec[2:1], status_vec[0], status_vec[18:3], status_vec[34:19]);
        $display("ROPE_LOOKUP_TRACE rmsnorm_rope_source_path position=%0d pair=0 valid=%0d cos=%0d sin=%0d", status_vec[39:36], status_vec[35], $signed(status_vec[48:41]), $signed(status_vec[56:49]));
        $display("ROPE_LOOKUP_TRACE rmsnorm_rope_source_path position=%0d pair=1 valid=%0d cos=%0d sin=%0d", status_vec[61:58], status_vec[57], $signed(status_vec[70:63]), $signed(status_vec[78:71]));
        $display("RMS_OUTPUT_TRACE rmsnorm_rope_source_path %0d %0d %0d %0d", $signed({ {14{rms_output_vec[0*18 + 17]}}, rms_output_vec[0*18 +: 18] }), $signed({ {14{rms_output_vec[1*18 + 17]}}, rms_output_vec[1*18 +: 18] }), $signed({ {14{rms_output_vec[2*18 + 17]}}, rms_output_vec[2*18 +: 18] }), $signed({ {14{rms_output_vec[3*18 + 17]}}, rms_output_vec[3*18 +: 18] }));
        $display("ROPE_OUTPUT_TRACE rmsnorm_rope_source_path %0d %0d %0d %0d", $signed({ {14{rope_output_vec[0*18 + 17]}}, rope_output_vec[0*18 +: 18] }), $signed({ {14{rope_output_vec[1*18 + 17]}}, rope_output_vec[1*18 +: 18] }), $signed({ {14{rope_output_vec[2*18 + 17]}}, rope_output_vec[2*18 +: 18] }), $signed({ {14{rope_output_vec[3*18 + 17]}}, rope_output_vec[3*18 +: 18] }));
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL rmsnorm_rope_source_path done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("PASS rmsnorm_rope_source_path");
        $finish;
    end
endmodule
