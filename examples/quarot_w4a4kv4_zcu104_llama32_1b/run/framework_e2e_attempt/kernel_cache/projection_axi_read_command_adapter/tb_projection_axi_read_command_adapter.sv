`timescale 1ns/1ps

module tb_projection_axi_read_command_adapter;
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
    logic [31:0] stable_araddr;
    logic [7:0] stable_arlen;
    logic [2:0] stable_arsize;
    logic [1:0] stable_arburst;
    logic [7:0] stable_arid;
    integer stall_cycles;

    projection_axi_read_command_adapter dut (
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
        .axi_arid_o(axi_arid_o)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    task automatic check_fields_stable;
        begin
            if (axi_araddr_o !== stable_araddr || axi_arlen_o !== stable_arlen ||
                axi_arsize_o !== stable_arsize || axi_arburst_o !== stable_arburst ||
                axi_arid_o !== stable_arid) begin
                $display("FAIL projection_axi_read_command_adapter AR field changed during ready-low stall");
                $fatal;
            end
        end
    endtask

    initial begin
        aresetn = 1'b0;
        start_i = 1'b0;
        axi_arready_i = 1'b0;
        stall_cycles = 0;
        #20;
        aresetn = 1'b1;
        #10;
        start_i = 1'b1;
        #10;
        if (!axi_arvalid_o) begin
            $display("FAIL projection_axi_read_command_adapter axi_arvalid_o not asserted");
            $fatal;
        end
        if (axi_araddr_o != 32'h00120000 ||
            axi_arlen_o != 8'h03 ||
            axi_arsize_o != 3'h4 ||
            axi_arburst_o != 2'b01 ||
            axi_arid_o != 8'h02) begin
            $display("FAIL projection_axi_read_command_adapter unexpected AR command addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h",
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
                $display("FAIL projection_axi_read_command_adapter ready-low stall was not maintained");
                $fatal;
            end
            check_fields_stable();
            stall_cycles = stall_cycles + 1;
        end
        axi_arready_i = 1'b1;
        #10;
        if (!done_o || axi_arvalid_o) begin
            $display("FAIL projection_axi_read_command_adapter handshake did not complete done=%0b arvalid=%0b", done_o, axi_arvalid_o);
            $fatal;
        end
        check_fields_stable();
        axi_arready_i = 1'b0;
        #20;
        if (!done_o) begin
            $display("FAIL projection_axi_read_command_adapter done_o did not hold while start_i remained high");
            $fatal;
        end
        start_i = 1'b0;
        #20;
        if (done_o) begin
            $display("FAIL projection_axi_read_command_adapter done_o did not clear after start_i deasserted");
            $fatal;
        end
        $display("AXI_COMMAND_TRACE projection_axi_read_command_adapter addr=0x%0h len=%0d size=%0d burst=%0d id=0x%0h beats=4",
                 stable_araddr, stable_arlen, stable_arsize, stable_arburst, stable_arid);
        $display("AXI_BACKPRESSURE_TRACE projection_axi_read_command_adapter ready_low_cycles=%0d arvalid_held=1", stall_cycles);
        $display("AXI_FIELD_STABILITY_TRACE projection_axi_read_command_adapter stable_during_ready_low=1 checked_fields=addr,len,size,burst,id");
        $display("PASS projection_axi_read_command_adapter");
        $finish;
    end
endmodule
