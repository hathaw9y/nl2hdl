`timescale 1ns/1ps

module projection_axi_read_command_adapter #(
    parameter int ADDR_WIDTH = 32,
    parameter int MEM_DATA_WIDTH = 128,
    parameter logic [ADDR_WIDTH-1:0] FIXTURE_REQUEST_ADDR = 32'h00120000,
    parameter int FIXTURE_REQUEST_BEATS = 4,
    parameter logic [7:0] FIXTURE_AXI_ID = 8'h02
) (
    input  logic                      aclk,
    input  logic                      aresetn,
    input  logic                      start_i,
    output logic                      done_o,
    output logic                      axi_arvalid_o,
    input  logic                      axi_arready_i,
    output logic [ADDR_WIDTH-1:0]     axi_araddr_o,
    output logic [7:0]                axi_arlen_o,
    output logic [2:0]                axi_arsize_o,
    output logic [1:0]                axi_arburst_o,
    output logic [7:0]                axi_arid_o
);
    localparam int BYTES_PER_BEAT = MEM_DATA_WIDTH / 8;
    localparam logic [2:0] ARSIZE_VALUE = 3'($clog2(BYTES_PER_BEAT));
    localparam logic [7:0] ARLEN_VALUE = 8'(FIXTURE_REQUEST_BEATS - 1);
    localparam logic [1:0] ARBURST_INCR = 2'b01;

    typedef enum logic [1:0] {
        IDLE  = 2'b00,
        ISSUE = 2'b01,
        DONE  = 2'b10
    } state_t;

    state_t state_r;
    logic done_r;
    logic arvalid_r;
    logic [ADDR_WIDTH-1:0] araddr_r;
    logic [7:0] arlen_r;
    logic [2:0] arsize_r;
    logic [1:0] arburst_r;
    logic [7:0] arid_r;

    assign done_o = done_r;
    assign axi_arvalid_o = arvalid_r;
    assign axi_araddr_o = araddr_r;
    assign axi_arlen_o = arlen_r;
    assign axi_arsize_o = arsize_r;
    assign axi_arburst_o = arburst_r;
    assign axi_arid_o = arid_r;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            arvalid_r <= 1'b0;
            araddr_r <= '0;
            arlen_r <= '0;
            arsize_r <= '0;
            arburst_r <= '0;
            arid_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    arvalid_r <= 1'b0;
                    if (start_i) begin
                        araddr_r <= FIXTURE_REQUEST_ADDR;
                        arlen_r <= ARLEN_VALUE;
                        arsize_r <= ARSIZE_VALUE;
                        arburst_r <= ARBURST_INCR;
                        arid_r <= FIXTURE_AXI_ID;
                        arvalid_r <= 1'b1;
                        state_r <= ISSUE;
                    end
                end
                ISSUE: begin
                    if (arvalid_r && axi_arready_i) begin
                        arvalid_r <= 1'b0;
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        done_r <= 1'b0;
                        state_r <= IDLE;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    arvalid_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
