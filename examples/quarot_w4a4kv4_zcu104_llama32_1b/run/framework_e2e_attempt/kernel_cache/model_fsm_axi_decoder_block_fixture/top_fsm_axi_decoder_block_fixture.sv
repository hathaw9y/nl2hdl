`timescale 1ns/1ps

module top_fsm_axi_decoder_block_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int TOP_TRACE_WIDTH = 16,
    parameter int LAYER_STATUS_WIDTH = 64,
    parameter int STATUS_WIDTH = 64
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    typedef enum logic [2:0] {
        IDLE          = 3'd0,
        LAYER_START   = 3'd1,
        LAYER_BUSY    = 3'd2,
        LAYER_RELEASE = 3'd3,
        DONE          = 3'd4
    } state_t;

    state_t state_r;
    logic done_r;
    logic layer_start_r;
    logic layer_done_w;
    logic [TOP_TRACE_WIDTH-1:0] top_trace_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] layer_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    logic [LAYER_STATUS_WIDTH-1:0] layer_status_w;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = status_r;

    layer_fsm_axi_decoder_block_fixture u_layer_fsm_axi_decoder_block_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(layer_start_r),
        .done_o(layer_done_w),
        .final_output_o(layer_output_w),
        .status_o(layer_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            layer_start_r <= 1'b0;
            top_trace_r <= '0;
            final_output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    layer_start_r <= 1'b0;
                    if (start_i) begin
                        top_trace_r <= '0;
                        final_output_r <= '0;
                        status_r <= '0;
                        state_r <= LAYER_START;
                    end
                end
                LAYER_START: begin
                    layer_start_r <= 1'b1;
                    top_trace_r[0*8 +: 8] <= 8'h53;
                    state_r <= LAYER_BUSY;
                end
                LAYER_BUSY: begin
                    if (layer_done_w) begin
                        layer_start_r <= 1'b0;
                        top_trace_r[1*8 +: 8] <= 8'h54;
                        final_output_r <= layer_output_w;
                        status_r <= {12'h54a, layer_status_w[48 +: 4], layer_status_w[16 +: 16], layer_status_w[15:0], 8'h54, top_trace_r[7:0]};
                        state_r <= LAYER_RELEASE;
                    end
                end
                LAYER_RELEASE: begin
                    if (!layer_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        layer_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    layer_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
