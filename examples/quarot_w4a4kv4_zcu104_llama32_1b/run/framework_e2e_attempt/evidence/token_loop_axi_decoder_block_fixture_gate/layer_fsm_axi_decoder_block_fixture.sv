`timescale 1ns/1ps

module layer_fsm_axi_decoder_block_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int LAYER_TRACE_WIDTH = 16,
    parameter int BLOCK_STATUS_WIDTH = 176,
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
        BLOCK_START   = 3'd1,
        BLOCK_BUSY    = 3'd2,
        BLOCK_RELEASE = 3'd3,
        DONE          = 3'd4
    } state_t;

    state_t state_r;
    logic done_r;
    logic block_start_r;
    logic block_done_w;
    logic [LAYER_TRACE_WIDTH-1:0] layer_trace_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] block_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    logic [BLOCK_STATUS_WIDTH-1:0] block_status_w;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = status_r;

    decoder_block_axi_attention_mlp_fixture u_decoder_block_axi_attention_mlp_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(block_start_r),
        .done_o(block_done_w),
        .final_output_o(block_output_w),
        .status_o(block_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            block_start_r <= 1'b0;
            layer_trace_r <= '0;
            final_output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    block_start_r <= 1'b0;
                    if (start_i) begin
                        layer_trace_r <= '0;
                        final_output_r <= '0;
                        status_r <= '0;
                        state_r <= BLOCK_START;
                    end
                end
                BLOCK_START: begin
                    block_start_r <= 1'b1;
                    layer_trace_r[0*8 +: 8] <= 8'h41;
                    state_r <= BLOCK_BUSY;
                end
                BLOCK_BUSY: begin
                    if (block_done_w) begin
                        block_start_r <= 1'b0;
                        layer_trace_r[1*8 +: 8] <= 8'h42;
                        final_output_r <= block_output_w;
                        status_r <= {12'h4a1, block_status_w[80 +: 4], block_status_w[31:0], 8'h42, layer_trace_r[7:0]};
                        state_r <= BLOCK_RELEASE;
                    end
                end
                BLOCK_RELEASE: begin
                    if (!block_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        block_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    block_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
