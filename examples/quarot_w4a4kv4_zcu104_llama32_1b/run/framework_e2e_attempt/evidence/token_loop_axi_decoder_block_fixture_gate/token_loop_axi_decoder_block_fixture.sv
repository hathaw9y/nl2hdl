`timescale 1ns/1ps

module token_loop_axi_decoder_block_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int TOKEN_TRACE_WIDTH = 32,
    parameter int TOP_STATUS_WIDTH = 64,
    parameter int STATUS_WIDTH = 64
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    typedef enum logic [3:0] {
        IDLE           = 4'd0,
        TOKEN0_START   = 4'd1,
        TOKEN0_BUSY    = 4'd2,
        TOKEN0_RELEASE = 4'd3,
        TOKEN1_START   = 4'd4,
        TOKEN1_BUSY    = 4'd5,
        TOKEN1_RELEASE = 4'd6,
        DONE           = 4'd7
    } state_t;

    state_t state_r;
    logic done_r;
    logic top_start_r;
    logic top_done_w;
    logic [TOKEN_TRACE_WIDTH-1:0] token_trace_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] top_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    logic [TOP_STATUS_WIDTH-1:0] top_status_w;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = status_r;

    top_fsm_axi_decoder_block_fixture u_top_fsm_axi_decoder_block_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(top_start_r),
        .done_o(top_done_w),
        .final_output_o(top_output_w),
        .status_o(top_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            top_start_r <= 1'b0;
            token_trace_r <= '0;
            final_output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    top_start_r <= 1'b0;
                    if (start_i) begin
                        token_trace_r <= '0;
                        final_output_r <= '0;
                        status_r <= '0;
                        state_r <= TOKEN0_START;
                    end
                end
                TOKEN0_START: begin
                    top_start_r <= 1'b1;
                    token_trace_r[0*8 +: 8] <= 8'h61;
                    state_r <= TOKEN0_BUSY;
                end
                TOKEN0_BUSY: begin
                    if (top_done_w) begin
                        top_start_r <= 1'b0;
                        token_trace_r[1*8 +: 8] <= 8'h62;
                        state_r <= TOKEN0_RELEASE;
                    end
                end
                TOKEN0_RELEASE: begin
                    top_start_r <= 1'b0;
                    if (!top_done_w) begin
                        state_r <= TOKEN1_START;
                    end
                end
                TOKEN1_START: begin
                    top_start_r <= 1'b1;
                    token_trace_r[2*8 +: 8] <= 8'h63;
                    state_r <= TOKEN1_BUSY;
                end
                TOKEN1_BUSY: begin
                    if (top_done_w) begin
                        top_start_r <= 1'b0;
                        token_trace_r[3*8 +: 8] <= 8'h64;
                        final_output_r <= top_output_w;
                        status_r <= {12'h6a1, top_status_w[48 +: 4], top_status_w[15:0], 8'h64, token_trace_r[23:0]};
                        state_r <= TOKEN1_RELEASE;
                    end
                end
                TOKEN1_RELEASE: begin
                    top_start_r <= 1'b0;
                    if (!top_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        top_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    top_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
