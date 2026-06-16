`timescale 1ns/1ps

module model_fsm_axi_decoder_block_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int MODEL_TRACE_WIDTH = 32,
    parameter int TOKEN_STATUS_WIDTH = 64,
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
        LAYER0_START   = 4'd1,
        LAYER0_BUSY    = 4'd2,
        LAYER0_RELEASE = 4'd3,
        LAYER1_START   = 4'd4,
        LAYER1_BUSY    = 4'd5,
        LAYER1_RELEASE = 4'd6,
        DONE           = 4'd7
    } state_t;

    state_t state_r;
    logic done_r;
    logic token_start_r;
    logic token_done_w;
    logic [MODEL_TRACE_WIDTH-1:0] model_trace_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] token_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    logic [TOKEN_STATUS_WIDTH-1:0] token_status_w;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = status_r;

    token_loop_axi_decoder_block_fixture u_token_loop_axi_decoder_block_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(token_start_r),
        .done_o(token_done_w),
        .final_output_o(token_output_w),
        .status_o(token_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            token_start_r <= 1'b0;
            model_trace_r <= '0;
            final_output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    token_start_r <= 1'b0;
                    if (start_i) begin
                        model_trace_r <= '0;
                        final_output_r <= '0;
                        status_r <= '0;
                        state_r <= LAYER0_START;
                    end
                end
                LAYER0_START: begin
                    token_start_r <= 1'b1;
                    model_trace_r[0*8 +: 8] <= 8'h71;
                    state_r <= LAYER0_BUSY;
                end
                LAYER0_BUSY: begin
                    if (token_done_w) begin
                        token_start_r <= 1'b0;
                        model_trace_r[1*8 +: 8] <= 8'h72;
                        state_r <= LAYER0_RELEASE;
                    end
                end
                LAYER0_RELEASE: begin
                    token_start_r <= 1'b0;
                    if (!token_done_w) begin
                        state_r <= LAYER1_START;
                    end
                end
                LAYER1_START: begin
                    token_start_r <= 1'b1;
                    model_trace_r[2*8 +: 8] <= 8'h73;
                    state_r <= LAYER1_BUSY;
                end
                LAYER1_BUSY: begin
                    if (token_done_w) begin
                        token_start_r <= 1'b0;
                        model_trace_r[3*8 +: 8] <= 8'h74;
                        final_output_r <= token_output_w;
                        status_r <= {12'h7b1, token_status_w[48 +: 4], token_status_w[15:0], 8'h74, model_trace_r[23:0]};
                        state_r <= LAYER1_RELEASE;
                    end
                end
                LAYER1_RELEASE: begin
                    token_start_r <= 1'b0;
                    if (!token_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        token_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    token_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
