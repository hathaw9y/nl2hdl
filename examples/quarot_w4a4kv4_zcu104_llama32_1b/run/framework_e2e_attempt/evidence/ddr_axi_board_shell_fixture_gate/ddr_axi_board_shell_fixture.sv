`timescale 1ns/1ps

module ddr_axi_board_shell_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int SHELL_TRACE_WIDTH = 16,
    parameter int MODEL_STATUS_WIDTH = 64,
    parameter int STATUS_WIDTH = 64
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    localparam logic [3:0] ATTENTION_REQUEST_MASK = 4'hf;
    localparam logic [2:0] MLP_REQUEST_MASK       = 3'h7;
    localparam logic [6:0] ALL_REQUEST_MASK       = 7'h7f;
    localparam logic [5:0] PROJECTION_COUNT       = 6'd7;

    typedef enum logic [2:0] {
        IDLE          = 3'd0,
        MODEL_START   = 3'd1,
        MODEL_BUSY    = 3'd2,
        MODEL_RELEASE = 3'd3,
        DONE          = 3'd4
    } state_t;

    state_t state_r;
    logic done_r;
    logic model_start_r;
    logic model_done_w;
    logic [SHELL_TRACE_WIDTH-1:0] shell_trace_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] model_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    logic [MODEL_STATUS_WIDTH-1:0] model_status_w;
    logic [STATUS_WIDTH-1:0] status_r;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = status_r;

    model_fsm_axi_decoder_block_fixture u_model_fsm_axi_decoder_block_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(model_start_r),
        .done_o(model_done_w),
        .final_output_o(model_output_w),
        .status_o(model_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            model_start_r <= 1'b0;
            shell_trace_r <= '0;
            final_output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    model_start_r <= 1'b0;
                    if (start_i) begin
                        shell_trace_r <= '0;
                        final_output_r <= '0;
                        status_r <= '0;
                        state_r <= MODEL_START;
                    end
                end
                MODEL_START: begin
                    model_start_r <= 1'b1;
                    shell_trace_r[0*8 +: 8] <= 8'h81;
                    state_r <= MODEL_BUSY;
                end
                MODEL_BUSY: begin
                    if (model_done_w) begin
                        model_start_r <= 1'b0;
                        shell_trace_r[1*8 +: 8] <= 8'h82;
                        final_output_r <= model_output_w;
                        status_r <= {
                            8'hdd,
                            PROJECTION_COUNT,
                            model_status_w[48 +: 4],
                            ALL_REQUEST_MASK,
                            MLP_REQUEST_MASK,
                            ATTENTION_REQUEST_MASK,
                            model_status_w[15:0],
                            8'h82,
                            shell_trace_r[7:0]
                        };
                        state_r <= MODEL_RELEASE;
                    end
                end
                MODEL_RELEASE: begin
                    model_start_r <= 1'b0;
                    if (!model_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        model_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    model_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
