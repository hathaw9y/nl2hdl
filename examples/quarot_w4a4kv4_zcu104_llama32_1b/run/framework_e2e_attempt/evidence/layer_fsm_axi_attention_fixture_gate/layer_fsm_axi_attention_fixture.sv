`timescale 1ns/1ps

module layer_fsm_axi_attention_fixture #(
    parameter int LAYER_TRACE_WIDTH = 16,
    parameter int CHILD_STATUS_WIDTH = 96,
    parameter int OUT_VALUES = 4,
    parameter int OUT_WIDTH = 16,
    parameter int STATUS_WIDTH = LAYER_TRACE_WIDTH + CHILD_STATUS_WIDTH
) (
    input  logic                                    aclk,
    input  logic                                    aresetn,
    input  logic                                    start_i,
    output logic                                    done_o,
    output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o,
    output logic [STATUS_WIDTH-1:0]                 status_o
);
    typedef enum logic [2:0] {
        IDLE        = 3'd0,
        CHILD_START = 3'd1,
        CHILD_BUSY  = 3'd2,
        CHILD_RELEASE = 3'd3,
        DONE          = 3'd4
    } state_t;

    state_t state_r;
    logic done_r;
    logic child_start_r;
    logic child_done_w;
    logic [LAYER_TRACE_WIDTH-1:0] layer_trace_r;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] output_r;
    logic [STATUS_WIDTH-1:0] status_r;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] child_output_w;
    logic [CHILD_STATUS_WIDTH-1:0] child_status_w;

    assign done_o = done_r;
    assign output_o = output_r;
    assign status_o = status_r;

    decoder_child_axi_attention_datapath u_decoder_child_axi_attention_datapath (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(child_start_r),
        .done_o(child_done_w),
        .output_o(child_output_w),
        .status_o(child_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            child_start_r <= 1'b0;
            layer_trace_r <= '0;
            output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    child_start_r <= 1'b0;
                    if (start_i) begin
                        layer_trace_r <= '0;
                        output_r <= '0;
                        status_r <= '0;
                        state_r <= CHILD_START;
                    end
                end
                CHILD_START: begin
                    child_start_r <= 1'b1;
                    layer_trace_r[0*8 +: 8] <= 8'h41;
                    state_r <= CHILD_BUSY;
                end
                CHILD_BUSY: begin
                    if (child_done_w) begin
                        child_start_r <= 1'b0;
                        layer_trace_r[1*8 +: 8] <= 8'h42;
                        output_r <= child_output_w;
                        status_r <= {child_status_w, 8'h42, layer_trace_r[7:0]};
                        state_r <= CHILD_RELEASE;
                    end
                end
                CHILD_RELEASE: begin
                    if (!child_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        child_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    child_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
