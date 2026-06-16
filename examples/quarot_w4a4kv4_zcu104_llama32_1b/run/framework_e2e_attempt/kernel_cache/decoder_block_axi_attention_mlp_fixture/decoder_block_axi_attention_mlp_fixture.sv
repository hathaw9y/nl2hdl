`timescale 1ns/1ps

module decoder_block_axi_attention_mlp_fixture #(
    parameter int VALUES = 4,
    parameter int ELEM_WIDTH = 16,
    parameter int BLOCK_TRACE_WIDTH = 32,
    parameter int ATTENTION_STATUS_WIDTH = 96,
    parameter int MLP_STATUS_WIDTH = 96,
    parameter int STATUS_WIDTH = 176
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    output logic signed [VALUES*ELEM_WIDTH-1:0]       final_output_o,
    output logic [STATUS_WIDTH-1:0]                   status_o
);
    typedef enum logic [3:0] {
        IDLE                  = 4'd0,
        AXI_ATTENTION_START   = 4'd1,
        AXI_ATTENTION_BUSY    = 4'd2,
        AXI_ATTENTION_RELEASE = 4'd3,
        MLP_START             = 4'd4,
        MLP_BUSY              = 4'd5,
        MLP_RELEASE           = 4'd6,
        DONE                  = 4'd7
    } state_t;

    localparam logic signed [VALUES*ELEM_WIDTH-1:0] MLP_HIDDEN_FIXTURE = {
        16'sd1, 16'sd1, 16'sd4, 16'sd0
    };

    state_t state_r;
    logic done_r;
    logic axi_attention_start_r;
    logic mlp_start_r;
    logic axi_attention_done_w;
    logic mlp_done_w;
    logic [BLOCK_TRACE_WIDTH-1:0] block_trace_r;
    logic [ATTENTION_STATUS_WIDTH-1:0] attention_status_w;
    logic [ATTENTION_STATUS_WIDTH-1:0] captured_attention_status_r;
    logic [MLP_STATUS_WIDTH-1:0] mlp_status_w;
    logic [MLP_STATUS_WIDTH-1:0] captured_mlp_status_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] attention_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] captured_attention_r;
    logic signed [VALUES*ELEM_WIDTH-1:0] mlp_output_w;
    logic signed [VALUES*ELEM_WIDTH-1:0] final_output_r;
    logic [STATUS_WIDTH-1:0] status_r;
    logic [3:0] captured_axi_metadata_r;

    assign done_o = done_r;
    assign final_output_o = final_output_r;
    assign status_o = status_r;

    decoder_child_axi_attention_datapath u_decoder_child_axi_attention_datapath (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(axi_attention_start_r),
        .done_o(axi_attention_done_w),
        .output_o(attention_output_w),
        .status_o(attention_status_w)
    );

    residual_mlp_fixture u_residual_mlp_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(mlp_start_r),
        .hidden_input_i(MLP_HIDDEN_FIXTURE),
        .attention_output_i(captured_attention_r),
        .done_o(mlp_done_w),
        .final_output_o(mlp_output_w),
        .status_o(mlp_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            axi_attention_start_r <= 1'b0;
            mlp_start_r <= 1'b0;
            block_trace_r <= '0;
            captured_attention_status_r <= '0;
            captured_mlp_status_r <= '0;
            captured_attention_r <= '0;
            final_output_r <= '0;
            status_r <= '0;
            captured_axi_metadata_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    axi_attention_start_r <= 1'b0;
                    mlp_start_r <= 1'b0;
                    if (start_i) begin
                        block_trace_r <= '0;
                        captured_attention_status_r <= '0;
                        captured_mlp_status_r <= '0;
                        captured_attention_r <= '0;
                        final_output_r <= '0;
                        status_r <= '0;
                        captured_axi_metadata_r <= '0;
                        state_r <= AXI_ATTENTION_START;
                    end
                end
                AXI_ATTENTION_START: begin
                    axi_attention_start_r <= 1'b1;
                    block_trace_r[0*8 +: 8] <= 8'ha1;
                    state_r <= AXI_ATTENTION_BUSY;
                end
                AXI_ATTENTION_BUSY: begin
                    if (axi_attention_done_w) begin
                        axi_attention_start_r <= 1'b0;
                        block_trace_r[1*8 +: 8] <= 8'ha2;
                        captured_attention_r <= attention_output_w;
                        captured_attention_status_r <= attention_status_w;
                        captured_axi_metadata_r <= attention_status_w[63:60];
                        state_r <= AXI_ATTENTION_RELEASE;
                    end
                end
                AXI_ATTENTION_RELEASE: begin
                    if (!axi_attention_done_w) begin
                        state_r <= MLP_START;
                    end
                end
                MLP_START: begin
                    mlp_start_r <= 1'b1;
                    block_trace_r[2*8 +: 8] <= 8'hb1;
                    state_r <= MLP_BUSY;
                end
                MLP_BUSY: begin
                    if (mlp_done_w) begin
                        mlp_start_r <= 1'b0;
                        block_trace_r[3*8 +: 8] <= 8'hb2;
                        captured_mlp_status_r <= mlp_status_w;
                        final_output_r <= mlp_output_w;
                        status_r <= {mlp_status_w[79:0], 12'h000, captured_axi_metadata_r, captured_attention_status_r[47:0], 8'hb2, block_trace_r[23:0]};
                        state_r <= MLP_RELEASE;
                    end
                end
                MLP_RELEASE: begin
                    if (!mlp_done_w) begin
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        axi_attention_start_r <= 1'b0;
                        mlp_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    axi_attention_start_r <= 1'b0;
                    mlp_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
