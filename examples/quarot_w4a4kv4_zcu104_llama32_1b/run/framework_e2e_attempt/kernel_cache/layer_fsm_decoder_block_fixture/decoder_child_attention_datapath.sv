`timescale 1ns/1ps

module decoder_child_attention_datapath #(
    parameter int TRACE_WIDTH = 48,
    parameter int OUT_VALUES = 4,
    parameter int OUT_WIDTH = 16,
    parameter int STATUS_WIDTH = 96
) (
    input  logic                                    aclk,
    input  logic                                    aresetn,
    input  logic                                    start_i,
    output logic                                    done_o,
    output logic signed [OUT_VALUES*OUT_WIDTH-1:0]  output_o,
    output logic [STATUS_WIDTH-1:0]                 status_o
);
    typedef enum logic [3:0] {
        IDLE                   = 4'd0,
        SOURCE_PATH_START      = 4'd1,
        SOURCE_PATH_BUSY       = 4'd2,
        PROJECTION_SHELL_START = 4'd3,
        PROJECTION_SHELL_BUSY  = 4'd4,
        ATTENTION_KV_START     = 4'd5,
        ATTENTION_KV_BUSY      = 4'd6,
        DONE                   = 4'd7
    } state_t;

    state_t state_r;
    logic done_r;
    logic source_path_start_r;
    logic projection_shell_start_r;
    logic attention_kv_start_r;
    logic [TRACE_WIDTH-1:0] trace_r;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] output_r;
    logic [STATUS_WIDTH-1:0] status_r;

    logic source_path_done_w;
    logic signed [4*18-1:0] source_rms_output_w;
    logic signed [4*18-1:0] source_rope_output_w;
    logic [79:0] source_status_w;

    logic projection_shell_done_w;
    logic signed [2*32-1:0] projection_output_w;
    logic [63:0] projection_status_w;

    logic attention_kv_done_w;
    logic signed [4*16-1:0] attention_output_w;
    logic [63:0] attention_status_w;

    assign done_o = done_r;
    assign output_o = output_r;
    assign status_o = status_r;

    rmsnorm_rope_source_path u_rmsnorm_rope_source_path (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(source_path_start_r),
        .norm_token_i(2'd0),
        .rope_position_i(4'd7),
        .done_o(source_path_done_w),
        .rms_output_o(source_rms_output_w),
        .rope_output_o(source_rope_output_w),
        .status_o(source_status_w)
    );

    projection_internal_stream_shell u_projection_internal_stream_shell (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(projection_shell_start_r),
        .done_o(projection_shell_done_w),
        .output_o(projection_output_w),
        .shell_status_o(projection_status_w)
    );

    attention_kv_cache_fixture u_attention_kv_cache_fixture (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(attention_kv_start_r),
        .done_o(attention_kv_done_w),
        .output_o(attention_output_w),
        .status_o(attention_status_w)
    );

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            done_r <= 1'b0;
            source_path_start_r <= 1'b0;
            projection_shell_start_r <= 1'b0;
            attention_kv_start_r <= 1'b0;
            trace_r <= '0;
            output_r <= '0;
            status_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    source_path_start_r <= 1'b0;
                    projection_shell_start_r <= 1'b0;
                    attention_kv_start_r <= 1'b0;
                    if (start_i) begin
                        trace_r <= '0;
                        output_r <= '0;
                        status_r <= '0;
                        state_r <= SOURCE_PATH_START;
                    end
                end
                SOURCE_PATH_START: begin
                    source_path_start_r <= 1'b1;
                    trace_r[0*8 +: 8] <= 8'h11;
                    state_r <= SOURCE_PATH_BUSY;
                end
                SOURCE_PATH_BUSY: begin
                    if (source_path_done_w) begin
                        source_path_start_r <= 1'b0;
                        trace_r[1*8 +: 8] <= 8'h12;
                        state_r <= PROJECTION_SHELL_START;
                    end
                end
                PROJECTION_SHELL_START: begin
                    projection_shell_start_r <= 1'b1;
                    trace_r[2*8 +: 8] <= 8'h21;
                    state_r <= PROJECTION_SHELL_BUSY;
                end
                PROJECTION_SHELL_BUSY: begin
                    if (projection_shell_done_w) begin
                        projection_shell_start_r <= 1'b0;
                        trace_r[3*8 +: 8] <= 8'h22;
                        state_r <= ATTENTION_KV_START;
                    end
                end
                ATTENTION_KV_START: begin
                    attention_kv_start_r <= 1'b1;
                    trace_r[4*8 +: 8] <= 8'h31;
                    state_r <= ATTENTION_KV_BUSY;
                end
                ATTENTION_KV_BUSY: begin
                    if (attention_kv_done_w) begin
                        attention_kv_start_r <= 1'b0;
                        trace_r[5*8 +: 8] <= 8'h32;
                        output_r <= attention_output_w;
                        status_r <= {16'h0000, 8'hac, attention_status_w[7:0], projection_status_w[7:0], source_status_w[7:0], 8'h32, trace_r[39:0]};
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        source_path_start_r <= 1'b0;
                        projection_shell_start_r <= 1'b0;
                        attention_kv_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    source_path_start_r <= 1'b0;
                    projection_shell_start_r <= 1'b0;
                    attention_kv_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
