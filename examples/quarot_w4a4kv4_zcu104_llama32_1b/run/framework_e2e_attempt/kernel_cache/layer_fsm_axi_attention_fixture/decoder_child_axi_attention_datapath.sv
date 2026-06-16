`timescale 1ns/1ps

module decoder_child_axi_attention_datapath #(
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
        IDLE                 = 4'd0,
        SOURCE_PATH_START    = 4'd1,
        SOURCE_PATH_BUSY     = 4'd2,
        PROJECTION_AXI_START = 4'd3,
        PROJECTION_AXI_BUSY  = 4'd4,
        ATTENTION_KV_START   = 4'd5,
        ATTENTION_KV_BUSY    = 4'd6,
        DONE                 = 4'd7
    } state_t;

    state_t state_r;
    logic done_r;
    logic source_path_start_r;
    logic projection_axi_start_r;
    logic attention_kv_start_r;
    logic [TRACE_WIDTH-1:0] trace_r;
    logic signed [OUT_VALUES*OUT_WIDTH-1:0] output_r;
    logic [STATUS_WIDTH-1:0] status_r;
    logic [3:0] projection_axi_metadata_good_r;
    logic [3:0] projection_axi_status_low_r;

    logic source_path_done_w;
    logic signed [4*18-1:0] source_rms_output_w;
    logic signed [4*18-1:0] source_rope_output_w;
    logic [79:0] source_status_w;

    logic projection_axi_done_w;
    logic k_projection_axi_done_w;
    logic v_projection_axi_done_w;
    logic o_projection_axi_done_w;
    logic projection_axi_all_done_w;
    logic signed [2*32-1:0] projection_output_w;
    logic signed [2*32-1:0] k_projection_output_w;
    logic signed [2*32-1:0] v_projection_output_w;
    logic signed [2*32-1:0] o_projection_output_w;
    logic signed [2*32-1:0] projection_output_summary_w;
    logic [63:0] projection_status_w;
    logic [63:0] k_projection_status_w;
    logic [63:0] v_projection_status_w;
    logic [63:0] o_projection_status_w;
    logic [3:0] projection_axi_metadata_good_w;
    logic [3:0] projection_axi_status_low_w;

    logic attention_kv_done_w;
    logic signed [4*16-1:0] attention_output_w;
    logic [63:0] attention_status_w;

    assign done_o = done_r;
    assign output_o = output_r;
    assign status_o = status_r;
    assign projection_axi_all_done_w = projection_axi_done_w & k_projection_axi_done_w & v_projection_axi_done_w & o_projection_axi_done_w;
    assign projection_axi_metadata_good_w = projection_status_w[45:42] & k_projection_status_w[45:42] & v_projection_status_w[45:42] & o_projection_status_w[45:42];
    assign projection_axi_status_low_w = projection_status_w[3:0] & k_projection_status_w[3:0] & v_projection_status_w[3:0] & o_projection_status_w[3:0];
    assign projection_output_summary_w = projection_output_w ^ k_projection_output_w ^ v_projection_output_w ^ o_projection_output_w;

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

    projection_axi_stream_integration u_projection_axi_stream_integration (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(projection_axi_start_r),
        .done_o(projection_axi_done_w),
        .output_o(projection_output_w),
        .integration_status_o(projection_status_w)
    );

    projection_axi_stream_integration u_k_projection_axi_stream_integration (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(projection_axi_start_r),
        .done_o(k_projection_axi_done_w),
        .output_o(k_projection_output_w),
        .integration_status_o(k_projection_status_w)
    );

    projection_axi_stream_integration u_v_projection_axi_stream_integration (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(projection_axi_start_r),
        .done_o(v_projection_axi_done_w),
        .output_o(v_projection_output_w),
        .integration_status_o(v_projection_status_w)
    );

    projection_axi_stream_integration u_o_projection_axi_stream_integration (
        .aclk(aclk),
        .aresetn(aresetn),
        .start_i(projection_axi_start_r),
        .done_o(o_projection_axi_done_w),
        .output_o(o_projection_output_w),
        .integration_status_o(o_projection_status_w)
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
            projection_axi_start_r <= 1'b0;
            attention_kv_start_r <= 1'b0;
            trace_r <= '0;
            output_r <= '0;
            status_r <= '0;
            projection_axi_metadata_good_r <= '0;
            projection_axi_status_low_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    source_path_start_r <= 1'b0;
                    projection_axi_start_r <= 1'b0;
                    attention_kv_start_r <= 1'b0;
                    if (start_i) begin
                        trace_r <= '0;
                        output_r <= '0;
                        status_r <= '0;
                        projection_axi_metadata_good_r <= '0;
                        projection_axi_status_low_r <= '0;
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
                        state_r <= PROJECTION_AXI_START;
                    end
                end
                PROJECTION_AXI_START: begin
                    projection_axi_start_r <= 1'b1;
                    trace_r[2*8 +: 8] <= 8'h21;
                    state_r <= PROJECTION_AXI_BUSY;
                end
                PROJECTION_AXI_BUSY: begin
                    if (projection_axi_all_done_w) begin
                        projection_axi_start_r <= 1'b0;
                        projection_axi_metadata_good_r <= projection_axi_metadata_good_w;
                        projection_axi_status_low_r <= projection_axi_status_low_w;
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
                        status_r <= {projection_output_summary_w[15:0], 8'hac, attention_status_w[7:0], projection_axi_metadata_good_r, projection_axi_status_low_r, source_status_w[7:0], 8'h32, trace_r[39:0]};
                        done_r <= 1'b1;
                        state_r <= DONE;
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        source_path_start_r <= 1'b0;
                        projection_axi_start_r <= 1'b0;
                        attention_kv_start_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    source_path_start_r <= 1'b0;
                    projection_axi_start_r <= 1'b0;
                    attention_kv_start_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
