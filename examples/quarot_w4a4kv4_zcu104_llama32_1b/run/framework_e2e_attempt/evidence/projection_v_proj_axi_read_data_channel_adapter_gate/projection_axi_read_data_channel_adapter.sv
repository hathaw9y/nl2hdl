`timescale 1ns/1ps

module projection_axi_read_data_channel_adapter #(
    parameter int MEM_DATA_WIDTH = 128,
    parameter int PAYLOAD_WIDTH = 32,
    parameter int FIXTURE_READ_BEATS = 2,
    parameter int PAYLOADS_PER_BEAT = 4,
    parameter int FIXTURE_PAYLOADS = 8,
    parameter logic [7:0] EXPECTED_AXI_ID = 8'h02
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    output logic                                      done_o,
    input  logic                                      axi_rvalid_i,
    output logic                                      axi_rready_o,
    input  logic [MEM_DATA_WIDTH-1:0]                 axi_rdata_i,
    input  logic [7:0]                                axi_rid_i,
    input  logic [1:0]                                axi_rresp_i,
    input  logic                                      axi_rlast_i,
    output logic                                      payload_valid_o,
    input  logic                                      payload_ready_i,
    output logic [PAYLOAD_WIDTH-1:0]                  payload_word_o,
    output logic                                      payload_last_o,
    output logic [FIXTURE_READ_BEATS-1:0]             accepted_beat_trace_o,
    output logic [FIXTURE_READ_BEATS-1:0]             rlast_trace_o,
    output logic [FIXTURE_READ_BEATS-1:0]             rid_error_trace_o,
    output logic [FIXTURE_READ_BEATS-1:0]             rresp_error_trace_o,
    output logic [FIXTURE_READ_BEATS-1:0]             rlast_error_trace_o,
    output logic [FIXTURE_PAYLOADS-1:0]               payload_trace_o,
    output logic [FIXTURE_PAYLOADS-1:0]               payload_ready_low_trace_o,
    output logic [15:0]                               status_o
);
    localparam int BEAT_IDX_W = (FIXTURE_READ_BEATS <= 1) ? 1 : $clog2(FIXTURE_READ_BEATS + 1);
    localparam int CHUNK_IDX_W = (PAYLOADS_PER_BEAT <= 1) ? 1 : $clog2(PAYLOADS_PER_BEAT);
    localparam int PAYLOAD_COUNT_W = (FIXTURE_PAYLOADS <= 1) ? 1 : $clog2(FIXTURE_PAYLOADS + 1);
    localparam logic [CHUNK_IDX_W-1:0] LAST_CHUNK = CHUNK_IDX_W'(PAYLOADS_PER_BEAT - 1);

    typedef enum logic [1:0] {
        IDLE   = 2'b00,
        R_WAIT = 2'b01,
        EMIT   = 2'b10,
        DONE   = 2'b11
    } state_t;

    state_t state_r;
    logic [MEM_DATA_WIDTH-1:0] mem_word_r;
    logic [BEAT_IDX_W-1:0] beat_count_r;
    logic [CHUNK_IDX_W-1:0] chunk_idx_r;
    logic [PAYLOAD_COUNT_W-1:0] payload_count_r;
    logic [PAYLOAD_WIDTH-1:0] payload_word_r;
    logic payload_valid_r;
    logic payload_last_r;
    logic done_r;
    logic current_beat_final_r;
    logic [FIXTURE_READ_BEATS-1:0] accepted_beat_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rlast_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rid_error_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rresp_error_trace_r;
    logic [FIXTURE_READ_BEATS-1:0] rlast_error_trace_r;
    logic [FIXTURE_PAYLOADS-1:0] payload_trace_r;
    logic [FIXTURE_PAYLOADS-1:0] payload_ready_low_trace_r;
    logic r_fire_w;
    logic payload_fire_w;
    logic [CHUNK_IDX_W-1:0] next_chunk_idx_w;
    logic last_chunk_w;
    logic final_beat_w;
    logic final_payload_w;
    logic expected_rlast_w;

    assign axi_rready_o = (state_r == R_WAIT) && (int'(beat_count_r) < FIXTURE_READ_BEATS);
    assign done_o = done_r;
    assign payload_valid_o = payload_valid_r;
    assign payload_word_o = payload_word_r;
    assign payload_last_o = payload_last_r;
    assign accepted_beat_trace_o = accepted_beat_trace_r;
    assign rlast_trace_o = rlast_trace_r;
    assign rid_error_trace_o = rid_error_trace_r;
    assign rresp_error_trace_o = rresp_error_trace_r;
    assign rlast_error_trace_o = rlast_error_trace_r;
    assign payload_trace_o = payload_trace_r;
    assign payload_ready_low_trace_o = payload_ready_low_trace_r;
    assign status_o = {
        2'b00,
        done_r,
        payload_valid_r,
        axi_rready_o,
        |rlast_error_trace_r,
        |rresp_error_trace_r,
        |rid_error_trace_r,
        2'(state_r),
        BEAT_IDX_W'(beat_count_r),
        PAYLOAD_COUNT_W'(payload_count_r)
    };

    assign r_fire_w = axi_rvalid_i && axi_rready_o;
    assign payload_fire_w = payload_valid_r && payload_ready_i;
    assign next_chunk_idx_w = chunk_idx_r + CHUNK_IDX_W'(1);
    assign last_chunk_w = (chunk_idx_r == LAST_CHUNK);
    assign final_beat_w = current_beat_final_r;
    assign final_payload_w = final_beat_w && last_chunk_w;
    assign expected_rlast_w = (int'(beat_count_r) == (FIXTURE_READ_BEATS - 1));

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= IDLE;
            mem_word_r <= '0;
            beat_count_r <= '0;
            chunk_idx_r <= '0;
            payload_count_r <= '0;
            payload_word_r <= '0;
            payload_valid_r <= 1'b0;
            payload_last_r <= 1'b0;
            done_r <= 1'b0;
            current_beat_final_r <= 1'b0;
            accepted_beat_trace_r <= '0;
            rlast_trace_r <= '0;
            rid_error_trace_r <= '0;
            rresp_error_trace_r <= '0;
            rlast_error_trace_r <= '0;
            payload_trace_r <= '0;
            payload_ready_low_trace_r <= '0;
        end else begin
            if (payload_valid_r && !payload_ready_i && int'(payload_count_r) < FIXTURE_PAYLOADS) begin
                payload_ready_low_trace_r[int'(payload_count_r)] <= 1'b1;
            end
            case (state_r)
                IDLE: begin
                    done_r <= 1'b0;
                    payload_valid_r <= 1'b0;
                    payload_last_r <= 1'b0;
                    if (start_i) begin
                        state_r <= R_WAIT;
                        mem_word_r <= '0;
                        beat_count_r <= '0;
                        chunk_idx_r <= '0;
                        payload_count_r <= '0;
                        payload_word_r <= '0;
                        current_beat_final_r <= 1'b0;
                        accepted_beat_trace_r <= '0;
                        rlast_trace_r <= '0;
                        rid_error_trace_r <= '0;
                        rresp_error_trace_r <= '0;
                        rlast_error_trace_r <= '0;
                        payload_trace_r <= '0;
                        payload_ready_low_trace_r <= '0;
                    end
                end
                R_WAIT: begin
                    payload_valid_r <= 1'b0;
                    payload_last_r <= 1'b0;
                    if (r_fire_w) begin
                        mem_word_r <= axi_rdata_i;
                        accepted_beat_trace_r[int'(beat_count_r)] <= 1'b1;
                        rlast_trace_r[int'(beat_count_r)] <= axi_rlast_i;
                        rid_error_trace_r[int'(beat_count_r)] <= (axi_rid_i != EXPECTED_AXI_ID);
                        rresp_error_trace_r[int'(beat_count_r)] <= (axi_rresp_i != 2'b00);
                        rlast_error_trace_r[int'(beat_count_r)] <= (axi_rlast_i != expected_rlast_w);
                        payload_word_r <= axi_rdata_i[0 +: PAYLOAD_WIDTH];
                        payload_valid_r <= 1'b1;
                        payload_last_r <= expected_rlast_w && (LAST_CHUNK == '0);
                        current_beat_final_r <= expected_rlast_w;
                        chunk_idx_r <= '0;
                        beat_count_r <= beat_count_r + BEAT_IDX_W'(1);
                        state_r <= EMIT;
                    end
                end
                EMIT: begin
                    if (payload_fire_w) begin
                        payload_trace_r[int'(payload_count_r)] <= 1'b1;
                        payload_count_r <= payload_count_r + PAYLOAD_COUNT_W'(1);
                        if (final_payload_w) begin
                            payload_valid_r <= 1'b0;
                            done_r <= 1'b1;
                            state_r <= DONE;
                        end else if (last_chunk_w) begin
                            payload_valid_r <= 1'b0;
                            payload_last_r <= 1'b0;
                            chunk_idx_r <= '0;
                            state_r <= R_WAIT;
                        end else begin
                            chunk_idx_r <= next_chunk_idx_w;
                            payload_word_r <= mem_word_r[int'(next_chunk_idx_w)*PAYLOAD_WIDTH +: PAYLOAD_WIDTH];
                            payload_last_r <= final_beat_w && (next_chunk_idx_w == LAST_CHUNK);
                        end
                    end
                end
                DONE: begin
                    if (!start_i) begin
                        state_r <= IDLE;
                        done_r <= 1'b0;
                        payload_valid_r <= 1'b0;
                        payload_last_r <= 1'b0;
                    end
                end
                default: begin
                    state_r <= IDLE;
                    done_r <= 1'b0;
                    payload_valid_r <= 1'b0;
                    payload_last_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
