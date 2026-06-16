`timescale 1ns/1ps

// Bounded QuaRot W4A4KV4 support fixture.
// This is not a full QuaRot or LLaMA target implementation.
module quarot_w4a4kv4_support #(
    parameter int N         = 4,
    parameter int I8_WIDTH  = 8,
    parameter int A4_WIDTH  = 4,
    parameter int ROT_WIDTH = 10
) (
    input  logic                         aclk,
    input  logic                         aresetn,
    input  logic                         start_i,
    input  logic signed [N*I8_WIDTH-1:0] input_vec_i,
    input  logic        [N*A4_WIDTH-1:0] kv_vec0_i,
    input  logic        [N*A4_WIDTH-1:0] kv_vec1_i,
    output logic                         done_o,
    output logic signed [N*ROT_WIDTH-1:0] rotated_vec_o,
    output logic        [N*A4_WIDTH-1:0] a4_packed_o,
    output logic        [N*A4_WIDTH-1:0] kv_vec0_roundtrip_o,
    output logic        [N*A4_WIDTH-1:0] kv_vec1_roundtrip_o,
    output logic        [7:0]            status_o
);

    typedef enum logic [0:0] {
        STATE_IDLE = 1'b0,
        STATE_DONE = 1'b1
    } state_t;

    localparam logic signed [ROT_WIDTH-1:0] A4_MIN_W = -10'sd8;
    localparam logic signed [ROT_WIDTH-1:0] A4_MAX_W =  10'sd7;

    state_t state_r;
    state_t state_n;

    logic signed [I8_WIDTH-1:0]  input_elem_w [N];
    logic signed [ROT_WIDTH-1:0] input_ext_w [N];
    logic signed [ROT_WIDTH-1:0] rot_elem_w [N];
    logic signed [A4_WIDTH-1:0]  a4_elem_w [N];
    logic        [N*ROT_WIDTH-1:0] rotated_vec_w;
    logic        [N*A4_WIDTH-1:0]  a4_packed_w;
    logic        [N*A4_WIDTH-1:0]  kv_vec0_roundtrip_w;
    logic        [N*A4_WIDTH-1:0]  kv_vec1_roundtrip_w;
    logic                          sat_hi_seen_w;
    logic                          sat_lo_seen_w;
    logic                          neg_seen_w;
    logic                          kv_roundtrip_match_w;
    logic        [7:0]             status_w;

    function automatic logic signed [A4_WIDTH-1:0] saturate_to_a4(
        input logic signed [ROT_WIDTH-1:0] value_i
    );
        begin
            if (value_i > A4_MAX_W) begin
                saturate_to_a4 = 4'sd7;
            end else if (value_i < A4_MIN_W) begin
                saturate_to_a4 = -4'sd8;
            end else begin
                saturate_to_a4 = value_i[A4_WIDTH-1:0];
            end
        end
    endfunction

    assign input_elem_w[0] = $signed(input_vec_i[0*I8_WIDTH +: I8_WIDTH]);
    assign input_elem_w[1] = $signed(input_vec_i[1*I8_WIDTH +: I8_WIDTH]);
    assign input_elem_w[2] = $signed(input_vec_i[2*I8_WIDTH +: I8_WIDTH]);
    assign input_elem_w[3] = $signed(input_vec_i[3*I8_WIDTH +: I8_WIDTH]);

    assign input_ext_w[0] = {{(ROT_WIDTH-I8_WIDTH){input_elem_w[0][I8_WIDTH-1]}}, input_elem_w[0]};
    assign input_ext_w[1] = {{(ROT_WIDTH-I8_WIDTH){input_elem_w[1][I8_WIDTH-1]}}, input_elem_w[1]};
    assign input_ext_w[2] = {{(ROT_WIDTH-I8_WIDTH){input_elem_w[2][I8_WIDTH-1]}}, input_elem_w[2]};
    assign input_ext_w[3] = {{(ROT_WIDTH-I8_WIDTH){input_elem_w[3][I8_WIDTH-1]}}, input_elem_w[3]};

    always_comb begin
        state_n = state_r;

        case (state_r)
            STATE_IDLE: begin
                if (start_i) begin
                    state_n = STATE_DONE;
                end
            end

            STATE_DONE: begin
                if (!start_i) begin
                    state_n = STATE_IDLE;
                end
            end

            default: begin
                state_n = STATE_IDLE;
            end
        endcase
    end

    // Unnormalized H4 fixture rotation, element order x0..x3.
    assign rot_elem_w[0] = input_ext_w[0] + input_ext_w[1] + input_ext_w[2] + input_ext_w[3];
    assign rot_elem_w[1] = input_ext_w[0] - input_ext_w[1] + input_ext_w[2] - input_ext_w[3];
    assign rot_elem_w[2] = input_ext_w[0] + input_ext_w[1] - input_ext_w[2] - input_ext_w[3];
    assign rot_elem_w[3] = input_ext_w[0] - input_ext_w[1] - input_ext_w[2] + input_ext_w[3];

    assign a4_elem_w[0] = saturate_to_a4(rot_elem_w[0]);
    assign a4_elem_w[1] = saturate_to_a4(rot_elem_w[1]);
    assign a4_elem_w[2] = saturate_to_a4(rot_elem_w[2]);
    assign a4_elem_w[3] = saturate_to_a4(rot_elem_w[3]);

    assign rotated_vec_w = {rot_elem_w[3], rot_elem_w[2], rot_elem_w[1], rot_elem_w[0]};
    assign a4_packed_w = {a4_elem_w[3], a4_elem_w[2], a4_elem_w[1], a4_elem_w[0]};
    assign kv_vec0_roundtrip_w = kv_vec0_i;
    assign kv_vec1_roundtrip_w = kv_vec1_i;
    assign sat_hi_seen_w = (rot_elem_w[0] > A4_MAX_W) | (rot_elem_w[1] > A4_MAX_W)
                         | (rot_elem_w[2] > A4_MAX_W) | (rot_elem_w[3] > A4_MAX_W);
    assign sat_lo_seen_w = (rot_elem_w[0] < A4_MIN_W) | (rot_elem_w[1] < A4_MIN_W)
                         | (rot_elem_w[2] < A4_MIN_W) | (rot_elem_w[3] < A4_MIN_W);
    assign neg_seen_w = input_elem_w[0][I8_WIDTH-1] | input_elem_w[1][I8_WIDTH-1]
                      | input_elem_w[2][I8_WIDTH-1] | input_elem_w[3][I8_WIDTH-1];
    assign kv_roundtrip_match_w = (kv_vec0_roundtrip_w == kv_vec0_i)
                                & (kv_vec1_roundtrip_w == kv_vec1_i);
    assign status_w = {
        3'b000,
        neg_seen_w,
        kv_roundtrip_match_w,
        sat_lo_seen_w,
        sat_hi_seen_w,
        1'b1
    };

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= STATE_IDLE;
            done_o <= 1'b0;
            rotated_vec_o <= '0;
            a4_packed_o <= '0;
            kv_vec0_roundtrip_o <= '0;
            kv_vec1_roundtrip_o <= '0;
            status_o <= '0;
        end else begin
            state_r <= state_n;
            done_o <= (state_n == STATE_DONE);

            if ((state_r == STATE_IDLE) && start_i) begin
                rotated_vec_o <= rotated_vec_w;
                a4_packed_o <= a4_packed_w;
                kv_vec0_roundtrip_o <= kv_vec0_roundtrip_w;
                kv_vec1_roundtrip_o <= kv_vec1_roundtrip_w;
                status_o <= status_w;
            end
        end
    end

endmodule
