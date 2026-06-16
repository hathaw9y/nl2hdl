`timescale 1ns/1ps

module w4a4_ws_systolic_tile #(
    parameter int ROWS     = 2,
    parameter int COLS     = 3,
    parameter int K        = 4,
    parameter int W_BITS   = 4,
    parameter int A_BITS   = 4,
    parameter int ACC_BITS = 32
) (
    input  logic                                      aclk,
    input  logic                                      aresetn,
    input  logic                                      start_i,
    input  logic [ROWS*K*A_BITS-1:0]                  activation_tile_i,
    input  logic [K*COLS*W_BITS-1:0]                  weight_tile_i,
    output logic                                      done_o,
    output logic signed [ROWS*COLS*ACC_BITS-1:0]      acc_tile_o,
    output logic [31:0]                               status_o
);
    localparam int K_IDX_W = (K <= 1) ? 1 : $clog2(K);

    typedef enum logic [1:0] {
        STATE_IDLE = 2'b00,
        STATE_RUN  = 2'b01,
        STATE_DONE = 2'b10
    } state_t;

    state_t state_r;
    logic [K_IDX_W-1:0] k_idx_r;
    logic [ROWS*K*A_BITS-1:0] activation_hold_r;
    logic [K*COLS*W_BITS-1:0] weight_hold_r;
    logic signed [ACC_BITS-1:0] acc_r [0:ROWS*COLS-1];
    logic weights_loaded_r;
    logic products_valid_r;

    assign done_o = (state_r == STATE_DONE);
    assign status_o = {
        8'(state_r),
        8'(ROWS * COLS),
        8'(K),
        4'(k_idx_r),
        done_o,
        weights_loaded_r,
        products_valid_r,
        (state_r == STATE_RUN)
    };

    generate
        for (genvar acc_idx = 0; acc_idx < ROWS*COLS; acc_idx++) begin : gen_acc_pack
            assign acc_tile_o[acc_idx*ACC_BITS +: ACC_BITS] = acc_r[acc_idx];
        end
    endgenerate

    function automatic logic signed [ACC_BITS-1:0] sign_extend_activation(
        input logic [A_BITS-1:0] element_i
    );
        begin
            sign_extend_activation = {{(ACC_BITS-A_BITS){element_i[A_BITS-1]}}, element_i};
        end
    endfunction

    function automatic logic signed [ACC_BITS-1:0] sign_extend_weight(
        input logic [W_BITS-1:0] element_i
    );
        begin
            sign_extend_weight = {{(ACC_BITS-W_BITS){element_i[W_BITS-1]}}, element_i};
        end
    endfunction

    function automatic logic signed [ACC_BITS-1:0] activation_at(
        input int row_idx_i,
        input int k_idx_i
    );
        int flat_idx;
        logic [A_BITS-1:0] packed_element;
        begin
            flat_idx = (row_idx_i * K) + k_idx_i;
            packed_element = activation_hold_r[flat_idx*A_BITS +: A_BITS];
            activation_at = sign_extend_activation(packed_element);
        end
    endfunction

    function automatic logic signed [ACC_BITS-1:0] weight_at(
        input int k_idx_i,
        input int col_idx_i
    );
        int flat_idx;
        logic [W_BITS-1:0] packed_element;
        begin
            flat_idx = (k_idx_i * COLS) + col_idx_i;
            packed_element = weight_hold_r[flat_idx*W_BITS +: W_BITS];
            weight_at = sign_extend_weight(packed_element);
        end
    endfunction

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state_r <= STATE_IDLE;
            k_idx_r <= '0;
            activation_hold_r <= '0;
            weight_hold_r <= '0;
            weights_loaded_r <= 1'b0;
            products_valid_r <= 1'b0;
            for (int acc_idx = 0; acc_idx < ROWS*COLS; acc_idx++) begin
                acc_r[acc_idx] <= '0;
            end
        end else begin
            case (state_r)
                STATE_IDLE: begin
                    k_idx_r <= '0;
                    weights_loaded_r <= 1'b0;
                    products_valid_r <= 1'b0;

                    if (start_i) begin
                        state_r <= STATE_RUN;
                        activation_hold_r <= activation_tile_i;
                        weight_hold_r <= weight_tile_i;
                        weights_loaded_r <= 1'b1;
                        for (int acc_idx = 0; acc_idx < ROWS*COLS; acc_idx++) begin
                            acc_r[acc_idx] <= '0;
                        end
                    end
                end

                STATE_RUN: begin
                    products_valid_r <= 1'b1;
                    weights_loaded_r <= 1'b1;
                    for (int row_idx = 0; row_idx < ROWS; row_idx++) begin
                        for (int col_idx = 0; col_idx < COLS; col_idx++) begin
                            acc_r[(row_idx*COLS)+col_idx] <=
                                acc_r[(row_idx*COLS)+col_idx] +
                                (activation_at(row_idx, int'(k_idx_r)) * weight_at(int'(k_idx_r), col_idx));
                        end
                    end

                    if (int'(k_idx_r) == (K - 1)) begin
                        state_r <= STATE_DONE;
                    end else begin
                        k_idx_r <= k_idx_r + K_IDX_W'(1);
                    end
                end

                STATE_DONE: begin
                    weights_loaded_r <= 1'b1;

                    if (!start_i) begin
                        state_r <= STATE_IDLE;
                        k_idx_r <= '0;
                        weights_loaded_r <= 1'b0;
                        products_valid_r <= 1'b0;
                    end
                end

                default: begin
                    state_r <= STATE_IDLE;
                    k_idx_r <= '0;
                    weights_loaded_r <= 1'b0;
                    products_valid_r <= 1'b0;
                end
            endcase
        end
    end
endmodule
