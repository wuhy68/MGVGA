module module_107598(
    input clk,
    input rst_n,
    input [15:0] in,
    output [15:0] out
);

    // Fibonacci sequence generator
    reg [15:0] fib_reg [1:16];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            fib_reg[1] <= 16'h0000;
            fib_reg[2] <= 16'h0001;
            for (i = 3; i <= 16; i = i + 1) begin
                fib_reg[i] <= fib_reg[i-1] + fib_reg[i-2];
            end
        end else begin
            for (i = 2; i <= 16; i = i + 1) begin
                fib_reg[i] <= fib_reg[i-1];
            end
        end
    end
    
    // Binary-to-Gray code converter
    wire [15:0] gray_code;
    assign gray_code[0] = in[0];
    assign gray_code[1] = in[0] ^ in[1];
    assign gray_code[2] = in[1] ^ in[2];
    assign gray_code[3] = in[2] ^ in[3];
    assign gray_code[4] = in[3] ^ in[4];
    assign gray_code[5] = in[4] ^ in[5];
    assign gray_code[6] = in[5] ^ in[6];
    assign gray_code[7] = in[6] ^ in[7];
    assign gray_code[8] = in[7] ^ in[8];
    assign gray_code[9] = in[8] ^ in[9];
    assign gray_code[10] = in[9] ^ in[10];
    assign gray_code[11] = in[10] ^ in[11];
    assign gray_code[12] = in[11] ^ in[12];
    assign gray_code[13] = in[12] ^ in[13];
    assign gray_code[14] = in[13] ^ in[14];
    assign gray_code[15] = in[14] ^ in[15];
    
    // XOR module
    assign out = fib_reg[16] ^ gray_code;

endmodule