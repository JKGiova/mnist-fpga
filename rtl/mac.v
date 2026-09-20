module mac(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire enable,

    input wire signed [7:0] input_value,
    input wire signed [7:0] weight_value,
    input wire signed [31:0] bias,

    output reg signed [31:0] result
    );

    wire signed [15:0] product;
    wire signed [31:0] product_extended;

    assign product = input_value * weight_value;

    assign product_extended = {
        {16{product[15]}}, 
        product
    };

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result <= 32'sd0;
        end else if (clear) begin
            result <= bias;
        end else if (enable) begin
            result <= result + product_extended;
        end
    end
endmodule