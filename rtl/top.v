module top(
    input wire clk,
    input wire rst,
    input wire start,
    input wire in_valid,
    input wire signed [7:0] input_value,
    input wire signed [7:0] weight_value,
    input wire signed [31:0] bias,

    output reg busy,
    output reg done,
    output reg signed [31:0] result
    );
    reg signed [31:0] acc;
    reg [9:0] count;

    wire signed [15:0] product;

    assign product = input_value * weight_value;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 0;
            count <= 0;
            busy <= 0;
            done <= 0;
        end else begin
            done <= 0;
            if (start) begin
                acc <= 0; // Initialize accumulator with 0s
                count <= 0;
                busy <= 1;
            end else if (busy && in_valid) begin
                acc <= acc + product; // Accumulate the product

                if (count == 783) begin // Assuming we want to process 784 inputs
                    result <= acc + product + bias; // Add bias to the accumulated result
                    busy <= 0;
                    done <= 1;
                end else begin
                    count <= count + 1;
                end
            end
        end
    end
endmodule