'timescale 1ns / 1ps 
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
    reg finish_pending;

    wire signed [15:0] product;

    assign product = input_value * weight_value;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 0;
            count <= 0;
            busy <= 0;
            done <= 0;
            finish_pending <= 0;
        end else begin
            done <= 0;
            if (start) begin
                acc <= bias; // Initialize accumulator with bias
                count <= 0;
                busy <= 1;
                finish_pending <= 0;
            end else if (busy && in_valid) begin
                acc <= acc + product; // Accumulate the product

                if (count == 783) begin // Assuming we want to process 784 inputs
                    busy <= 0;
                    done <= 1;
                end else begin
                    count <= count + 1;
                end
            end if (finish_pending) begin
                result         <= acc + bias;
                done           <= 1;
                finish_pending <= 0;
            end
        end
    end
endmodule