`timescale 1ns / 1ps
module tb_top;
    reg clk;
    reg rst;
    reg start;
    reg in_valid;

    reg signed [7:0] input_value;
    reg signed [7:0] weight_value;
    reg signed [31:0] bias;

    wire busy;
    wire done;
    wire signed [31:0] result;

    top dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .in_valid(in_valid),
        .input_value(input_value),
        .weight_value(weight_value),
        .bias(bias),
        .busy(busy),
        .done(done),
        .result(result)
    );

    always #5 clk = ~clk; // Clock generation

    integer i;

    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        in_valid = 0;
        input_value = 0;
        weight_value = 0;
        bias = 10;

        #20;
        rst = 0;

        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        in_valid = 1;

        for (i = 0; i < 784; i = i + 1) begin
            input_value = 2; // Example input values
            weight_value = 3; // Example weight values
            @(negedge clk);
        end

        wait(done);
        $display("Result: %d", result);
        if (result == 4714) begin
            $display("Test passed!");
        end else begin
            $display("Test failed!");
        end
        $finish;
    end
    endmodule