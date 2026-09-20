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

    integer i;
    reg signed [31:0] expected_result;

    top dut (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .in_valid     (in_valid),
        .input_value  (input_value),
        .weight_value (weight_value),
        .bias         (bias),
        .busy         (busy),
        .done         (done),
        .result       (result)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        clk             = 1'b0;
        rst             = 1'b1;
        start           = 1'b0;
        in_valid        = 1'b0;
        input_value     = 8'sd0;
        weight_value    = 8'sd0;
        bias            = 32'sd10;
        expected_result = 32'sd4714;

        // Reset
        #20;
        rst = 1'b0;

        // Start the computation
        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start    = 1'b0;
        in_valid = 1'b1;

        // Send 784 input-weight pairs
        for (i = 0; i < 784; i = i + 1) begin
            input_value  = 8'sd2;
            weight_value = 8'sd3;
            @(negedge clk);
        end

        in_valid = 1'b0;

        // Wait indefinitely for the hardware result
        wait(done == 1'b1);

        if (result == expected_result) begin
            $display("TOP TEST PASSED");
            $display("Result: %0d", result);
        end else begin
            $display("TOP TEST FAILED");
            $display("Expected: %0d", expected_result);
            $display("Received: %0d", result);
        end

        $finish;
    end

endmodule