`timescale 1ns / 1ps

module tb_mac;

    reg clk;
    reg rst;
    reg clear;
    reg enable;

    reg signed [7:0] input_value;
    reg signed [7:0] weight_value;
    reg signed [31:0] bias;

    wire signed [31:0] result;

    mac dut (
        .clk          (clk),
        .rst          (rst),
        .clear        (clear),
        .enable       (enable),
        .input_value  (input_value),
        .weight_value (weight_value),
        .bias         (bias),
        .result       (result)
    );

    always #5 clk = ~clk;

    task accumulate;
        input signed [7:0] test_input;
        input signed [7:0] test_weight;

        begin
            // Change inputs away from the active rising edge
            @(negedge clk);

            input_value  = test_input;
            weight_value = test_weight;
            enable       = 1'b1;

            // MAC captures the values here
            @(posedge clk);
            #1;

            enable = 1'b0;

            $display(
                "input=%0d weight=%0d result=%0d",
                test_input,
                test_weight,
                result
            );
        end
    endtask

    initial begin
        clk          = 1'b0;
        rst          = 1'b1;
        clear        = 1'b0;
        enable       = 1'b0;
        input_value  = 8'sd0;
        weight_value = 8'sd0;
        bias         = 32'sd10;

        // Keep reset active for two clock edges
        repeat (2) @(posedge clk);

        // Release reset on the falling edge
        @(negedge clk);
        rst = 1'b0;

        // Load bias into the accumulator
        clear = 1'b1;

        @(posedge clk);
        #1;
        clear = 1'b0;

        $display("Bias loaded: result=%0d", result);

        // result = 10 + (2 * 3) = 16
        accumulate(8'sd2, 8'sd3);

        // result = 16 + (-4 * 5) = -4
        accumulate(-8'sd4, 8'sd5);

        // result = -4 + (-3 * -2) = 2
        accumulate(-8'sd3, -8'sd2);

        if (result == 32'sd2) begin
            $display("MAC TEST PASSED");
            $display("Final result: %0d", result);
        end else begin
            $display("MAC TEST FAILED");
            $display("Expected: 2");
            $display("Received: %0d", result);
        end

        $finish;
    end

endmodule