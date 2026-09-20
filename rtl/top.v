module top (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,
    input  wire                    in_valid,

    input  wire signed [7:0]       input_value,
    input  wire signed [7:0]       weight_value,
    input  wire signed [31:0]      bias,

    output reg                     busy,
    output reg                     done,
    output wire signed [31:0]      result
);

    reg [9:0] count;

    wire mac_enable;
    wire signed [31:0] mac_result;

    assign mac_enable = busy && in_valid;
    assign result = mac_result;

    mac mac_unit (
        .clk          (clk),
        .rst          (rst),
        .clear        (start),
        .enable       (mac_enable),
        .input_value  (input_value),
        .weight_value (weight_value),
        .bias         (bias),
        .result       (mac_result)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 10'd0;
            busy  <= 1'b0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start) begin
                count <= 10'd0;
                busy  <= 1'b1;
            end else if (busy && in_valid) begin
                if (count == 10'd783) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    count <= count + 10'd1;
                end
            end
        end
    end

endmodule