<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
![Verilog-2001][verilog-shield]
![Status: Prototype][status-shield]

<div align="center">
  <h1>MNIST FPGA</h1>

  <p>
    Building a quantized handwritten-digit classifier in plain Verilog,
    one hardware block at a time.
    <br />
    <a href="#getting-started"><strong>Run the simulations »</strong></a>
    <br /><br />
    <a href="#architecture">Explore the Architecture</a>
    &middot;
    <a href="https://github.com/JKGiova/mnist-fpga/issues">Report a Bug</a>
    &middot;
    <a href="https://github.com/JKGiova/mnist-fpga/issues">Request a Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the Project</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the Project

MNIST FPGA is an experimental hardware project aimed at implementing a
small neural network for handwritten-digit recognition directly on an FPGA.

The target model is a fully connected **784 → 16 → 10 multilayer
perceptron (MLP)**:

- 784 inputs from a flattened 28 × 28 image.
- 16 hidden neurons with ReLU activation.
- 10 output scores, one for each digit from 0 to 9.
- An argmax operation to select the predicted digit.

The project explores the hardware behind neural-network inference:
signed integer arithmetic, multiply-accumulate units, quantization,
on-chip memory, and parallel execution.

The longer-term goal is to investigate SRAM/BRAM-based accelerator
architectures and build experience relevant to a future ASIC design.

> **Current status:** this repository implements a reusable MAC and a
> controller that processes 784 input–weight pairs. The complete MNIST
> classifier is not implemented yet. Trained weights, weight memories,
> activation functions, and board communication interfaces are planned.

### Built With

- **Verilog-2001** — hardware modules and testbenches.
- **Icarus Verilog** — compilation and simulation using `iverilog` and `vvp`.
- **Git** — version control.

The current RTL uses no SystemVerilog or board-specific IP.
FPGA synthesis, timing closure, and deployment remain future work.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

### Current Implementation

The current design computes one neuron's weighted sum:

```math
y = b + \sum_{i=0}^{783} x_i w_i
```

| Component | Function |
| :--- | :--- |
| `mac.v` | Multiplies two signed 8-bit operands and accumulates their product into a signed 32-bit register. |
| `top.v` | Starts the operation, counts 784 accepted pairs, and generates `busy` and `done`. |
| `tb_mac.v` | Tests signed arithmetic, including negative operands. |
| `tb_top.v` | Streams 784 pairs and checks the final result. |

The MAC has the following control priority:

1. `rst`: reset the accumulator to zero.
2. `clear`: load the bias into the accumulator.
3. `enable`: add the current product.
4. Otherwise, hold the current value.

### Numeric Formats

| Quantity | Current format |
| :--- | :--- |
| Input | Signed INT8 |
| Weight | Signed INT8 |
| Product | Signed INT16 |
| Bias | Signed INT32 |
| Accumulator / result | Signed INT32 |

The product is explicitly sign-extended from 16 to 32 bits before
accumulation.

Arithmetic is not saturating. If the accumulator exceeds the signed
32-bit range, it wraps. Model export and validation must account for
the supported numeric ranges.

### Planned Full Network

```math
h = \operatorname{ReLU}(W_1x + b_1)
```

```math
z = W_2h + b_2
```

```math
\hat{y} = \operatorname{argmax}(z)
```

The integer hardware implementation will also requantize hidden
activations between layers. Softmax is unnecessary when only the
predicted class is required.

| Layer | Dimensions | Weights / MAC operations per image |
| :--- | :--- | ---: |
| Hidden layer | 784 → 16 | 12,544 |
| Output layer | 16 → 10 | 160 |
| Total | | 12,704 |

INT8 weights alone require **12,704 bytes**, approximately **12.4 KiB**,
excluding biases and quantization metadata.

The proposed baseline uses 16 parallel MAC units for the hidden layer:
each receives the same input but a different weight. The compute units
can then be reused for the output layer.

Weight-memory layout, read latency, and bandwidth must be designed
alongside the compute units. Quantization scales, rounding, and
saturation must match the software reference.

No MNIST accuracy, FPGA resource usage, maximum clock frequency, or
inference-throughput measurements are reported yet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure

| Path | Description |
| :--- | :--- |
| `rtl/mac.v` | Reusable signed multiply-accumulate unit. |
| `rtl/top.v` | Controller for a 784-term weighted sum. |
| `sim/tb_mac.v` | Standalone MAC testbench. |
| `sim/tb_top.v` | Controller and MAC integration testbench. |
| `.gitignore` | Excludes `mac_test.out` and `top_test.out`. |
| `README.md` | Project overview and simulation instructions. |

## Getting Started

### Prerequisites

Install:

- Git.
- Icarus Verilog, including `iverilog` and `vvp`.

Make sure these commands are available in your terminal.
No FPGA board is required to run the simulations.

### Installation

Clone the repository:

```sh
git clone https://github.com/JKGiova/mnist-fpga.git
cd mnist-fpga
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Run the following commands from the repository root.

### Run the MAC Test

```sh
iverilog -g2001 -s tb_mac -o mac_test.out rtl/mac.v sim/tb_mac.v
vvp mac_test.out
```

The test evaluates:

```text
10 + (2 × 3) + (-4 × 5) + (-3 × -2) = 2
```

Expected application output:

```text
Bias loaded: result=10
input=2 weight=3 result=16
input=-4 weight=5 result=-4
input=-3 weight=-2 result=2
MAC TEST PASSED
Final result: 2
```

### Run the Controller Test

```sh
iverilog -g2001 -s tb_top -o top_test.out rtl/mac.v rtl/top.v sim/tb_top.v
vvp top_test.out
```

The test supplies 784 pairs with input `2`, weight `3`, and bias `10`:

```text
784 × 2 × 3 + 10 = 4714
```

Expected application output:

```text
TOP TEST PASSED
Result: 4714
```

The simulator may also print a termination message from `$finish`.

The `-g2001` option explicitly selects Verilog-2001 mode.
The `-s` option selects the testbench to run.

> These are basic functional tests, not exhaustive verification.
> Check the printed pass/fail messages: the testbenches do not explicitly
> return a failing process exit code on a comparison failure.
> The controller test deliberately waits for `done` without a timeout.

### Top-Level Interface

| Signal | Direction | Width | Description |
| :--- | :--- | :--- | :--- |
| `clk` | Input | 1 bit | Rising-edge clock. |
| `rst` | Input | 1 bit | Active-high asynchronous reset. |
| `start` | Input | 1 bit | Loads the bias and starts or restarts an operation. |
| `in_valid` | Input | 1 bit | Marks the current input–weight pair as valid. |
| `input_value` | Input | 8 bits, signed | Input operand. |
| `weight_value` | Input | 8 bits, signed | Weight operand. |
| `bias` | Input | 32 bits, signed | Bias sampled when `start` is asserted. |
| `busy` | Output | 1 bit | An accumulation is active. |
| `done` | Output | 1 bit | One-clock completion pulse. |
| `result` | Output | 32 bits, signed | Running accumulator; final value when `done` is high. |

### Transaction Sequence

1. Apply reset.
2. Set `bias` and assert `start` for one clock.
3. Deassert `start`, then supply 784 input–weight pairs.
4. Assert `in_valid` on each cycle containing a valid pair.
5. Sample the final result when `done` is high.

With reset inactive, a pair is accepted on a rising edge when:

```verilog
busy && in_valid && !start
```

Gaps in `in_valid` pause both accumulation and counting.

A new `start` discards any in-progress computation and reloads the bias.
No pair is consumed on that edge.

After the 784th accepted pair:

- The accumulator contains the final result.
- `busy` is cleared.
- `done` is asserted for one clock.

The result remains available until reset or the next operation.
In simulation, sample after the nonblocking assignments have settled.

The current interface consists of synchronous parallel signals.
It is not UART, Ethernet, or another external communication protocol.

The testbenches generate a 100 MHz simulated clock. This is not a
measured FPGA operating frequency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Implement a signed INT8 × INT8 MAC with an INT32 accumulator.
- [x] Separate the MAC from the 784-pair controller.
- [x] Add standalone and integration testbenches.
- [ ] Extend tests with boundary values, input stalls, reset, and repeated operations.
- [ ] Train the 784 → 16 → 10 MLP.
- [ ] Define a bit-accurate integer software reference.
- [ ] Export weights, biases, and quantization parameters.
- [ ] Add BRAM-compatible weight storage and input buffering.
- [ ] Implement the hidden layer with 16 parallel MAC units.
- [ ] Add ReLU and requantization.
- [ ] Implement the output layer and argmax.
- [ ] Validate MNIST predictions against the software reference.
- [ ] Synthesize for a selected FPGA and measure resource use and timing.
- [ ] Add host communication, initially UART.
- [ ] Explore Gigabit Ethernet and continuous inference.
- [ ] Investigate greater parallelism and memory-bandwidth trade-offs.

## Contributing

Suggestions and proposed changes can be submitted through
[GitHub Issues][issues-url] and pull requests.

For RTL changes:

- Keep the code compatible with Verilog-2001.
- Include a relevant testbench.
- Run both existing simulations.
- Explain any changes to signal timing, signedness, or arithmetic behavior.
- Keep board-specific integration separate from the core computation.

## License

No project license file is currently included in this repository.

## Contact

Maintainer: [@JKGiova](https://github.com/JKGiova)

Project: [JKGiova/mnist-fpga](https://github.com/JKGiova/mnist-fpga)

For project questions, use [GitHub Issues][issues-url].

## Acknowledgments

- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
  by othneildrew — README structure and presentation.
- [Icarus Verilog](https://github.com/steveicarus/iverilog)
  — compilation and simulation tooling.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/JKGiova/mnist-fpga.svg?style=for-the-badge
[contributors-url]: https://github.com/JKGiova/mnist-fpga/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/JKGiova/mnist-fpga.svg?style=for-the-badge
[stars-url]: https://github.com/JKGiova/mnist-fpga/stargazers
[issues-shield]: https://img.shields.io/github/issues/JKGiova/mnist-fpga.svg?style=for-the-badge
[issues-url]: https://github.com/JKGiova/mnist-fpga/issues
[verilog-shield]: https://img.shields.io/badge/Verilog-2001-2563eb?style=for-the-badge
[status-shield]: https://img.shields.io/badge/Status-Prototype-f59e0b?style=for-the-badge