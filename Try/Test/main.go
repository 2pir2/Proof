package main

import (
	"fmt"
)

// Define the simplified model with inputs, weights, and biases
type ProveModelCircuit struct {
	Weights  [2][4][4]float64 // 2 layers, each with 4 neurons, each neuron has 4 inputs
	Biases   [2][4]float64    // 2 layers, each with 4 biases (one for each neuron)
	X        [3][4]float64    // Array of 3 input vectors (each 4 values)
	Expected [3]int           // Array of 3 expected outputs (index of max value)
}

// ReLU activation function to replace negative values with 0
func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Function to simulate the neural network and print outputs for debugging
func ComputeOutputs(circuit *ProveModelCircuit) {
	// Loop through each input-output pair (3 input-output pairs in total)
	for k := 0; k < 3; k++ {
		outputLayer := make([][]float64, 2) // Output for each layer (2 layers)
		outputLayer[0] = make([]float64, 4) // First layer outputs (4 neurons)
		outputLayer[1] = make([]float64, 4) // Second layer outputs (4 neurons)

		// Combine the loops for both Layer 1 and Layer 2
		for layer := 0; layer < 2; layer++ {
			for i := 0; i < 4; i++ {
				// Compute the sum of weights * inputs for each neuron in the current layer
				if layer == 0 {
					// Layer 1: Compute using inputs
					outputLayer[layer][i] = circuit.Biases[layer][i]
					for j := 0; j < 4; j++ {
						outputLayer[layer][i] += circuit.Weights[layer][i][j] * circuit.X[k][j]
					}
				} else {
					// Layer 2: Compute using outputs from the previous layer
					outputLayer[layer][i] = circuit.Biases[layer][i]
					for j := 0; j < 4; j++ {
						outputLayer[layer][i] += circuit.Weights[layer][i][j] * outputLayer[layer-1][j]
					}
				}
				// Apply ReLU activation (Replace negative values with 0)
				outputLayer[layer][i] = ReLU(outputLayer[layer][i])
			}
		}

		// Debug: Print intermediate outputs for Layer 1 and Layer 2
		fmt.Printf("Input %d: Layer 1 Output: %v\n", k+1, outputLayer[0])
		fmt.Printf("Input %d: Layer 2 Output: %v\n", k+1, outputLayer[1])

		// Find the maximum value (argmax) in the second layer output
		maxVal := outputLayer[1][0]
		maxIdx := 0
		for i := 1; i < 4; i++ {
			if outputLayer[1][i] > maxVal {
				maxVal = outputLayer[1][i]
				maxIdx = i
			}
		}

		// Debug: Print the argmax and expected output
		fmt.Printf("Input %d: Argmax (Predicted): %d, Expected: %d\n", k+1, maxIdx, circuit.Expected[k])

		// Check if the predicted output matches the expected output
		if maxIdx != circuit.Expected[k] {
			fmt.Printf("Mismatch for Input %d: Expected %d but got %d\n", k+1, circuit.Expected[k], maxIdx)
		} else {
			fmt.Printf("Input %d: Correct prediction!\n", k+1)
		}
	}
}

func main() {
	// Example circuit data
	circuit := &ProveModelCircuit{
		Weights: [2][4][4]float64{
			{
				{0, 0, 0, 1},
				{0, 0, 1, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
			},
			{
				{0, 0, 0, 1},
				{0, 0, 1, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
			},
		},
		Biases: [2][4]float64{
			{0, 0, 0, 0},
			{0, 0, 0, 0},
		},
		X: [3][4]float64{
			{1.0, 0.5, -0.3, 0.2},
			{0.6, -0.2, 0.8, -0.1},
			{0.2, 0.1, -0.5, 0.4},
		},
		Expected: [3]int{0, 2, 3},
	}

	// Compute and debug the outputs
	ComputeOutputs(circuit)
}
