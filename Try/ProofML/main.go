package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/cmp"
)

// ProveModelCircuit defines the circuit structure
type ProveModelCircuit struct {
	A        [][][]frontend.Variable `gnark:",secret"` // Weights (m x n)
	Biases   [][]frontend.Variable    `gnark:",secret"` // Biases (m x 1)
	X        []frontend.Variable      `gnark:",public"` // Input vector (n)
	Expected []frontend.Variable      `gnark:",public"` // Expected outputs (slice)
}

// Define defines the circuit's constraints
func (circuit *ProveModelCircuit) Define(api frontend.API) error {
	// Number of neurons
	numberNeurons := len(circuit.A[0])

	// Initialize the output variable
	output := make([]frontend.Variable, numberNeurons)

	// Calculate the output for each neuron
	for i := 0; i < numberNeurons; i++ {
		output[i] = circuit.Biases[0][i]
		for j := 0; j < len(circuit.X); j++ {
			output[i] = api.Add(output[i], api.Mul(circuit.A[0][i][j], circuit.X[j]))
		}
		// Apply ReLU activation
		output[i] = api.Select(cmp.IsLess(api, output[i], 0), 0, output[i])
	}

	// Find the maximum value (argmax) in the output layer
	maxVal := output[0]
	maxIdx := frontend.Variable(0)
	for i := 1; i < numberNeurons; i++ {
		isLess := cmp.IsLess(api, maxVal, output[i])
		maxVal = api.Select(isLess, output[i], maxVal)
		maxIdx = api.Select(isLess, i, maxIdx)
	}

	// Assert that the predicted class matches the expected output
	api.AssertIsEqual(maxIdx, circuit.Expected)

	return nil
}

func readJSONFile(filename string, target interface{}) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("error opening file %s: %v", filename, err)
	}
	defer file.Close()

	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return fmt.Errorf("error reading file %s: %v", filename, err)
	}

	if err := json.Unmarshal(bytes, target); err != nil {
		return fmt.Errorf("error unmarshaling JSON from file %s: %v", filename, err)
	}

	return nil
}

func main() {
	var weightsData struct {
		Weights [][][]float64 `json:"weights"`
		Biases [][]float64 `json:"biases"`
	}
	
	var expectedData struct {
		Expected []float64 `json:"outputs"`
	}

	// Read JSON files
	if err := readJSONFile("weights.json", &weightsData); err != nil {
		fmt.Println(err)
		return
	}

	if err := readJSONFile("outputs.json", &expectedData); err != nil{
		fmt.Println(err)
		return
	}
	

	// Convert JSON data to frontend.Variable
	A := make([][][]frontend.Variable, len(weightsData.Weights))
	for i, layer := range weightsData.Weights {
		A[i] = make([][]frontend.Variable, len(layer))
		for j, neuron := range layer {
			A[i][j] = make([]frontend.Variable, len(neuron))
			for k, weight := range neuron {
				A[i][j][k] = frontend.Variable(weight)
			}
		}
	}
	
	Biases := make([][]frontend.Variable, len(weightsData.Biases))
	for i, biasLayer := range weightsData.Biases {
		Biases[i] = make([]frontend.Variable, len(biasLayer))
		for j, bias := range biasLayer {
			Biases[i][j] = frontend.Variable(bias)
		}
	}
	
	Output := make([]frontend.Variable, len(expectedData.Expected))
	for i, datads := range expectedData.Expected {
		Output[i] = frontend.Variable(datads)
	}
	fmt.Print(Output)
}
