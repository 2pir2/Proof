package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/big"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/math/cmp"
)

// ProveModelCircuit defines the circuit structure for a simplified two-layer neural network
type ProveModelCircuit struct {
	Weights  [2][4][4]frontend.Variable `gnark:",secret"` // 2 layers, each with 4 neurons, each neuron has 4 inputs
	Biases   [2][4]frontend.Variable    `gnark:",secret"` // 2 layers, each with 4 biases (one for each neuron)
	X        [3][4]frontend.Variable    `gnark:",public"` // Array of 3 input vectors (each 4 values)
	Expected [3]frontend.Variable       `gnark:",public"` // Array of 3 expected outputs
}

// Define the constraints of the circuit
func (circuit *ProveModelCircuit) Define(api frontend.API) error {
	// Loop through each input-output pair (3 input-output pairs in total)
	for k := 0; k < 3; k++ {
		outputLayer := make([][]frontend.Variable, 2) // Output for each layer (2 layers)
		outputLayer[0] = make([]frontend.Variable, 4) // First layer outputs (4 neurons)
		outputLayer[1] = make([]frontend.Variable, 4) // Second layer outputs (4 neurons)

		// Combine the loops for both Layer 1 and Layer 2
		for layer := 0; layer < 2; layer++ {
			for i := 0; i < 4; i++ {
				// Use either inputs or previous layer's output as the input
				if layer == 0 {
					outputLayer[layer][i] = circuit.Biases[layer][i]
					for j := 0; j < 4; j++ {
						outputLayer[layer][i] = api.Add(outputLayer[layer][i], api.Mul(circuit.Weights[layer][i][j], circuit.X[k][j]))
					}
				} else {
					outputLayer[layer][i] = circuit.Biases[layer][i]
					for j := 0; j < 4; j++ {
						outputLayer[layer][i] = api.Add(outputLayer[layer][i], api.Mul(circuit.Weights[layer][i][j], outputLayer[layer-1][j]))
					}
				}
				// Apply ReLU activation
				isNegative := cmp.IsLess(api, outputLayer[layer][i], 0)
				outputLayer[layer][i] = api.Select(isNegative, 0, outputLayer[layer][i])
			}
		}

		// Find the maximum value (argmax) in the second layer output
		maxVal := outputLayer[1][0]
		maxIdx := frontend.Variable(0)

		for i := 1; i < 4; i++ {
			isLess := cmp.IsLess(api, maxVal, outputLayer[1][i])

			maxVal = api.Select(isLess, outputLayer[1][i], maxVal)
			maxIdx = api.Select(isLess, frontend.Variable(i), maxIdx)

		}

		// Assert that the predicted output matches the expected output for each input

		api.AssertIsEqual(maxVal, outputLayer[1][3])

	}

	return nil
}

// float64ToBigInt converts a float64 to a big.Int by scaling
func float64ToBigInt(value float64) *big.Int {
	scaledValue := value * 1e6 // Scaling factor to convert float to int
	return big.NewInt(int64(scaledValue))
}

func float64ToBigIntNoScale(value float64) *big.Int {
	return big.NewInt(int64(value)) // No scaling applied here
}

func main() {
	// Step 1: Open the JSON files
	weightsFile, err := os.Open("weights.json")
	if err != nil {
		fmt.Println("Error opening weights file:", err)
		return
	}
	defer weightsFile.Close()

	inputsFile, err := os.Open("inputs.json")
	if err != nil {
		fmt.Println("Error opening inputs file:", err)
		return
	}
	defer inputsFile.Close()

	outputsFile, err := os.Open("outputs.json")
	if err != nil {
		fmt.Println("Error opening outputs file:", err)
		return
	}
	defer outputsFile.Close()

	// Step 2: Read the file contents into byte slices
	weightsByteValue, err := ioutil.ReadAll(weightsFile)
	if err != nil {
		fmt.Println("Error reading weights file:", err)
		return
	}

	inputsByteValue, err := ioutil.ReadAll(inputsFile)
	if err != nil {
		fmt.Println("Error reading inputs file:", err)
		return
	}

	outputsByteValue, err := ioutil.ReadAll(outputsFile)
	if err != nil {
		fmt.Println("Error reading outputs file:", err)
		return
	}

	// Step 3: Unmarshal the byte slices into the respective structs
	var weightsData struct {
		Weights [2][4][4]float64 `json:"weights"` // 3D array for weights (2 layers, each with 4 neurons)
		Biases  [2][4]float64    `json:"biases"`  // 2D array for biases (2 layers, each with 4 biases)
	}

	var inputData struct {
		Inputs [3][4]float64 `json:"inputs"` // Array of 3 input vectors (each 4 values)
	}

	var expectedData struct {
		Expected [3]float64 `json:"outputs"` // Array of 3 expected outputs
	}

	err = json.Unmarshal(weightsByteValue, &weightsData)
	if err != nil {
		fmt.Println("Error unmarshalling weights JSON:", err)
		return
	}

	err = json.Unmarshal(inputsByteValue, &inputData)
	if err != nil {
		fmt.Println("Error unmarshalling inputs JSON:", err)
		return
	}

	err = json.Unmarshal(outputsByteValue, &expectedData)
	if err != nil {
		fmt.Println("Error unmarshalling outputs JSON:", err)
		return
	}

	// Create the circuit assignment
	assignment := &ProveModelCircuit{}
	// Assign weights and biases
	for layer := 0; layer < 2; layer++ {
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				assignment.Weights[layer][i][j] = frontend.Variable(float64ToBigInt(weightsData.Weights[layer][i][j]))
			}
			assignment.Biases[layer][i] = frontend.Variable(float64ToBigInt(weightsData.Biases[layer][i]))
		}
	}

	// Assign inputs and expected outputs
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			assignment.X[i][j] = frontend.Variable(float64ToBigInt(inputData.Inputs[i][j]))
		}
		assignment.Expected[i] = frontend.Variable(float64ToBigIntNoScale(expectedData.Expected[i]))
	}

	fmt.Println("expected assignment:", assignment)

	var myCircuit ProveModelCircuit
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	cs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	pk, vk, _ := groth16.Setup(cs)
	proof, errproof := groth16.Prove(cs, pk, witness)
	fmt.Println("Error Proving: ", errproof)
	publicWitness, _ := witness.Public()
	errverify := groth16.Verify(proof, vk, publicWitness)
	fmt.Println("Error in Verifying: ", errverify)
	if errverify == nil && errproof == nil {
		fmt.Println("Verification succeeded")
	}
}
