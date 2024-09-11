// package main

// import (
// 	"encoding/json"
// 	"fmt"
// 	"io/ioutil"
// 	"math/big"
// 	"os"

// 	"github.com/consensys/gnark-crypto/ecc"
// 	"github.com/consensys/gnark/backend/groth16"
// 	"github.com/consensys/gnark/frontend"
// 	"github.com/consensys/gnark/frontend/cs/r1cs"
// 	"github.com/consensys/gnark/std/math/cmp"
// )

// // ProveModelCircuit defines the circuit structure for a two-layer neural network with 3D weights, 2D biases, and 2D inputs
// type ProveModelCircuit struct {
// 	Weights  [2][3][4]frontend.Variable `gnark:",secret"` // 2 layers, each with 3 neurons, each neuron has 4 inputs
// 	Biases   [2][3]frontend.Variable    `gnark:",secret"` // 2 layers, each with 3 biases (one for each neuron)
// 	X        [3][4]frontend.Variable    `gnark:",public"` // 2D array for inputs (3 sets of inputs, each with 4 values)
// 	Expected [3]frontend.Variable       `gnark:",public"` // Expected output (3 values from final layer)
// }

// // Define the constraints of the circuit
// func (circuit *ProveModelCircuit) Define(api frontend.API) error {
// 	// First layer output (3 neurons)
// 	outputLayer1 := make([]frontend.Variable, 3)

// 	// Calculate the output for each neuron in the first layer using the first set of inputs
// 	for i := 0; i < 3; i++ {
// 		outputLayer1[i] = circuit.Biases[0][i] // First layer biases
// 		for j := 0; j < 4; j++ {
// 			outputLayer1[i] = api.Add(outputLayer1[i], api.Mul(circuit.Weights[0][i][j], circuit.X[i][j]))
// 		}
// 		// Apply ReLU activation
// 		isNegative := cmp.IsLess(api, outputLayer1[i], 0)
// 		outputLayer1[i] = api.Select(isNegative, 0, outputLayer1[i])
// 	}

// 	// Second layer output (final layer with 3 neurons)
// 	outputLayer2 := make([]frontend.Variable, 3)

// 	// Calculate the output for each neuron in the second layer
// 	for i := 0; i < 3; i++ {
// 		outputLayer2[i] = circuit.Biases[1][i] // Second layer biases
// 		for j := 0; j < 3; j++ {
// 			outputLayer2[i] = api.Add(outputLayer2[i], api.Mul(circuit.Weights[1][i][j], outputLayer1[j]))
// 		}
// 		// Apply ReLU activation
// 		isNegative := cmp.IsLess(api, outputLayer2[i], 0)
// 		outputLayer2[i] = api.Select(isNegative, 0, outputLayer2[i])
// 	}

// 	// Assert that the final layer output matches the expected output
// 	for i := 0; i < 3; i++ {
// 		fmt.Print("expetcted", outputLayer2[i])
// 		api.AssertIsEqual(outputLayer2[i], circuit.Expected[i])
// 	}

// 	return nil
// }

// // float64ToBigInt converts a float64 to a big.Int by scaling
// func float64ToBigInt(value float64) *big.Int {
// 	scaledValue := value * 1e9 // Scaling factor to convert float to int
// 	return big.NewInt(int64(scaledValue))
// }

// func float64ToBigIntNoScale(value float64) *big.Int {
// 	return big.NewInt(int64(value)) // No scaling applied here
// }

// func main() {
// 	// Step 1: Open the JSON files
// 	weightsFile, err := os.Open("weights.json")
// 	if err != nil {
// 		fmt.Println("Error opening weights file:", err)
// 		return
// 	}
// 	defer weightsFile.Close()

// 	inputsFile, err := os.Open("inputs.json")
// 	if err != nil {
// 		fmt.Println("Error opening inputs file:", err)
// 		return
// 	}
// 	defer inputsFile.Close()

// 	outputsFile, err := os.Open("outputs.json")
// 	if err != nil {
// 		fmt.Println("Error opening outputs file:", err)
// 		return
// 	}
// 	defer outputsFile.Close()

// 	// Step 2: Read the file contents into byte slices
// 	weightsByteValue, err := ioutil.ReadAll(weightsFile)
// 	if err != nil {
// 		fmt.Println("Error reading weights file:", err)
// 		return
// 	}

// 	inputsByteValue, err := ioutil.ReadAll(inputsFile)
// 	if err != nil {
// 		fmt.Println("Error reading inputs file:", err)
// 		return
// 	}

// 	outputsByteValue, err := ioutil.ReadAll(outputsFile)
// 	if err != nil {
// 		fmt.Println("Error reading outputs file:", err)
// 		return
// 	}

// 	// Step 3: Unmarshal the byte slices into the respective structs
// 	var weightsData struct {
// 		Weights [2][3][4]float64 `json:"weights"` // 3D array for weights
// 		Biases  [2][3]float64    `json:"biases"`  // 2D array for biases
// 	}

// 	var inputData struct {
// 		Inputs [3][4]float64 `json:"inputs"` // 2D array for inputs (3 sets of 4 values)
// 	}

// 	var expectedData struct {
// 		Expected [3]float64 `json:"outputs"`
// 	}

// 	err = json.Unmarshal(weightsByteValue, &weightsData)
// 	if err != nil {
// 		fmt.Println("Error unmarshalling weights JSON:", err)
// 		return
// 	}

// 	err = json.Unmarshal(inputsByteValue, &inputData)
// 	if err != nil {
// 		fmt.Println("Error unmarshalling inputs JSON:", err)
// 		return
// 	}

// 	err = json.Unmarshal(outputsByteValue, &expectedData)
// 	if err != nil {
// 		fmt.Println("Error unmarshalling outputs JSON:", err)
// 		return
// 	}

// 	// Create the circuit assignment
// 	assignment := &ProveModelCircuit{}
// 	// Assign weights and biases
// 	for layer := 0; layer < 2; layer++ {
// 		for i := 0; i < 3; i++ {
// 			for j := 0; j < 4; j++ {
// 				assignment.Weights[layer][i][j] = frontend.Variable(float64ToBigInt(weightsData.Weights[layer][i][j]))
// 			}
// 			assignment.Biases[layer][i] = frontend.Variable(float64ToBigInt(weightsData.Biases[layer][i]))
// 		}
// 	}

// 	// Assign inputs and expected output
// 	for i := 0; i < 3; i++ {
// 		for j := 0; j < 4; j++ {
// 			assignment.X[i][j] = frontend.Variable(float64ToBigInt(inputData.Inputs[i][j]))
// 		}
// 	}
// 	for i := 0; i < 3; i++ {
// 		assignment.Expected[i] = frontend.Variable(float64ToBigIntNoScale(expectedData.Expected[i]))
// 	}

// 	fmt.Println("expected assignment:", assignment.Expected)
// 	var myCircuit ProveModelCircuit
// 	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
// 	cs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
// 	pk, vk, _ := groth16.Setup(cs)
// 	proof, errproof := groth16.Prove(cs, pk, witness)
// 	fmt.Println("Error Proving: ", errproof)
// 	publicWitness, _ := witness.Public()
// 	errverify := groth16.Verify(proof, vk, publicWitness)
// 	fmt.Println("Error in Verifying: ", errverify)
// 	if errverify == nil && errproof == nil {
// 		fmt.Println("Verification succeeded")
// 	}
// }
