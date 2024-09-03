package main

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

type EqualCircuit struct {
    A frontend.Variable
    B frontend.Variable `gnark:",public"`
}

func (circuit *EqualCircuit) Define(api frontend.API) error {
    // The logic: A == B
    api.AssertIsEqual(circuit.A, circuit.B)
    return nil
}



func main() {
    var myCircuit EqualCircuit
	assignment := &EqualCircuit {
		A: 3,
		B: 33,
	}
	witness, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	cs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &myCircuit)
	pk, vk, _ := groth16.Setup(cs)
	proof, errproof := groth16.Prove(cs, pk, witness)
	fmt.Println("Error Proving ", errproof)
	publicWitness, _ := witness.Public()
	errverify := groth16.Verify(proof, vk, publicWitness)
    fmt.Println("Error in Verifying: ", errverify)
	
	if errverify == nil && errproof == nil{fmt.Print("Verify Successed")}

}
