use std::{collections::HashMap, env::args, f64::consts::PI, fs::File, path::PathBuf};

use exmex::{eval_str, ExError};
use num::Complex;
use qulacs_bridge::ffi::{
    add_gate_copy, merge, new_diagonal_matrix_gate, new_measurement, new_quantum_circuit,
    new_r_x_gate, new_r_z_gate, new_x_gate, new_y_gate, new_z_gate, QuantumCircuit,
};
use qulacs_bridge::UniquePtr;
use serde::{Deserialize, Serialize};
use tket_json_rs::{circuit_json::Command, OpType, SerialCircuit};

#[derive(Serialize, Deserialize)]
struct NodeDefinition {
    function_name: String,
    inputs: HashMap<String, PathBuf>,
    outputs: HashMap<String, PathBuf>,
    done_path: PathBuf,
    log_path: Option<PathBuf>,
}

fn get_arg(cmd: &Command, index: usize) -> u32 {
    cmd.args[index].1[0]
        .try_into()
        .expect("failed to get argument for command")
}

fn get_param(cmd: &Command, index: usize) -> Result<f64, ExError> {
    eval_str(&cmd.op.params.as_ref().expect("No params")[index])
}

fn convert_circuit(circuit: SerialCircuit) -> UniquePtr<QuantumCircuit> {
    let qulacs_circuit = new_quantum_circuit(circuit.qubits.len().try_into().unwrap());

    for command in circuit.commands {
        let gate = match command.op.op_type {
            // Pauli Gates
            OpType::X => new_x_gate(get_arg(&command, 0)),
            OpType::Y => new_y_gate(get_arg(&command, 0)),
            OpType::Z => new_z_gate(get_arg(&command, 0)),
            // Quantinuum Gateset
            OpType::PhasedX => {
                let index = get_arg(&command, 0);
                let alpha = get_param(&command, 0).unwrap();
                let beta = get_param(&command, 1).unwrap();
                merge(
                    &merge(&new_r_z_gate(index, beta), &new_r_x_gate(index, alpha)),
                    &new_r_z_gate(index, -beta),
                )
            }
            OpType::ZZPhase => {
                let index_1 = get_arg(&command, 0);
                let index_2 = get_arg(&command, 1);
                let alpha = get_param(&command, 1).unwrap();

                let exponent = Complex {
                    re: 0.0,
                    im: 0.5 * PI * alpha,
                };
                let neg_exponent = -exponent;

                let outer = neg_exponent.exp();
                let inner = exponent.exp();

                new_diagonal_matrix_gate(
                    &[index_1, index_2],
                    &[outer.into(), inner.into(), inner.into(), outer.into()],
                )
            }
            OpType::Measure => {
                // TODO: This doesn't feel safe to me without checking
                // what kind of registers these are.
                let index = get_arg(&command, 0);
                let reg = get_arg(&command, 1);

                new_measurement(index, reg)
            }
            _ => unimplemented!(),
        };

        add_gate_copy(&qulacs_circuit, &gate);
    }

    qulacs_circuit
}

fn simulate_circuit() {}

fn simulate_circuits(list_circ: &[SerialCircuit], n_shot: i32) {}

fn run(node_definition: &NodeDefinition) -> std::io::Result<()> {
    match &*node_definition.function_name {
        "simulate_circuits" => {
            let circuits_file = File::open(&node_definition.inputs["circuits"])?;
            let circuits: Vec<_> = serde_json::from_reader(&circuits_file)?;

            let results = simulate_circuits(&circuits, 100);
            Ok(())
        }
        _ => unimplemented!(),
    }
}

fn main() -> std::io::Result<()> {
    let args = args();
    let node_definition_path = args
        .skip(1)
        .next()
        .expect("expected a node definition path as first argument.");

    let node_definition_file = File::open(node_definition_path)?;
    let node_definition: NodeDefinition = serde_json::from_reader(node_definition_file)?;

    run(&node_definition);

    Ok(())
}

#[cfg(test)]
mod tests {
    use tket_json_rs::circuit_json::Operation;

    use super::*;

    #[test]
    fn test_get_param_parse_expression() {
        let mut op = Operation::from_optype(OpType::PhasedX);
        op.params = Some(vec!["1/2".to_string()]);
        let command = Command {
            op,
            args: Vec::new(),
            opgroup: None,
        };

        let param = get_param(&command, 0).unwrap();
        assert_eq!(param, 0.5);
    }
}
