mod results;

use std::f64::consts::PI;
use std::{collections::HashMap, env::args, fs::File, path::PathBuf};

use anyhow::Result;
use fasteval::Evaler;
#[cfg(feature = "mpi")]
use mpi::topology::Communicator;
use qulacs_bridge::ffi::{
    add_gate_copy, merge, new_cnot_gate, new_h_gate, new_pauli_rotation_gate, new_quantum_circuit,
    new_quantum_state, new_r_x_gate, new_r_z_gate, new_x_gate, new_y_gate, new_z_gate,
    set_zero_state, update_quantum_state, Pauli, QuantumCircuit,
};
#[cfg(not(feature = "mpi"))]
use qulacs_bridge::ffi::{get_classical_register, new_measurement};
#[cfg(feature = "mpi")]
use qulacs_bridge::ffi::{new_identity_gate, quantum_state_sampling};
use qulacs_bridge::UniquePtr;
use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use tket_json_rs::{circuit_json::Command, OpType, SerialCircuit};

#[cfg(feature = "mpi")]
use crate::results::{convert_samples, BackendResult};
#[cfg(not(feature = "mpi"))]
use crate::results::{convert_shots, BackendResult};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
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

struct Evaluator {
    parser: fasteval::Parser,
    slab: fasteval::Slab,
}

impl Evaluator {
    fn new() -> Self {
        Self {
            parser: fasteval::Parser::new(),
            slab: fasteval::Slab::new(),
        }
    }

    fn eval_param(&mut self, cmd: &Command, index: usize) -> Result<f64> {
        let param = &cmd.op.params.as_ref().expect("No params")[index];
        // Hacks to get around the way symengine specifies particular operators
        // and constants. Certainly not perfect and should be dealt with later.
        let param = param.to_string().replace("**", "^");
        let param = param.to_string().replace("pi", "pi()");
        let mut ns = fasteval::EmptyNamespace;

        Ok(self
            .parser
            .parse(&param, &mut self.slab.ps)?
            .from(&self.slab.ps)
            .eval(&self.slab, &mut ns)?)
    }
}

fn new_rng(seed: Option<u64>) -> Box<dyn RngCore> {
    if let Some(seed) = seed {
        Box::new(SmallRng::seed_from_u64(seed))
    } else {
        Box::new(rand::rng())
    }
}

fn convert_circuit(
    circuit: &SerialCircuit,
    #[cfg(feature = "mpi")] _rng: &mut Box<dyn RngCore>,
    #[cfg(not(feature = "mpi"))] rng: &mut Box<dyn RngCore>,
) -> Result<UniquePtr<QuantumCircuit>> {
    let qulacs_circuit = new_quantum_circuit(circuit.qubits.len().try_into().unwrap());
    let mut evaluator = Evaluator::new();

    for command in &circuit.commands {
        let gate = match command.op.op_type {
            // Pauli Gates
            OpType::X => new_x_gate(get_arg(&command, 0)),
            OpType::Y => new_y_gate(get_arg(&command, 0)),
            OpType::Z => new_z_gate(get_arg(&command, 0)),
            // Hadamard
            OpType::H => new_h_gate(get_arg(&command, 0)),
            // CNOT
            OpType::CX => {
                let control = get_arg(&command, 0);
                let target = get_arg(&command, 1);
                new_cnot_gate(control, target)
            }
            // Quantinuum Gateset
            OpType::PhasedX => {
                let index = get_arg(&command, 0);
                // Qulacs rotations are the opposite direction
                // to pytket rotations.
                let alpha = -evaluator.eval_param(&command, 0).unwrap() * PI;
                let beta = -evaluator.eval_param(&command, 1).unwrap() * PI;
                merge(
                    &merge(&new_r_z_gate(index, beta), &new_r_x_gate(index, alpha)),
                    &new_r_z_gate(index, -beta),
                )
            }
            OpType::ZZPhase => {
                let index_1 = get_arg(&command, 0);
                let index_2 = get_arg(&command, 1);
                let alpha = -evaluator.eval_param(&command, 0).unwrap() * PI;

                new_pauli_rotation_gate(&[index_1, index_2], &[Pauli::Z, Pauli::Z], alpha)
            }
            // Mid-circuit measurement is not supported for MPI.
            #[cfg(feature = "mpi")]
            OpType::Measure => {
                let index = get_arg(&command, 0);

                new_identity_gate(index)
            }
            #[cfg(not(feature = "mpi"))]
            OpType::Measure => {
                let index = get_arg(&command, 0);
                let reg = get_arg(&command, 1);
                let seed = rng.random();

                new_measurement(index, reg, seed)
            }
            _ => unimplemented!(),
        };

        add_gate_copy(&qulacs_circuit, &gate);
    }

    Ok(qulacs_circuit)
}

fn simulate_circuit(
    circuit: &SerialCircuit,
    n_shot: u32,
    mut rng: &mut Box<dyn RngCore>,
) -> Result<BackendResult> {
    let n_qubits = circuit.qubits.len();
    let bits = circuit.bits.clone();
    let qubits = circuit.qubits.clone();

    let state = new_quantum_state(n_qubits.try_into().unwrap(), true);
    let circuit = convert_circuit(circuit, &mut rng)?;

    // TODO: We need the same seed on each node for MPI?
    #[cfg(feature = "mpi")]
    let shots = {
        set_zero_state(&state);
        update_quantum_state(&circuit, &state, rng.random());
        let samples = quantum_state_sampling(&state, n_shot, rng.random());
        convert_samples(n_shot as usize, samples)
    };

    #[cfg(not(feature = "mpi"))]
    let shots = {
        let mut shots = Vec::new();
        for _ in 0..n_shot {
            set_zero_state(&state);
            update_quantum_state(&circuit, &state, rng.random());
            let register = get_classical_register(&state);
            shots.push(register);
        }
        convert_shots(shots)
    };
    Ok(BackendResult {
        bits,
        qubits,
        shots,
    })
}

fn simulate_circuits(
    list_circ: &[SerialCircuit],
    n_shot: u32,
    seed: Option<u64>,
) -> Result<Vec<BackendResult>> {
    let mut rng = new_rng(seed);
    list_circ
        .iter()
        .map(|circuit| simulate_circuit(circuit, n_shot, &mut rng))
        .collect()
}

fn run(node_definition: &NodeDefinition) -> Result<()> {
    match &*node_definition.function_name {
        "submit" => {
            let circuits_file = File::open(&node_definition.inputs["circuits"])?;
            let circuits: Vec<SerialCircuit> = serde_json::from_reader(&circuits_file)?;

            let n_shots_file = File::open(&node_definition.inputs["n_shots"])?;
            let n_shots: u32 = serde_json::from_reader(&n_shots_file)?;

            let results = simulate_circuits(&circuits, n_shots, None)?;

            let outputs_file = File::create(&node_definition.outputs["backend_results"])?;
            serde_json::to_writer(outputs_file, &results)?;

            File::create(&node_definition.done_path)?;
            Ok(())
        }
        "submit_single" => {
            let circuit_file = File::open(&node_definition.inputs["circuit"])?;
            let circuit: SerialCircuit = serde_json::from_reader(&circuit_file)?;

            let n_shots_file = File::open(&node_definition.inputs["n_shots"])?;
            let n_shots: u32 = serde_json::from_reader(&n_shots_file)?;

            let mut rng = new_rng(None);
            let result = simulate_circuit(&circuit, n_shots, &mut rng)?;

            let output_file = File::create(&node_definition.outputs["backend_result"])?;
            serde_json::to_writer(output_file, &result)?;

            File::create(&node_definition.done_path)?;
            Ok(())
        }


        #[cfg(feature = "mpi")]
        "submit_single_mpi" => {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            let _size = world.size();
            let rank = world.rank();

            let circuit_file = File::open(&node_definition.inputs["circuit"])?;
            let circuit: SerialCircuit = serde_json::from_reader(&circuit_file)?;

            let n_shots_file = File::open(&node_definition.inputs["n_shots"])?;
            let n_shots: u32 = serde_json::from_reader(&n_shots_file)?;

            let mut rng = new_rng(None);
            let result = simulate_circuit(&circuit, n_shots, &mut rng)?;

            if rank == 0 {
                let output_file = File::create(&node_definition.outputs["backend_result"])?;
                serde_json::to_writer(output_file, &result)?;

                File::create(&node_definition.done_path)?;
            }

            Ok(())
        }

        _ => unimplemented!(),
    }
}

fn main() -> Result<()> {
    let args = args();
    let node_definition_path = args
        .skip(1)
        .next()
        .expect("expected a node definition path as first argument.");

    let node_definition_file = File::open(node_definition_path)?;
    let node_definition: NodeDefinition = serde_json::from_reader(node_definition_file)?;

    run(&node_definition)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tket_json_rs::circuit_json::Operation;

    use super::*;

    #[rstest]
    #[case("1", 1.0)]
    #[case("1/2", 0.5)]
    #[case("2**2", 4.0)]
    #[case("2**2", 4.0)]
    #[case("pi", 3.141592653589793)]
    #[case("-(234.0 - 0.304074011085012*pi**(-1))", -233.90321023614007)]
    #[case("-234.0 + 0.304074011085012/pi", -233.90321023614007)]
    #[case("0.306088405064804", 0.306088405064804)]
    fn get_param_parse_expression(#[case] expr: &str, #[case] expected: f64) {
        let mut op = Operation::from_optype(OpType::PhasedX);
        op.params = Some(vec![expr.to_string()]);
        let command = Command {
            op,
            args: Vec::new(),
            opgroup: None,
        };

        let mut evaluator = Evaluator::new();
        let param = evaluator.eval_param(&command, 0).unwrap();
        assert_eq!(param, expected);
    }

    #[test]
    fn test_rng() {
        let mut rng1 = new_rng(None);
        let mut rng2 = new_rng(None);

        // Some chance this could potentially happen, but it's
        // reasonably unlikely.
        assert_ne!(rng1.random::<u32>(), rng2.random::<u32>());
        assert_ne!(rng1.random::<u64>(), rng2.random::<u64>());
    }

    #[test]
    fn test_seeded_rng() {
        let mut rng1 = new_rng(Some(1));
        let mut rng2 = new_rng(Some(1));

        assert_eq!(rng1.random::<u32>(), rng2.random::<u32>());
        assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());

        insta::assert_debug_snapshot!(rng1.random::<u32>());
        insta::assert_debug_snapshot!(rng2.random::<u32>());
    }

    #[test]
    fn bell_measurement_circuit() -> Result<()> {
        let file = File::open("data/bell_measurement.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(None);
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0 || shot == 192
        }));

        Ok(())
    }

    #[test]
    fn bell_measurement_circuit_seeded() -> Result<()> {
        let file = File::open("data/bell_measurement.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(Some(1));
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0 || shot == 192
        }));

        insta::assert_debug_snapshot!(result.shots);

        Ok(())
    }

    #[test]
    fn phased_x_circuit() -> Result<()> {
        let file = File::open("data/phasedx.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(None);
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0 || shot == 128
        }));

        Ok(())
    }

    #[test]
    fn phased_x_circuit_seeded() -> Result<()> {
        let file = File::open("data/phasedx.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(Some(1));
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0 || shot == 128
        }));

        insta::assert_debug_snapshot!(result.shots);

        Ok(())
    }

    #[test]
    fn zz_phase_circuit() -> Result<()> {
        let file = File::open("data/zzphase.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(None);
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0
        }));

        Ok(())
    }

    #[test]
    fn zz_phase_circuit_seeded() -> Result<()> {
        let file = File::open("data/zzphase.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(Some(1));
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0
        }));

        insta::assert_debug_snapshot!(result.shots);

        Ok(())
    }

    #[test]
    fn phased_x_zz_phase_circuit() -> Result<()> {
        let file = File::open("data/phasedx_zzphase.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(None);
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0 || shot == 64 || shot == 128 || shot == 192
        }));

        Ok(())
    }

    #[test]
    fn phased_x_zz_phase_circuit_seeded() -> Result<()> {
        let file = File::open("data/phasedx_zzphase.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(Some(1));
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        assert!(result.shots.array.iter().all(|x| {
            let shot = x[0];
            shot == 0 || shot == 64 || shot == 128 || shot == 192
        }));

        insta::assert_debug_snapshot!(result.shots);

        Ok(())
    }

    #[test]
    fn chemistry_example_circuit() -> Result<()> {
        let file = File::open("data/chemistry.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(None);
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);

        Ok(())
    }

    #[test]
    fn chemistry_example_circuit_seeded() -> Result<()> {
        let file = File::open("data/chemistry.json")?;
        let circuit: SerialCircuit = serde_json::from_reader(&file)?;

        let mut rng = new_rng(Some(1));
        let result = simulate_circuit(&circuit, 10, &mut rng)?;

        assert_eq!(result.shots.array.len(), 10);
        insta::assert_debug_snapshot!(result.shots);

        Ok(())
    }
}
