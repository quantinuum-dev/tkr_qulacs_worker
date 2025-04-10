use bitvec::{
    order::{Lsb0, Msb0},
    vec::BitVec,
};
use serde::{Deserialize, Serialize};
use tket_json_rs::register::{Bit, Qubit};

#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Debug)]
pub struct BackendResult {
    pub qubits: Vec<Qubit>,
    pub bits: Vec<Bit>,
    // pub counts: Vec<Count>,
    pub shots: OutcomeArray,
}

#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Debug)]
pub struct Count {
    pub outcome: OutcomeArray,
    pub count: i32,
}

#[derive(Deserialize, Serialize, Hash, Eq, PartialEq, Ord, PartialOrd, Clone, Debug)]
pub struct OutcomeArray {
    pub width: usize,
    pub array: Vec<Vec<u8>>,
}

#[inline]
fn convert_shot(shot: Vec<u64>) -> Vec<u8> {
    let bits: BitVec<u8, Msb0> = BitVec::<_, Lsb0>::from_vec(shot)
        .chunks(64)
        .map(|x| *x.first().unwrap())
        .collect();
    bits.into_vec()
}

#[inline]
pub(crate) fn convert_shots(shots: Vec<Vec<u64>>) -> OutcomeArray {
    let width = shots.first().unwrap().len();
    OutcomeArray {
        width,
        array: shots.into_iter().map(convert_shot).collect(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::convert_shot;

    #[rstest]
    #[case(vec![0, 0, 0, 0, 0, 0, 0, 0], vec![0])]
    #[case(vec![1, 0, 0, 0, 0, 0, 0, 0], vec![128])]
    #[case(vec![0, 1, 0, 0, 0, 0, 0, 0], vec![64])]
    #[case(vec![0, 0, 1, 0, 0, 0, 0, 0], vec![32])]
    #[case(vec![0, 0, 0, 1, 0, 0, 0, 0], vec![16])]
    #[case(vec![0, 0, 0, 0, 1, 0, 0, 0], vec![8])]
    #[case(vec![0, 0, 0, 0, 0, 1, 0, 0], vec![4])]
    #[case(vec![0, 0, 0, 0, 0, 0, 1, 0], vec![2])]
    #[case(vec![0, 0, 0, 0, 0, 0, 0, 1], vec![1])]
    #[case(vec![1, 1, 0, 0, 0, 0, 0, 0], vec![192])]
    #[case(vec![0, 0, 0, 0, 0, 0, 0, 1, 1], vec![1, 128])]
    #[case(vec![0, 0, 0, 0, 0, 0, 0, 1, 1], vec![1, 128])]
    fn convert_shot_examples(#[case] shot: Vec<u64>, #[case] expected: Vec<u8>) {
        assert_eq!(convert_shot(shot.to_vec()), expected);
    }
}
