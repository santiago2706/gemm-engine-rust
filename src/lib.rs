pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

// Export the matrix module so it can be used in other parts of the crate
pub mod benchmark;
pub mod matrix;
pub mod menu;
pub mod simulation;
pub mod visualizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
