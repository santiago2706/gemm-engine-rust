pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,

}
impl Matrix{
    pub fn new(rows: usize, cols:usize) -> Self{
        Self{
            rows, 
            cols,
            data: vec![0.0; rows * cols]
        }
    }
    pub fn cols(&self) -> usize{
        self.cols
    }
    pub fn rows(&self) -> usize{
        self.rows
    }
    //Función interna y privada para obtener el índice de un elemento en la matriz
    fn index(&self, rows: usize, cols: usize) -> usize{
        assert!(rows < self.rows, "Rows of index out of bounds");
        assert!(cols <self.cols, "Cols of index out of bounds");
        rows* self.cols + cols
    }
    //Función usada para obtener valores de la matriz 
    fn get(&self, rows: usize, cols: usize) -> f32{
        let idx = self.index(rows,cols);
        self.data[idx]
    }
    //Función para cambiar el valor de una matriz( usamos &mut self para editar la matriz)
    fn set(&mut self, rows: usize, cols: usize, value: f32){
        let idx = self.index(rows, cols);
        self.data[idx] = value;
    }

    //Construye una matriz a partir de sus valores en orden fila por fila
    pub fn from_vec(rows: usize, cols: usize, values: Vec<f32>) -> Self {
        assert_eq!(rows * cols, values.len(), "La cantidad de valores no coincide con rows * cols");
        Self { rows, cols, data: values }
    }

    //Imprime la matriz en consola con una etiqueta
    pub fn print(&self, label: &str) {
        println!("{}\n", label);
        for i in 0..self.rows {
            let row: Vec<String> = (0..self.cols).map(|j| self.get(i, j).to_string()).collect();
            println!("{}", row.join(" "));
        }
        println!();
    }

    //Multiplica esta matriz con otra, mostrando cada lectura y acumulación
    pub fn multiply_verbose(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Dimensiones incompatibles para la multiplicación");

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = 0.0f32;
                for k in 0..self.cols {
                    let a = self.get(i, k);
                    let b = other.get(k, j);
                    println!("Leyendo A[{}][{}]", i, k);
                    println!("Leyendo B[{}][{}]", k, j);
                    acc += a * b;
                    println!("{} × {} = {}", a, b, a * b);
                    println!("Acumulado = {}\n", acc);
                }
                result.set(i, j, acc);
            }
        }
        result
    }

}
//pruebas del codigo y que se ejecutan con cargo a la libreria de pruebas de rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::new(3, 4);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 4);
    }

    #[test]
    fn test_matrix_access() {
        // Criterio de finalización de la Tarea 3
        let mut m = Matrix::new(3, 3);

        m.set(0, 0, 1.0);
        m.set(1, 2, 5.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 5.0);
    }

    #[test]
    #[should_panic(expected = "Índice de fila fuera de límites")]
    fn test_matrix_out_of_bounds() {
        let m = Matrix::new(3, 3);
        // Esto debería causar un pánico (crash controlado) debido a los asserts
        m.get(20, 50);
    }
}