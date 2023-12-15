use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::accuracy;


pub fn lr(x_train: DenseMatrix<f64>, x_test: DenseMatrix<f64>, y_train: Vec<i64>, y_test: Vec<i64>) {
    // model 
    let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();

    // predictions 
    let preds = linear_regression.predict(&x_test).unwrap(); 

    // accuracy
    let acc = accuracy(&y_test, &preds);
    println!("Linear Regression accuracy: {:?}", acc);

}