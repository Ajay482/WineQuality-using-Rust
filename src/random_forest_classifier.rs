use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::ensemble::random_forest_classifier::*;
use smartcore::metrics::accuracy;


pub fn random_class(x_train: DenseMatrix<f64>, x_test: DenseMatrix<f64>, y_train: Vec<i64>, y_test: Vec<i64>) {
    // model 
    let random_forest_classification = RandomForestClassifier::fit(&x_train, &y_train, Default::default()).unwrap();
  
    // predictions 
    let preds = random_forest_classification.predict(&x_test).unwrap(); 

    // accuracy
    let acc = accuracy(&y_test, &preds);
    println!("Random Forest accuracy: {:?}", acc);
}
