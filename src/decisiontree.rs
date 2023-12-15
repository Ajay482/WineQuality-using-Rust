use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::*;
use smartcore::metrics::accuracy;


pub fn decision_tree_class(x_train: DenseMatrix<f64>, x_test: DenseMatrix<f64>, y_train: Vec<i64>, y_test: Vec<i64>) {
    // model 
    let tree = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default()).unwrap();
  
    // predictions 
    let preds = tree.predict(&x_test).unwrap(); 

    // accuracy
    let acc = accuracy(&y_test, &preds);
    println!("Decision Tree accuracy: {:?}", acc);

}
