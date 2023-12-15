use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::*;
use smartcore::metrics::accuracy;


pub fn knnclass(x_train: DenseMatrix<f64>, x_test: DenseMatrix<f64>, y_train: Vec<i64>, y_test: Vec<i64>) {

    // model 
    let knn = KNNClassifier::fit(&x_train, &y_train, Default::default()).unwrap();

    // predictions 
    let preds = knn.predict(&x_test).unwrap();

    // accuracy
    let acc = accuracy(&y_test, &preds);
    println!("KNN accuracy: {:?}", acc);

}