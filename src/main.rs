mod cleaning;
mod train_test_split;
mod linear_regression;
mod decisiontree;
mod random_forest_classifier;
mod knn_classifier;
use polars::prelude::*;
use std::env;


fn main() {

    env::set_var("RUST_BACKTRACE", "1");
    let file_path = r"C:\learning\winequality_final_project\winequality.csv";

    // Read the CSV file into a DataFrame
    let wine_eq_df = CsvReader::from_path(file_path)
        .unwrap()
        .has_header(true) // Set to true if your CSV has a header
        .finish()
        .unwrap();

    let wine_eq_df = cleaning::clean_df(wine_eq_df.clone()).unwrap();

    println!("{:?}", wine_eq_df);

    let (x_train, x_test, y_train, y_test) = train_test_split::train_test(wine_eq_df.clone()).unwrap();


    linear_regression::lr(x_train.clone(), x_test.clone(), y_train.clone(), y_test.clone());
    decisiontree::decision_tree_class(x_train.clone(), x_test.clone(), y_train.clone(), y_test.clone());
    random_forest_classifier::random_class(x_train.clone(), x_test.clone(), y_train.clone(), y_test.clone());
    knn_classifier::knnclass(x_train.clone(), x_test.clone(), y_train.clone(), y_test.clone());

}