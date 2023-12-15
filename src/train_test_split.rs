use polars::prelude::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::arrays::MutArray;
use smartcore::model_selection::train_test_split;

fn convert_features_to_matrix(df: &DataFrame) -> Result<DenseMatrix<f64>, Box<dyn std::error::Error>>{
    /* function to convert feature dataframe to a DenseMatrix, readable by smartcore*/

    let nrows = df.height();
    let ncols = df.width();

    // convert to array
    let features_res = df.to_ndarray::<Float64Type>(Default::default()).unwrap().clone();

    // create a zero matrix and populate with features
    let mut xmatrix: DenseMatrix<f64>  = Array2::zeros(nrows, ncols);

    // populate the matrix 
    // initialize row and column counters
    let mut col:  u32 = 0;
    let mut row:  u32 = 0;

    for val in features_res.iter(){

        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        // NB we are dereferencing the borrow with *val otherwise we would have a &val type, which is 
        // not what set wants
        xmatrix.set((m_row, m_col), *val);
        // check what we have to update
        if m_col == ncols-1 {
            row += 1 ;
            col = 0;
        } else{
            col += 1;
        }
    }
    
    // Ok so we can return DenseMatrix, otherwise we'll have std::result::Result<Densematrix, PolarsError>
    Ok(xmatrix)
}


pub fn train_test(df: DataFrame) 
    -> Result<(DenseMatrix<f64>, DenseMatrix<f64>, Vec<i64>, Vec<i64>), Box<dyn std::error::Error>> { 

    let features = df.select(vec!["type", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", 
                    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]);

    let target = df.column("quality").unwrap();

    let xmatrix = convert_features_to_matrix(&features.unwrap());

    let target_array = target
        .i64()?
        .to_vec();

    let mut y: Vec<i64> = Vec::new();
    for val in target_array.iter(){
        let vals = val.unwrap();
        y.push(vals);
    }

    let (x_train, x_test, y_train, y_test) = 
        train_test_split(&xmatrix.unwrap(), &y, 0.37, true, None);

    Ok((x_train, x_test, y_train, y_test))
}