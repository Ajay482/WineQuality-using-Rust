use polars::prelude::*;


fn replace_column(df: &mut DataFrame, column_name: &str, pattern: &str, replacep: &str) {
    let column = df.column(column_name).unwrap();
    let replaced = column
        .utf8()
        .unwrap()
        .apply(|s| s.map(|v| v.replace(pattern, replacep).into()))
        .into_series();
    let _ = df.replace(column_name, replaced);
}

fn mean_col(df: &mut DataFrame, column_name: &str) -> f64{
    let column = df.column(column_name).unwrap();
    let mean = (column.mean().map(|m| (m * 100.0).round() / 100.0)).unwrap_or(0.0);
    mean
}

pub fn clean_df(mut df: DataFrame) -> Result<DataFrame, Box<dyn std::error::Error>> {


    //Replace column data
    replace_column(&mut df, "type", "white", "0");
    replace_column(&mut df, "type", "red", "1");

    // change datatypes from str to float
    let out = df
    .clone()
    .lazy()
    .select([
        col("type").cast(DataType::Int64),
    ])
    .collect()?;

    let _ = df.replace("type", (*out.column("type").unwrap()).clone());

    df = df
        .clone()
        .lazy()
        .with_columns([col("fixed_acidity").fill_null(mean_col(&mut df, "fixed_acidity")), 
            col("volatile_acidity").fill_null(mean_col(&mut df, "volatile_acidity")), 
            col("citric_acid").fill_null(mean_col(&mut df, "citric_acid")), 
            col("residual_sugar").fill_null(mean_col(&mut df, "residual_sugar")), 
            col("chlorides").fill_null(mean_col(&mut df, "chlorides")), col("sulphates").fill_null(mean_col(&mut df, "sulphates")), 
            col("pH").fill_null(mean_col(&mut df, "pH")),])
        .collect()?;

    let null_count_series = df
        .get_columns()
        .iter()
        .map(|col| col.is_null().sum())
        .collect::<Vec<_>>()
        .into_iter()
        ;

    // Display the DataFrame with null counts
    println!("Null Values: {:?}", null_count_series);

    Ok(df)

}