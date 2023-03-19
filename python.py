import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import time

# Define a function to load data
@st.cache_data()
def load_data(file_path):
    return pd.read_csv(file_path)

# Define a function to merge datasets
def merge_datasets(df1, df2):
    # merge the two datasets on a common column
    merged_df = pd.merge(df1, df2, on='common_column')
    return merged_df

# Define a function to merge two datasets
def merge_datasets(data1, data2):
    merged_data = pd.merge(data1, data2, on='key_column')
    return merged_data

# Define function for converting categorical columns
def convert_categorical(df, method):
    if method == 'OneHot':
        encoder = OneHotEncoder()
        encoded_df = pd.DataFrame(encoder.fit_transform(df).toarray())
    elif method == 'Label':
        encoder = LabelEncoder()
        encoded_df = df.apply(encoder.fit_transform)
    return encoded_df


def app():
    # Define the title and description of the app
    st.title("Random Forest Regression")
    
    
# Run the streamlit app
if __name__ == '__main__':
    app()

# Upload two datasets
st.header('Upload and Merge Datasets')
data_file_1 = st.file_uploader('Upload first dataset', type=['csv', 'xlsx'])
data_file_2 = st.file_uploader('Upload second dataset', type=['csv', 'xlsx'])

# Merge the two datasets
st.header('Merged Dataset')
if data_file_1 is not None and data_file_2 is not None:
    with st.spinner('Loading datasets...'):
        df1 = load_data(data_file_1)
        df2 = load_data(data_file_2)
        with st.spinner('Merging datasets...'):
            merged_df = pd.merge(df1, df2, on='id')
            st.write('Merged Dataset')
            st.write(merged_df)

    # Check summary statistics and data types
    st.header('Summary Statistics and Data Types')
    st.write('Summary Statistics')
    st.write(merged_df.describe())
    st.write('Data Types')
    st.write(merged_df.dtypes)

    # Check correlation
    st.header('Correlation')
    st.write(merged_df.corr())
    
    #correlation with heatmap      
    st.subheader("Select column for checking target variable distribution with heatmap")
    target_col = st.selectbox("Target variable", merged_df.columns)
    if target_col != 'id':
        st.write("Target variable:", target_col)
        st.write("Summary statistics:")
        st.write(merged_df[target_col].describe())
        st.write("Distribution:")
        sns.histplot(data=merged_df, x=target_col, kde=True)
        st.pyplot()
        st.write("Correlation with other variables:")
        corr = merged_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()


    # Print numerical and categorical features
    st.header('Numerical and Categorical Features')
    numerical_features = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = merged_df.select_dtypes(include=['object']).columns.tolist()
    st.write('Numerical Features')
    st.write(numerical_features)
    st.write('Categorical Features')
    st.write(categorical_features)
    

    # Missing value imputation
    st.header('Missing Value Imputation')
    imputation_methods = ['Drop NA', 'Mean Imputation', 'Median Imputation', 'Mode Imputation']
    imputation_choice = st.selectbox('Choose an imputation method:', imputation_methods)
    if imputation_choice == 'Drop NA':
        cleaned_df = merged_df.dropna()
    elif imputation_choice == 'Mean Imputation':
        imputer = SimpleImputer(strategy='mean')
        cleaned_df = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns)
    elif imputation_choice == 'Median Imputation':
        imputer = SimpleImputer(strategy='median')
        cleaned_df = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns)
    elif imputation_choice == 'Mode Imputation':
        imputer = SimpleImputer(strategy='most_frequent')
        cleaned_df = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns)
        
    st.write('Cleaned Dataset')
    st.write(cleaned_df)
        
        
    #Convert categorical columns
    st.subheader('Convert Categorical Columns')
    cat_cols = merged_df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        method = st.radio('Select conversion method:', ['None','OneHot', 'Label'])
        encoded_df = convert_categorical(merged_df[cat_cols], method)
        merged_df = pd.concat([merged_df.drop(cat_cols, axis=1), cleaned_df], axis=1)
        st.write('Converted categorical columns:', cat_cols)
    else:
        st.write('No categorical columns found.')
        
    st.write('Encoded Dataset')
    st.write(encoded_df)
    
    # Add a button to start feature scaling
    st.header(" Feature Scaling")
    if st.button("Perform Feature Scaling"):
        # Choose a feature scaling method
        scaling_method = st.selectbox("Select a scaling method", ["None", "Standardization", "Normalization"])

        # Perform feature scaling
        if scaling_method == "Standardization":
            scaler = StandardScaler()
            merged_df = scaler.fit_transform(merged_df)
        elif scaling_method == "Normalization":
            scaler = MinMaxScaler()
            merged_df = scaler.fit_transform(merged_df)
        else:
            merged_df = merged_df
    
    # Define a function to show distribution plots
    def show_distribution_plots():
        st.write("## Distribution Plots")
        sns.set(style="white")
           
          
    # Show distribution plot
    if st.checkbox('Show distribution plot'):
        col = st.selectbox('Select a column', encoded_df.columns)
        sns.histplot(encoded_df[col], kde=True)
        st.pyplot()
                
                
    # Display outliers in boxplot
        st.header("Display Outliers")
        num_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        col = st.selectbox("Select a numerical column to display outliers", num_cols)
        if col is not None:
            fig, ax = plt.subplots()
            ax.boxplot(merged_df[col], vert=False)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
            
            # Choose outlier treatment method
            outlier_method = st.selectbox("Select outlier treatment method:", ['None', 'Z-score', 'IQR'])

            if outlier_method != 'None':
                # Detect and treat outliers using z-score
                if outlier_method == 'Z-score':
                    z = np.abs(stats.zscore(merged_df.select_dtypes(include=[np.number])))
                    threshold = st.number_input("Enter Z-score threshold value:")
                    merged_df = merged_df[(z < threshold).all(axis=1)]
                    st.write("Outliers removed using Z-score method:")
                    st.write(merged_df)

                # Detect and treat outliers using IQR
                elif outlier_method == 'IQR':
                    q1 = merged_df.select_dtypes(include=[np.number]).quantile(0.25)
                    q3 = merged_df.select_dtypes(include=[np.number]).quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    merged_df = merged_df[~((merged_df.select_dtypes(include=[np.number]) < lower_bound) | (merged_df.select_dtypes(include=[np.number]) > upper_bound)).any(axis=1)]
                    st.write("Outliers removed using IQR method:")
                    st.write(merged_df)
        
        
    #Split data into training and test sets
    # Allow users to select the target variable and the features for the model
    st.header('Select target variable and features')
    target_col = st.selectbox('Select the target variable', options=merged_df.columns)
    feature_cols = st.multiselect('Select the features', options=merged_df.columns.drop(target_col))
    
    #Split data into training and test sets
    # Perform train_test_split
    st.header('Train-test split')
    test_size = st.slider('Select the test size', min_value=0.1, max_value=0.5, step=0.1, value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(merged_df[feature_cols], merged_df[target_col], test_size=test_size, random_state=0)



    # Build model
    st.header('Random Forest Regression Model')
    
    #Hyper parameters
    st.header('Hyper parameters')
    n_estimators = st.slider('Number of trees', min_value=1, max_value=100, value=10, step=1)
    max_depth = st.slider('Max depth', min_value=1, max_value=100, value=5, step=1)
    min_samples_split = st.slider('min samples split', min_value=1, max_value=100, value=5, step=1)

    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    st.header('Model Evaluation')
    st.write(f'Training Score: {rf_model.score(X_train, y_train)}')
    st.write(f'Test Score: {rf_model.score(X_test, y_test)}')
    
    # Model Performance
    st.header('Model Performance')

    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write('y_pred:', y_pred)
    st.write('Mean Squared Error:', mse)
    st.write('R2 Score:', r2)
    
if __name__ == '__main__':
    app()
    
    
