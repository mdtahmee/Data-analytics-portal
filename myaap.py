import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(
    page_title='Consoleflare Analytics Portal',
    page_icon='📊'
)

# Title
st.title(':rainbow[Data Analytics Portal]')
st.subheader(':gray[Explore Data with Ease]', divider='rainbow')

file = st.file_uploader('Drop CSV or Excel File', type=['csv', 'xlsx'])
if file is not None:
    if file.name.endswith('csv'):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    st.dataframe(data)
    st.info('File is successfully uploaded', icon='🚨')

    st.subheader(':rainbow[Basic Information of the Dataset]', divider='rainbow')
    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Top and Bottom Rows', 'Data Types', 'Columns'])

    with tab1:
        st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset.')
        st.subheader(':gray[Statistical Summary of the Dataset]')
        st.dataframe(data.describe())

    with tab2:
        st.subheader(':gray[Top Rows]')
        toprows = st.slider('Number of rows you want', 1, data.shape[0], key='topslider')
        st.dataframe(data.head(toprows))
        st.subheader(':gray[Bottom Rows]')
        bottomrows = st.slider('Number of rows you want', 1, data.shape[0], key='bottomslider')
        st.dataframe(data.tail(bottomrows))

    with tab3:
        st.subheader(':gray[Data Types of Columns]')
        st.dataframe(data.dtypes)

    with tab4:
        st.subheader('Column Names in Dataset')
        st.write(list(data.columns))

    # Missing Value Handling
    st.subheader(':rainbow[Handle Missing Values]', divider='rainbow')
    with st.expander('Fill or Drop Missing Values'):
        missing_action = st.radio('Select Action', options=['Fill with Mean', 'Fill with Median', 'Drop Rows'])
        if st.button('Apply Missing Value Handling'):
            if missing_action == 'Fill with Mean':
                data.fillna(data.mean(), inplace=True)
            elif missing_action == 'Fill with Median':
                data.fillna(data.median(), inplace=True)
            elif missing_action == 'Drop Rows':
                data.dropna(inplace=True)
            st.success('Missing values handled successfully!', icon='✅')
            st.dataframe(data)

    # Column Renaming
    st.subheader(':rainbow[Rename Columns]', divider='rainbow')
    with st.expander('Rename Dataset Columns'):
        new_names = {}
        for col in data.columns:
            new_name = st.text_input(f'Rename column {col}:', value=col)
            new_names[col] = new_name
        if st.button('Apply Column Renaming'):
            data.rename(columns=new_names, inplace=True)
            st.success('Columns renamed successfully!', icon='✅')
            st.dataframe(data)

    # Data Transformation
    st.subheader(':rainbow[Transform Your Data]', divider='rainbow')
    with st.expander('Normalize or Scale Numerical Data'):
        numeric_cols = data.select_dtypes(include=['number']).columns
        transform_col = st.selectbox('Select Column to Transform', options=numeric_cols)
        transform_type = st.selectbox('Select Transformation Type', options=['Normalize', 'Standardize'])

        if st.button('Apply Transformation'):
            if transform_type == 'Normalize':
                scaler = MinMaxScaler()
                data[transform_col] = scaler.fit_transform(data[[transform_col]])
            elif transform_type == 'Standardize':
                scaler = StandardScaler()
                data[transform_col] = scaler.fit_transform(data[[transform_col]])

            st.success(f'{transform_type} transformation applied to {transform_col}!', icon='✅')
            st.dataframe(data)

    # Histogram
    st.subheader(':rainbow[Histogram Analysis]', divider='rainbow')
    with st.expander('Create Histograms'):
        numeric_cols = data.select_dtypes(include=['number']).columns
        hist_col = st.selectbox('Select Column for Histogram', options=numeric_cols)
        if st.button('Generate Histogram'):
            fig = px.histogram(data, x=hist_col, nbins=20, title=f'Histogram of {hist_col}', template='plotly')
            fig.update_traces(marker_color='blue')
            st.plotly_chart(fig)

    # Pie Chart
    st.subheader(':rainbow[Pie Chart Visualization]', divider='rainbow')
    with st.expander('Visualize Categorical Data'):
        category_col = st.selectbox('Select Column for Pie Chart', options=data.columns)
        if st.button('Create Pie Chart'):
            fig = px.pie(data, names=category_col, title=f'Pie Chart of {category_col}', template='plotly_dark')
            st.plotly_chart(fig)

    # Bar Chart
    st.subheader(':rainbow[Bar Chart Visualization]', divider='rainbow')
    with st.expander('Analyze Data Distribution'):
        bar_col = st.selectbox('Select Column for Bar Chart', options=data.columns)
        if st.button('Create Bar Chart'):
            value_counts = data[bar_col].value_counts().reset_index()
            value_counts.columns = ['index', bar_col]  # Rename columns appropriately
            fig = px.bar(value_counts, x='index', y=bar_col, title=f'Bar Chart of {bar_col}', template='plotly_white')
            fig.update_traces(marker_color='orange')
            st.plotly_chart(fig)

    # Box Plot
    st.subheader(':rainbow[Box Plot for Outlier Detection]', divider='rainbow')
    with st.expander('Detect Outliers with Box Plots'):
        box_col = st.selectbox('Select Column for Box Plot', options=numeric_cols)
        if st.button('Generate Box Plot'):
            fig = px.box(data, y=box_col, title=f'Box Plot of {box_col}', template='seaborn')
            fig.update_traces(marker_color='red')
            st.plotly_chart(fig)

    # Scatter Plot
    st.subheader(':rainbow[Scatter Plot]', divider='rainbow')
    with st.expander('Visualize Relationships'):
        x_axis = st.selectbox('Select X-axis', options=numeric_cols)
        y_axis = st.selectbox('Select Y-axis', options=numeric_cols)
        scatter_color = st.selectbox('Color by Column', options=[None] + list(data.columns))
        if st.button('Create Scatter Plot'):
            fig = px.scatter(data, x=x_axis, y=y_axis, color=scatter_color, title='Scatter Plot', template='plotly')
            st.plotly_chart(fig)

    # Sorting Options
    st.subheader(':rainbow[Sort Your Data]', divider='rainbow')
    with st.expander('Sort Dataset by Columns'):
        sort_col = st.selectbox('Select Column to Sort By', options=data.columns)
        sort_order = st.radio('Sort Order', options=['Ascending', 'Descending'])
        if st.button('Apply Sorting'):
            data.sort_values(by=sort_col, ascending=(sort_order == 'Ascending'), inplace=True)
            st.success('Data sorted successfully!', icon='✅')
            st.dataframe(data)

    # Outlier Detection and Handling
    st.subheader(':rainbow[Outlier Detection]', divider='rainbow')
    with st.expander('Identify and Handle Outliers'):
        outlier_col = st.selectbox('Select Column for Outlier Detection', options=numeric_cols)
        if st.button('Detect Outliers'):
            Q1 = data[outlier_col].quantile(0.25)
            Q3 = data[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[outlier_col] < lower_bound) | (data[outlier_col] > upper_bound)]

            st.write(f'Outliers in {outlier_col}:')
            st.dataframe(outliers)

            if st.button('Remove Outliers'):
                data = data[(data[outlier_col] >= lower_bound) & (data[outlier_col] <= upper_bound)]
                st.success('Outliers removed successfully!', icon='✅')
                st.dataframe(data)

    # Data Export
    st.subheader(':rainbow[Export Processed Data]', divider='rainbow')
    with st.expander('Save Your Data'):
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)
        st.download_button(label='Download CSV', data=csv, file_name='processed_data.csv', mime='text/csv')

        # Excel Export
        excel_file = st.checkbox('Download as Excel')
        if excel_file:
            @st.cache_data
            def convert_to_excel(df):
                return df.to_excel(index=False).encode('utf-8')

            excel_data = convert_to_excel(data)
            st.download_button(label='Download Excel', data=excel_data, file_name='processed_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
else:
    st.warning('Please upload a file to proceed.')
