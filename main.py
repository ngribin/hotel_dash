import streamlit as st
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k



# Set page title
st.set_page_config(page_title="Dashboard with Tabs")

def load_and_preprocess_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    dfs = pd.read_excel(xls, sheet_name=sheet_names, header=None)
    processed_dfs = {sheet_name: preprocess_data(df) for sheet_name, df in dfs.items()}
    return processed_dfs

# Data preprocessing
def preprocess_data(df):
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = df.iloc[0]
    df = df[1:]
    return df

# Creating tabs
tabs = ["Data", "Analytics", "Forecast", "Download", "Recommendations", "Other Tab"]
selected_tab = st.sidebar.radio("Select Tab", tabs)


# Loading your hotel dataset 
data = pd.read_excel("hotels.xlsx")
data = data.dropna()

# Create a Streamlit app with multiple tabs
#st.set_page_config(layout="wide")

# Function to create recommendations
def create_recommendations(user_location, user_score, user_stars):
    dataset = Dataset()
    dataset.fit(data['Location'], data['Title'])

    # The interaction matrix
    (interactions, _) = dataset.build_interactions(
        (user, item, 1.0) for user, item in zip(data['Location'], data['Title'])
    )

    # Training the LightFM model
    model = LightFM(loss='warp')
    model.fit(interactions, epochs=30, num_threads=2)

    # Maping user parameters to the dataset's internal IDs
    user_id = dataset.mapping()[0][user_location]

    # Filter hotels based on user_score and user_stars
    filtered_hotels = data[(data['Score'] >= user_score) & (data['Stars'] >= user_stars)]

    # Generate recommendations for the user based on the filtered hotels
    n_users, n_items = interactions.shape
    scores = model.predict(user_id, np.arange(n_items))

    # Get recommendations for the filtered hotels
    # filtered_indices = [dataset.mapping()[2][hotel] for hotel in filtered_hotels['Title']]
    # filtered_scores = scores[filtered_indices]
    # filtered_hotel_indices = np.argsort(-filtered_scores)[:5]
    # filtered_top_hotels = filtered_hotels.iloc[filtered_hotel_indices]['Title']

    filtered_indices = [dataset.mapping()[2][hotel] for hotel in filtered_hotels['Title']]
    filtered_scores = scores[filtered_indices]
    filtered_hotel_indices = np.argsort(-filtered_scores)[:5]
    filtered_top_hotels = filtered_hotels.iloc[filtered_hotel_indices][['Title', 'Url']]

    st.write(f"Top 5 hotel recommendations for user in '{user_location}' with a score of at least {user_score} and {user_stars} stars:")
    # for i, hotel in enumerate(filtered_top_hotels, 1):
    #     st.write(f"{i}. {hotel}")
    for i, row in enumerate(filtered_top_hotels.iterrows(), 1):
        index, hotel_data = row
        title = hotel_data['Title']
        url = hotel_data['Url']
        st.write(f"{i}. {title}")
        st.write(f"   URL: {url}")

if selected_tab == "Recommendations":
    st.sidebar.header("User Preferences")
    unique_locations = data['Location'].unique()
    user_location = st.sidebar.selectbox("Location:", unique_locations)
    user_score = st.sidebar.slider("Minimum Score:", 0, 10, 7)
    user_stars = st.sidebar.slider("Minimum Stars:", 0, 5, 4)

    create_recommendations(user_location, user_score, user_stars)

elif selected_tab == "Other Tab":
    st.write("This is another tab.")

    import plotly.express as px
    
    # Ваши данные о бронированиях (предположим, что вы уже их имеете)
    your_hotel_data = pd.DataFrame({
        'date': ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'],
        'bookings': [10, 15, 8, 12, 18]
    })
    
    # Данные о рынке гостиничных бронирований (пример)
    market_data = pd.DataFrame({
        'date': ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'],
        'bookings': [50, 60, 55, 58, 62]  # Пример данных о рынке
    })
    
    # Преобразуем столбец с датами в формат datetime
    your_hotel_data['date'] = pd.to_datetime(your_hotel_data['date'])
    market_data['date'] = pd.to_datetime(market_data['date'])
    
    # Группируем данные по дням и вычисляем сумму бронирований
    your_hotel_grouped = your_hotel_data.groupby('date')['bookings'].sum().reset_index()
    market_grouped = market_data.groupby('date')['bookings'].sum().reset_index()
    
   
    merged_data = pd.merge(your_hotel_grouped, market_grouped, on='date', suffixes=('_your_hotel', '_market'))
    merged_data['percentage_difference'] = ((merged_data['bookings_your_hotel'] - merged_data['bookings_market']) / merged_data['bookings_market']) * 100
    
   


    selected_columns = st.multiselect("Выберите колонки для отображения:", merged_data.columns)

    # Отображение данных в зависимости от выбранных колонок
    if selected_columns:
        st.write(merged_data[selected_columns])
    else:
        st.write("Выберите одну или несколько колонок для отображения.")

    
    if selected_columns:
        fig = px.line(merged_data, x='date', y=selected_columns, title="График темпов бронирования")  
        st.plotly_chart(fig)
    else:
        st.write("Выберите одну или несколько колонок для отображения.")


# Display content based on selected tab
if selected_tab == "Data":
    st.title("Data Tab")
    st.write("This is the Data tab content.")

    # Load data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Add filter widget
        grouping_column = st.selectbox("Select Grouping Column", df.columns)

        # Group data based on user choice and count unique rows in each group
        grouped_data = df.groupby(grouping_column)[grouping_column].count()

        # Display grouped data
        st.write("Grouped Data:")
        st.dataframe(grouped_data)

elif selected_tab == "Analytics":
    st.title("Analytics Tab")
    st.write("This is the Analytics tab content.")

    # analytics code
    import pandas as pd

    # Define the Excel file path
    # excel_file = r'C:\Users\theli\Downloads\ОТЧЕТ 2020.xlsx'
    excel_file = 'ОТЧЕТ 2023г (1) (1).xlsx'


    # List of sheet names to process
    sheet_names = [0, 1, 2, 3, 4, 5, 6, 7]


    st.write("Sheet_names:")
    st.dataframe(sheet_names)
    # Define the service column
    service_column = 14

    # List of names to exclude
    excluded_names = ['Грибина В.Э.', 'Сидорова И.М.', 'Сидорова И.М', 'Васильева Н.А.', 'Остаток на начало дня, руб',
                      'Администратор Мамаева Т.М.']

    # Initialize data structures to store total spending across all sheets
    total_spending_across_sheets = 0.0

    # Initialize data structures to store number of unique clients and spending across all sheets
    all_unique_clients = []
    all_sheet_spending = []

    # Iterate through sheet names
    for sheet_name in sheet_names:
        df = pd.read_excel(excel_file, sheet_name, header=None)

        # Data cleaning and preprocessing (same as your original code)
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        df.columns = df.iloc[0]
        df = df[1:]

        unique_customers = set()
        customer_spending = {}

        # Iterate through rows and calculate spending for the specified service
        for index, row in df.iterrows():
            customer_name = row[1]
            if (
                    isinstance(customer_name, str)  # Check if it's a string
                    and pd.notna(customer_name)
                    and customer_name.strip() not in excluded_names
                    and not any(char.isdigit() for char in str(customer_name))
            ):
                customer_name = customer_name.strip()  # Remove leading/trailing whitespace
                unique_customers.add(customer_name)
                spending = row[service_column]

                if isinstance(spending, pd.Series):
                    spending = spending.sum()
                if pd.notna(spending) and isinstance(spending, (float, int)):
                    cleaned_spending = float(spending)
                    if customer_name not in customer_spending:
                        customer_spending[customer_name] = cleaned_spending
                    else:
                        customer_spending[customer_name] += cleaned_spending

        # Store number of unique clients and spending for the sheet
        sheet_unique_clients = len(unique_customers)
        sheet_spending = sum(customer_spending.values())

        # Append values to respective lists
        all_unique_clients.append((sheet_name, sheet_unique_clients))
        all_sheet_spending.append((sheet_name, sheet_spending))

        # Accumulate total spending across all sheets
        total_spending_across_sheets += sheet_spending

    # Convert unique clients list to DataFrame
    unique_clients_df2 = pd.DataFrame(all_unique_clients, columns=['Month', 'UniqueClients'])

    # Convert sheet spending list to DataFrame
    sheet_spending_df = pd.DataFrame(all_sheet_spending, columns=['Month', 'TotalSpending'])
    sheet_spending_df['avg. revenue'] = sheet_spending_df['TotalSpending'] / unique_clients_df2['UniqueClients']
    sheet_spending_df['Difference'] = sheet_spending_df['TotalSpending'].diff(1).fillna(0)

    # Print the unique clients and sheet spending DataFrames
    st.write("Number of Unique Clients for Each Sheet:")
    st.dataframe(unique_clients_df2)

    st.write("Total Spending for Each Sheet:")
    st.dataframe(sheet_spending_df)

    st.write("Total spending across all sheets:", total_spending_across_sheets)
    st.write("Средний доход на клиента:", total_spending_across_sheets / sum(unique_clients_df2['UniqueClients']))

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime, timedelta
    from pmdarima import auto_arima


    # Display results in the Streamlit app
    st.write("Number of Unique Clients for Each Sheet:")
    st.dataframe(unique_clients_df2)

    st.write("Total Spending for Each Sheet:")
    st.dataframe(sheet_spending_df)

    st.write("Total spending across all sheets:", total_spending_across_sheets)
    st.write("Средний доход на клиента:", total_spending_across_sheets / sum(unique_clients_df2['UniqueClients']))

    # Visualize the unique clients data
    st.write("Visualizing Unique Clients Data:")

    # Convert 'Month' column to datetime format
    unique_clients_df2['Month'] = pd.to_datetime(unique_clients_df2['Month'])

    # Set 'Month' column as index
    unique_clients_df2.set_index('Month', inplace=True)

    # Plot unique clients data
    plt.figure(figsize=(10, 6))
    plt.plot(unique_clients_df2, marker='o')
    plt.title("Number of Unique Clients Over Time")
    plt.xlabel("Month")
    plt.ylabel("Unique Clients")
    st.pyplot(plt)


    # Fit ARIMA model
    st.write("ARIMA Prediction for Next 3 Months:")

    # Fit ARIMA model
    model = auto_arima(unique_clients_df2, seasonal=False, stepwise=True, suppress_warnings=True)

    # Filter data and predictions for years 2023 and 2024
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2024-12-31")
    filtered_unique_clients_df = unique_clients_df2[
        (unique_clients_df2.index >= start_date) & (unique_clients_df2.index <= end_date)]

    # Slider for selecting the number of future steps
    num_steps = st.slider("Select Number of Future Months", min_value=1, max_value=3, value=3)

    # Generate future dates starting from January 2023
    last_date = filtered_unique_clients_df.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_steps + 1)]

    # Predictions
    predictions, conf_int = model.predict(n_periods=num_steps, return_conf_int=True)

    # Create a DataFrame for predictions
    future_df = pd.DataFrame({'Predicted': predictions, 'Lower_CI': conf_int[:, 0], 'Upper_CI': conf_int[:, 1]},
                             index=future_dates)

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_unique_clients_df, label='Actual', marker='o')
    plt.plot(future_df['Predicted'], label='Predicted', linestyle='dashed', marker='o')
    plt.fill_between(future_df.index, future_df['Lower_CI'], future_df['Upper_CI'], color='gray', alpha=0.2)
    plt.title("ARIMA Prediction for Unique Clients (2023-2024)")
    plt.xlabel("Month")
    plt.ylabel("Unique Clients")
    plt.legend()
    st.pyplot(plt)



elif selected_tab == "Forecast":
    st.title("Forecast Tab")
    st.write("This is the Forecast tab content.")

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_squared_error

    import warnings

    warnings.filterwarnings('ignore')





    # Load foot traffic data
    data = pd.read_csv("foot_traffic.csv")

    results = adfuller(data['foot_traffic'])
    st.write('Statistic:', results[0])
    st.write('P - value:', results[1])

    alpha = 0.05


    def ad_test(data):
        results = adfuller(data)
        if results[1] > alpha:
            st.write('Time series is not stationary')
        else:
            st.write('We have stationary time series')
            st.write(results[1])


    ad_test(data['foot_traffic'])

    foot_traffic_diff = np.diff(data['foot_traffic'], n=1)
    ad_test(foot_traffic_diff)

    plot_acf(foot_traffic_diff, lags=20)

    plot_pacf(foot_traffic_diff, lags=20)
    plt.tight_layout()

    df_diff = pd.DataFrame({'foot_wear_diff': foot_traffic_diff})

    train = df_diff[:-52]
    test = df_diff[-52:]

    TRAIN_LEN = len(train)
    HORIZON = len(test)
    WINDOW = st.slider("Select Window Size", min_value=1, max_value=5, value=1)


    def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int,
                         window: int, method: str) -> list:
        total_len = train_len + horizon
        end_idx = train_len

        if method == 'mean':
            pred_mean = []

            for i in range(train_len, total_len, window):
                mean = np.mean(df[:i].values)
                pred_mean.extend(mean for _ in range(window))

            return pred_mean

        elif method == 'last':
            pred_last_value = []
            for i in range(train_len, total_len, window):
                last_value = df[:i].iloc[-1].values[0]
                pred_last_value.extend(last_value for _ in range(window))

            return pred_last_value

        elif method == 'AR':
            pred_AR = []
            for i in range(train_len, total_len, window):
                model = SARIMAX(df[:i], order=(3, 0, 0))
                res = model.fit(disp=False)
                predictions = res.get_prediction(0, i + window - 1)
                oos = predictions.predicted_mean.iloc[-window:]
                pred_AR.extend(oos)

            return pred_AR


    pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
    pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
    pred_AR = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'AR')

    test['pred_mean'] = pred_mean
    test['pred_last_value'] = pred_last_value
    test['pred_AR'] = pred_AR


    st.write("Forecast Results:")
    st.dataframe(test)

    fig, ax = plt.subplots()
    ax.plot(df_diff['foot_wear_diff'])
    ax.plot(test['foot_wear_diff'], 'b-', label='actual')
    ax.plot(test['pred_mean'], 'g:', label='mean')
    ax.plot(test['pred_last_value'], 'r-.', label='last')
    ax.plot(test['pred_AR'], 'k--', label='AR(3)')

    ax.legend(loc=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Diff. avg. weekly foot traffic')
    ax.axvspan(947, 998, color='#808080', alpha=0.2)
    ax.set_xlim(920, 999)
    ax.set_xticks([936, 988])
    ax.set_xticklabels([2018, 2019])
    fig.autofmt_xdate()
    plt.tight_layout()

    st.pyplot(fig)

    mean_mse = mean_squared_error(test['foot_wear_diff'], test['pred_mean'])
    last_mse = mean_squared_error(test['foot_wear_diff'], test['pred_last_value'])
    ar_mse = mean_squared_error(test['foot_wear_diff'], test['pred_AR'])

    st.write("Mean Squared Errors:")
    st.write("Mean Forecast:", mean_mse)
    st.write("Last Value Forecast:", last_mse)
    st.write("AR(3) Forecast:", ar_mse)


elif selected_tab == "Download":
    st.title("Download Tab")
    st.write("This is the Download tab content.")
    # Add download link or button here


