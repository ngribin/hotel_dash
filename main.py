import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.express as px
from lightfm import LightFM
import numpy as np
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

st.set_page_config(layout="wide")
pio.templates.default = "plotly_white"


st.title("Отель - 32 номера")

total_rooms = 32

def load_data():
    try:
        data = pd.read_csv('Radio_history.csv')
    except:
        url = 'https://raw.githubusercontent.com/ngribin/hotel_dash/main/Radio_history.csv'
        data = pd.read_csv(url)
    return data

data = load_data()

st.sidebar.title("Меню")
selected_tab = st.sidebar.radio("Вкладка:", ["Загрузка", "Категории", "Темпы продаж", "Прогноз", "Рекомендации"])

if selected_tab == 'Загрузка':
    st.subheader('Загрузка')
    st.write("### Бронирования")
    st.write(data)

    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    occupied_rooms = []

    for date in pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])):
        overlapping_bookings = ((data['arrDate'] < date) & (data['depDate'] > date))
        occupied_rooms_count = overlapping_bookings.sum()
        occupied_rooms.append(occupied_rooms_count)

    # Df
    overlap_df = pd.DataFrame({
        'Date': pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])),
        'Occupancy': occupied_rooms
    })

    st.write("### Номерной фонд 32 номера")
    st.plotly_chart(px.line(overlap_df, x='Date', y='Occupancy', title="Загрузка - номера").update_layout(
        xaxis_title="Дата", yaxis_title="Загрузка"
    ))

    your_hotel_data = pd.DataFrame({
        'date': ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'],
        'bookings': [10, 15, 8, 12, 18]
    })

    st.write("Разница")
    # Данные о рынке гостиничных бронирований 
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
        st.write("")

    
    # if selected_columns:
    #     fig = px.line(merged_data, x='date', y=selected_columns, title="График темпов бронирования")  
    #     st.plotly_chart(fig)
    # else:
    #     st.write("Выберите одну или несколько колонок для отображения.")
    import plotly.graph_objs as go

    if selected_columns:
        data = []
        for col in selected_columns:
            trace = go.Bar(
                x=merged_data['date'],
                y=merged_data[col],
                name=col
            )
            data.append(trace)
    
        layout = go.Layout(
            barmode='group',  
            title="График"
        )
    
        fig = go.Figure(data=data, layout=layout)
    
        st.plotly_chart(fig)
    else:
        st.write("")

    
elif selected_tab == "Категории":
    st.subheader("Категории")


    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    data['roomType_id'].unique()

    room_type_mapping = {
        516: 'Номер "Первой" категории двухместный',
        519: 'Номер "Первой" категории одноместный',
        522: 'Номер "Первой" категории двухместный с отдельными кроватями',
        1136: 'Улучшенный номер "Первой" категории двухместный',
        3074: 'Номер "Высшей" категории с круглой ванной',
        3113: 'Номер "Высшей" категории с хаммамом',
        3115: 'Номер "Высшей" категории люкс семейный',
        3116: 'Номер "Высшей" категории люкс',
    }

    unique_room_types = data['roomType_id'].unique()

    # собираем загрузку для каждой категории
    room_type_occupancy = {room_type_mapping[room_type]: [] for room_type in unique_room_types}

    for date in pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])):
        for room_type in unique_room_types:
            overlapping_bookings = ((data['arrDate'] < pd.Timestamp(date)) & (data['depDate'] > pd.Timestamp(date)) & (
                        data['roomType_id'] == room_type))
            occupied_rooms_count = overlapping_bookings.sum()
            room_type_occupancy[room_type_mapping[room_type]].append(occupied_rooms_count)

    occupancy_df = pd.DataFrame({
        'Date': pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])),
        **room_type_occupancy
    })

    # occupancy_df.columns[1:] - катег. с именами
    fig = px.line(occupancy_df, x='Date', y=occupancy_df.columns[1:], title="Загрузка по категориям")

    st.plotly_chart(fig, use_container_width=True)




elif selected_tab == "Темпы продаж":
    st.subheader("Темпы продаж на 3 дня")

    st.sidebar.subheader("Выбрать даты")
    start_date = st.sidebar.date_input("Start Date (Для 3 дней)", pd.Timestamp('2023-07-28'),
                                       key="start_date_tempo")
    end_date = st.sidebar.date_input("End Date (Для 3 дней)", pd.Timestamp('2023-07-30'), key="end_date_tempo")


    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

   
    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    period_data = data[(data['arrDate'] >= start_date) & (data['arrDate'] <= end_date)]

    booking_totals = {}

    # проходимся по каждому дню
    for date in pd.date_range(start=min(period_data['arrDate']), end=max(period_data['arrDate'])):
        bookings_on_date = period_data[(period_data['arrDate'] <= date) & (period_data['depDate'] >= date)]
        total_bookings = len(bookings_on_date)
        booking_pace_pct = (total_bookings / total_rooms) * 100
        booking_totals[date] = {'Всего бронирований': int(total_bookings), 'Темп продаж': round(booking_pace_pct, 2)}

    d3_booking_pace_df = pd.DataFrame(booking_totals)


    st.write(
        f'Темпы продаж {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}: {booking_pace_pct:.2f}% ')


    #
    st.plotly_chart(
        px.bar(d3_booking_pace_df.T, x=d3_booking_pace_df.T.index, y=['Всего бронирований', 'Темп продаж']).update_layout(xaxis_title="Дата",
                                                                                                   yaxis_title="%"),
        use_container_width=True)
    st.write(d3_booking_pace_df.T)
    st.subheader("Темпы продаж на 14")

    start_date = st.date_input("Start Date", pd.Timestamp('2023-07-28'))
    end_date = st.date_input("End Date", pd.Timestamp('2023-08-10'))

   
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    period_data = data[(data['arrDate'] >= start_date) & (data['arrDate'] <= end_date)]
    booking_totals = {}

    # проходимся по каждому дню
    for date in pd.date_range(start=min(period_data['arrDate']), end=max(period_data['arrDate'])):
        bookings_on_date = period_data[(period_data['arrDate'] <= date) & (period_data['depDate'] >= date)]
        total_bookings = len(bookings_on_date)
        booking_pace_pct = (total_bookings / total_rooms) * 100
        booking_totals[date] = {'Всего бронирований': int(total_bookings), 'Темп продаж': round(booking_pace_pct, 2)}

    d3_booking_pace_df = pd.DataFrame(booking_totals)

    st.write(
        f'Темпы продаж {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}: {booking_pace_pct:.2f}% ')

    #
    st.plotly_chart(
        px.bar(d3_booking_pace_df.T, x=d3_booking_pace_df.T.index,
               y=['Всего бронирований', 'Темп продаж']).update_layout(xaxis_title="Дата",
                                                                      yaxis_title="%"),
        use_container_width=True)
    st.write(d3_booking_pace_df.T)

elif selected_tab == "Прогноз":
    st.subheader("На стадии обучения :)")
    st.title("Forecast")
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
    

elif selected_tab == 'Рекомендации':

    data = pd.read_excel("hotels.xlsx")
    data = data.dropna()

    def create_recommendations(user_location, user_score, user_stars):
        dataset = Dataset()
        dataset.fit(data['Location'], data['Title'])
    
        (interactions, _) = dataset.build_interactions(
            (user, item, 1.0) for user, item in zip(data['Location'], data['Title'])
        )
    
        model = LightFM(loss='warp')
        model.fit(interactions, epochs=30, num_threads=2)
    
        user_id = dataset.mapping()[0][user_location]
    
        filtered_hotels = data[(data['Score'] >= user_score) & (data['Stars'] >= user_stars)]
    
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
    
        st.write(
            f"Топ 5 для гостя в '{user_location}' рейтингом отзывов {user_score} и {user_stars} звезды:")
        # for i, hotel in enumerate(filtered_top_hotels, 1):
        #     st.write(f"{i}. {hotel}")
        for i, row in enumerate(filtered_top_hotels.iterrows(), 1):
            index, hotel_data = row
            title = hotel_data['Title']
            url = hotel_data['Url']
            st.write(f"{i}. {title}")
            st.write(f"   URL: {url}")
    
    st.subheader("Рекомендации")
    st.sidebar.header("User Preferences")
    unique_locations = data['Location'].unique()
    user_location = st.sidebar.selectbox("Location:", unique_locations)
    user_score = st.sidebar.slider("Minimum Score:", 0, 10, 7)
    user_stars = st.sidebar.slider("Minimum Stars:", 0, 5, 4)

    create_recommendations(user_location, user_score, user_stars)
