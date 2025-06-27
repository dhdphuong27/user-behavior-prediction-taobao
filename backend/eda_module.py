import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import base64
import networkx as nx

def perform_eda_for_web(df):
    """
    Performs Exploratory Data Analysis (EDA) on the user behavior DataFrame,
    capturing all printed output as a string and all generated plots as
    base64 encoded images with data-driven descriptions.

    Args:
        df (pd.DataFrame): The input DataFrame containing user behavior data.

    Returns:
        tuple: A tuple containing:
            - str: All printed output captured during EDA.
            - list: A list of base64 encoded PNG images of the generated plots.
            - list: A list of data-driven descriptions for each corresponding graph.
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    all_plots = []
    plot_descriptions = []

    try:
        print("First few rows:\n", df.head().to_string())
        print("\nDataframe info:")
        df.info()
        print("\nDescriptive statistics:")
        print(df.describe(include='all').to_string()) 

        # Perform one-hot encoding
        df_eda = df.copy()
        behavior_encoded = pd.get_dummies(df_eda['Behavior'], prefix='Behavior')
        df_eda = pd.concat([df_eda, behavior_encoded], axis=1)
        print("\nDataFrame after one-hot encoding:")

        # Univariate Analysis - Categorical Variables (Day of Week)
        fig_day_of_week, ax_day_of_week = plt.subplots(figsize=(10, 6))
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df_eda['Day_of_Week'].value_counts()
        peak_day = day_counts.idxmax()
        lowest_day = day_counts.idxmin()
        
        sns.countplot(data=df_eda, x='Day_of_Week', hue='Day_of_Week', order=order, palette='magma', ax=ax_day_of_week, legend=False)
        ax_day_of_week.set_title('Distribution by Day of Week')
        ax_day_of_week.set_xlabel('Day of Week')
        ax_day_of_week.set_ylabel('Count')
        plt.tight_layout()
        img_buffer_day_of_week = io.BytesIO()
        fig_day_of_week.savefig(img_buffer_day_of_week, format='png', bbox_inches='tight')
        img_buffer_day_of_week.seek(0)
        all_plots.append(["Distribution by Day of Week", base64.b64encode(img_buffer_day_of_week.getvalue()).decode('utf-8')])
        
        # Data-driven description
        weekday_total = day_counts[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].sum()
        weekend_total = day_counts[['Saturday', 'Sunday']].sum()
        plot_descriptions.append(f"User activity peaks on {peak_day} ({day_counts[peak_day]:,} actions) and is lowest on {lowest_day} ({day_counts[lowest_day]:,} actions). Weekdays account for {weekday_total:,} actions ({weekday_total/(weekday_total+weekend_total)*100:.1f}%) while weekends contribute {weekend_total:,} actions ({weekend_total/(weekday_total+weekend_total)*100:.1f}%). This suggests {'higher weekday engagement' if weekday_total > weekend_total else 'balanced week-round activity'}, indicating the platform is primarily used {'for work-related browsing' if weekday_total > weekend_total*1.5 else 'consistently throughout the week'}.")
        plt.close(fig_day_of_week)

        # Univariate Analysis - Numeric Variables (Hour of Day)
        fig_hour_of_day, ax_hour_of_day = plt.subplots(figsize=(12, 6))
        hour_counts = df_eda['Hour_of_Day'].value_counts().sort_index()
        peak_hour = hour_counts.idxmax()
        lowest_hour = hour_counts.idxmin()
        
        sns.countplot(data=df_eda, x='Hour_of_Day', hue='Hour_of_Day', palette='coolwarm', order=sorted(df_eda['Hour_of_Day'].unique()), legend=False, ax=ax_hour_of_day)
        ax_hour_of_day.set_title('Behavior Counts by Hour of Day')
        ax_hour_of_day.set_xlabel('Hour of Day')
        ax_hour_of_day.set_ylabel('Count')
        plt.tight_layout()
        img_buffer_hour_of_day = io.BytesIO()
        fig_hour_of_day.savefig(img_buffer_hour_of_day, format='png', bbox_inches='tight')
        img_buffer_hour_of_day.seek(0)
        all_plots.append(["Behavior Counts by Hour of Day", base64.b64encode(img_buffer_hour_of_day.getvalue()).decode('utf-8')])
        
        # Data-driven description
        morning_hours = hour_counts[6:12].sum()  # 6 AM to 12 PM
        afternoon_hours = hour_counts[12:18].sum()  # 12 PM to 6 PM
        evening_hours = hour_counts[18:24].sum()  # 6 PM to 12 AM
        night_hours = hour_counts[0:6].sum()  # 12 AM to 6 AM
        
        peak_period = max([('morning', morning_hours), ('afternoon', afternoon_hours), ('evening', evening_hours), ('night', night_hours)], key=lambda x: x[1])
        
        plot_descriptions.append(f"Peak activity occurs at {peak_hour}:00 ({hour_counts[peak_hour]:,} actions) while the quietest hour is {lowest_hour}:00 ({hour_counts[lowest_hour]:,} actions). The {peak_period[0]} period (6-hour blocks) shows highest engagement with {peak_period[1]:,} total actions. {'Late night activity is minimal' if night_hours < morning_hours/2 else 'Significant late-night usage detected'}, suggesting {'standard business hours usage patterns' if peak_hour in range(9, 18) else 'non-traditional usage patterns'}. Optimal times for system maintenance would be around {lowest_hour}:00-{(lowest_hour+2)%24}:00.")
        plt.close(fig_hour_of_day)

        print("Unique users:", df_eda['User_ID'].nunique())
        print("Unique products:", df_eda['Product_ID'].nunique())
        print("Unique categories:", df_eda['Category_ID'].nunique())

        # Time Series Analysis
        df_eda['Date'] = pd.to_datetime(df_eda['Date'])
        daily_counts = df_eda.groupby('Date').size()

        start_date = pd.to_datetime('2017-11-22')
        end_date = pd.to_datetime('2018-01-06')
        filtered_daily_counts = daily_counts[(daily_counts.index >= start_date) & (daily_counts.index <= end_date)]

        peak_date = filtered_daily_counts.idxmax()
        lowest_date = filtered_daily_counts.idxmin()
        avg_daily = filtered_daily_counts.mean()

        fig_daily_counts, ax_daily_counts = plt.subplots(figsize=(12, 6))
        filtered_daily_counts.plot(marker='o', ax=ax_daily_counts)
        ax_daily_counts.set_title('Daily Count of User Actions')
        ax_daily_counts.set_xlabel('Date')
        ax_daily_counts.set_ylabel('Count')
        plt.tight_layout()
        img_buffer_daily_counts = io.BytesIO()
        fig_daily_counts.savefig(img_buffer_daily_counts, format='png', bbox_inches='tight')
        img_buffer_daily_counts.seek(0)
        all_plots.append(["Daily Count of User Actions", base64.b64encode(img_buffer_daily_counts.getvalue()).decode('utf-8')])
        
        # Data-driven description
        trend_direction = "increasing" if filtered_daily_counts.iloc[-5:].mean() > filtered_daily_counts.iloc[:5].mean() else "decreasing"
        volatility = filtered_daily_counts.std() / avg_daily
        
        plot_descriptions.append(f"Daily activity ranges from {filtered_daily_counts.min():,} to {filtered_daily_counts.max():,} actions, with an average of {avg_daily:.0f} daily actions. Peak activity occurred on {peak_date.strftime('%B %d, %Y')} ({filtered_daily_counts[peak_date]:,} actions), while the lowest was on {lowest_date.strftime('%B %d, %Y')} ({filtered_daily_counts[lowest_date]:,} actions). The data shows a {trend_direction} trend over time with {'high volatility' if volatility > 0.3 else 'moderate volatility'} (CV: {volatility:.2f}). {'Significant seasonal patterns may be present' if filtered_daily_counts.max() > avg_daily * 2 else 'Activity levels are relatively stable'}.")
        plt.close(fig_daily_counts)

        # Group Analysis - Top 10 Products
        top_products = df_eda['Product_ID'].value_counts().head(10)
        top_products = top_products.sort_values(ascending=False)

        fig_top_products, ax_top_products = plt.subplots(figsize=(12, 7))
        sns.barplot(x=top_products.index,
                    y=top_products.values,
                    hue=top_products.index,
                    palette='Blues_d',
                    ax=ax_top_products,
                    legend=False,
                    order=top_products.index)
        ax_top_products.set_title('Top 10 Products by Engagement')
        ax_top_products.set_xlabel('Product ID')
        ax_top_products.set_ylabel('Engagement Count')
        plt.tight_layout()
        img_buffer_top_products = io.BytesIO()
        fig_top_products.savefig(img_buffer_top_products, format='png', bbox_inches='tight')
        img_buffer_top_products.seek(0)
        all_plots.append(["Top 10 Products by Engagement", base64.b64encode(img_buffer_top_products.getvalue()).decode('utf-8')])
        
        # Data-driven description
        top_product_id = top_products.index[0]
        top_product_count = top_products.iloc[0]
        total_product_interactions = df_eda['Product_ID'].value_counts().sum()
        top_10_share = top_products.sum() / total_product_interactions * 100
        concentration_ratio = top_products.iloc[0] / top_products.iloc[-1]
        
        plot_descriptions.append(f"Product {top_product_id} dominates with {top_product_count:,} interactions, representing {top_product_count/total_product_interactions*100:.1f}% of all product activity. The top 10 products account for {top_10_share:.1f}% of total engagement, indicating {'high concentration' if top_10_share > 50 else 'moderate concentration'} in user preferences. The engagement ratio between #1 and #10 products is {concentration_ratio:.1f}:1, suggesting {'a clear winner with long-tail distribution' if concentration_ratio > 3 else 'relatively balanced popularity among top products'}. This suggests focusing marketing efforts on these high-performing products could yield significant returns.")
        plt.close(fig_top_products)

        # Group Analysis - Top 10 Categories
        top_categories = df_eda['Category_ID'].value_counts().head(10)
        fig_top_categories, ax_top_categories = plt.subplots(figsize=(12, 7))
        sns.barplot(x=top_categories.index,
                    y=top_categories.values,
                    hue=top_categories.index,
                    palette='Greens_d',
                    ax=ax_top_categories,
                    legend=False,
                    order=top_categories.index)
        ax_top_categories.set_title('Top 10 Categories by Engagement')
        ax_top_categories.set_xlabel('Category ID')
        ax_top_categories.set_ylabel('Engagement Count')
        plt.tight_layout()
        img_buffer_top_categories = io.BytesIO()
        fig_top_categories.savefig(img_buffer_top_categories, format='png', bbox_inches='tight')
        img_buffer_top_categories.seek(0)
        all_plots.append(["Top 10 Categories by Engagement", base64.b64encode(img_buffer_top_categories.getvalue()).decode('utf-8')])
        
        # Data-driven description
        top_category_id = top_categories.index[0]
        top_category_count = top_categories.iloc[0]
        total_category_interactions = df_eda['Category_ID'].value_counts().sum()
        top_10_category_share = top_categories.sum() / total_category_interactions * 100
        category_concentration = top_categories.iloc[0] / top_categories.iloc[-1]
        
        plot_descriptions.append(f"Category {top_category_id} leads with {top_category_count:,} interactions ({top_category_count/total_category_interactions*100:.1f}% of all activity). The top 10 categories capture {top_10_category_share:.1f}% of user engagement, indicating {'strong category preferences' if top_10_category_share > 60 else 'diverse category interests'}. The leader-to-10th ratio of {category_concentration:.1f}:1 shows {'dominant category preferences' if category_concentration > 4 else 'balanced category distribution'}. This pattern suggests {'niche market focus' if top_10_category_share > 70 else 'broad market appeal'} and indicates where inventory and marketing resources should be concentrated.")
        plt.close(fig_top_categories)

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # Heatmap of User Actions by Hour and Day
        df_eda['Day_of_Week'] = pd.Categorical(df_eda['Day_of_Week'], categories=day_order, ordered=True)
        pivot_table_actions = df_eda.pivot_table(index='Hour_of_Day', columns='Day_of_Week', values='User_ID', aggfunc='count')
        pivot_table_actions = pivot_table_actions.reindex(columns=day_order)

        # Find peak activity time
        peak_hour_day = pivot_table_actions.stack().idxmax()
        peak_value = pivot_table_actions.stack().max()
        lowest_hour_day = pivot_table_actions.stack().idxmin()
        lowest_value = pivot_table_actions.stack().min()

        fig_heatmap_actions, ax_heatmap_actions = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot_table_actions, annot=True, fmt=".0f", cmap='YlGnBu', ax=ax_heatmap_actions)
        ax_heatmap_actions.set_title('Heatmap of User Actions by Hour and Day of Week')
        ax_heatmap_actions.set_xlabel('Day of Week')
        ax_heatmap_actions.set_ylabel('Hour of Day')
        plt.tight_layout()
        img_buffer_heatmap_actions = io.BytesIO()
        fig_heatmap_actions.savefig(img_buffer_heatmap_actions, format='png', bbox_inches='tight')
        img_buffer_heatmap_actions.seek(0)
        all_plots.append(["Heatmap of User Actions by Hour and Day of Week", base64.b64encode(img_buffer_heatmap_actions.getvalue()).decode('utf-8')])
        
        # Data-driven description
        weekend_evening_activity = pivot_table_actions.loc[18:22, ['Saturday', 'Sunday']].mean().mean()
        weekday_business_activity = pivot_table_actions.loc[9:17, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean().mean()
        
        plot_descriptions.append(f"Peak activity occurs on {peak_hour_day[1]} at {peak_hour_day[0]}:00 with {peak_value:.0f} actions, while the quietest period is {lowest_hour_day[1]} at {lowest_hour_day[0]}:00 with {lowest_value:.0f} actions. {'Business hours show strong activity patterns' if weekday_business_activity > weekend_evening_activity else 'Evening and weekend usage is prominent'}, with weekday business hours averaging {weekday_business_activity:.0f} actions vs weekend evenings at {weekend_evening_activity:.0f} actions. This reveals {'professional/work-related usage patterns' if weekday_business_activity > weekend_evening_activity * 1.5 else 'consumer-oriented usage patterns'} and suggests optimal campaign timing and customer service staffing requirements.")
        plt.close(fig_heatmap_actions)

        # Heatmap of User Buys by Hour and Day
        buy_df = df_eda[df_eda['Behavior'] == 'Buy'].copy()
        
        if not buy_df.empty:
            buy_df['Day_of_Week'] = pd.Categorical(buy_df['Day_of_Week'], categories=day_order, ordered=True)
            pivot_table_buys = buy_df.pivot_table(index='Hour_of_Day', columns='Day_of_Week', values='User_ID', aggfunc='count')
            pivot_table_buys = pivot_table_buys.reindex(columns=day_order)
            
            buy_peak_hour_day = pivot_table_buys.stack().idxmax()
            buy_peak_value = pivot_table_buys.stack().max()
            total_buys = buy_df.shape[0]
            
            weekend_buys = pivot_table_buys[['Saturday', 'Sunday']].sum().sum()
            weekday_buys = pivot_table_buys[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].sum().sum()

            fig_heatmap_buys, ax_heatmap_buys = plt.subplots(figsize=(12, 7))
            sns.heatmap(pivot_table_buys, annot=True, fmt=".0f", cmap='YlGnBu', ax=ax_heatmap_buys)
            ax_heatmap_buys.set_title('Heatmap of User Purchases by Hour and Day of Week')
            ax_heatmap_buys.set_xlabel('Day of Week')
            ax_heatmap_buys.set_ylabel('Hour of Day')
            plt.tight_layout()
            img_buffer_heatmap_buys = io.BytesIO()
            fig_heatmap_buys.savefig(img_buffer_heatmap_buys, format='png', bbox_inches='tight')
            img_buffer_heatmap_buys.seek(0)
            all_plots.append(["Heatmap of User Purchases by Hour and Day of Week", base64.b64encode(img_buffer_heatmap_buys.getvalue()).decode('utf-8')])
            
            # Data-driven description
            plot_descriptions.append(f"Purchase behavior peaks on {buy_peak_hour_day[1]} at {buy_peak_hour_day[0]}:00 with {buy_peak_value:.0f} purchases from {total_buys:,} total purchases analyzed. {'Weekend purchases dominate' if weekend_buys > weekday_buys else 'Weekday purchases are more common'} ({weekend_buys:,} weekend vs {weekday_buys:,} weekday purchases). {'Purchase patterns differ significantly from general browsing' if buy_peak_hour_day != peak_hour_day else 'Purchase timing aligns with general activity patterns'}, indicating {'impulse buying during leisure time' if weekend_buys > weekday_buys else 'planned purchasing during business hours'}. Sales campaigns should target {buy_peak_hour_day[1]} around {buy_peak_hour_day[0]}:00 for maximum conversion impact.")
            plt.close(fig_heatmap_buys)
        else:
            plot_descriptions.append("No purchase data available for analysis - this suggests the dataset may be focused on browsing behavior rather than transactional data.")

        # User-based analysis
        df_user = df_eda.groupby(by='User_ID')
        user_behavior = df_user["Behavior"].value_counts().unstack().fillna(0) 

        # Repurchase rate
        if 'Buy' in user_behavior.columns:
            repurchase_users = user_behavior[user_behavior['Buy'] >= 2]
            repurchase_users_count = repurchase_users.shape[0]
            user_bought_count = df_eda[df_eda['Behavior']=='Buy']["User_ID"].nunique()
            if user_bought_count > 0:
                repurchase_rate = repurchase_users_count / user_bought_count
                print("\nRepurchase rate:{: .2f}%".format(repurchase_rate * 100))
            else:
                print("\nNo users with 'Buy' behavior found, cannot calculate repurchase rate.")
        else:
            print("\n'Buy' behavior column not found in user_behavior summary.")

        # Category correlation
        df_time_cat = df_eda[['User_ID', 'Category_ID', 'Timestamp']].copy()
        df_time_cat['Datetime'] = pd.to_datetime(df_time_cat['Timestamp'], unit='s')
        df_time_cat = df_time_cat.sort_values(by=['User_ID', 'Datetime'])

        df_time_cat['time_diff'] = df_time_cat.groupby('User_ID')['Datetime'].diff()
        df_time_cat['previous_category'] = df_time_cat['Category_ID'].shift(1)
        df_time_cat['previous_user'] = df_time_cat['User_ID'].shift(1)

        time_threshold = pd.Timedelta(minutes=5)
        close_interactions = df_time_cat[
            (df_time_cat['time_diff'] <= time_threshold) &
            (df_time_cat['User_ID'] == df_time_cat['previous_user'])
        ].copy()

        category_pair_counts = close_interactions.groupby(['previous_category', 'Category_ID']).size().reset_index(name='count')
        category_pair_counts = category_pair_counts[category_pair_counts['previous_category'] != category_pair_counts['Category_ID']]

        # Create a directed graph
        G = nx.DiGraph()
        top_n_pairs = 50
        top_category_pairs = category_pair_counts.sort_values(by='count', ascending=False).head(top_n_pairs)

        for index, row in top_category_pairs.iterrows():
            source_category = int(row['previous_category'])
            target_category = int(row['Category_ID'])
            count = row['count']
            G.add_edge(source_category, target_category, weight=count)

        # Calculate network metrics
        most_connected_category = max(G.nodes(), key=lambda x: G.degree(x)) if G.nodes() else None
        strongest_transition = top_category_pairs.iloc[0] if not top_category_pairs.empty else None
        total_transitions = top_category_pairs['count'].sum()

        fig_graph, ax_graph = plt.subplots(figsize=(14, 12))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.9, ax=ax_graph)

        edges = G.edges(data=True)
        weights = [d['weight'] for (u, v, d) in edges]
        if weights:  # Check if weights list is not empty
            max_weight = max(weights)
            nx.draw_networkx_edges(
                G, pos, edgelist=edges,
                width=[w / max_weight * 10 if max_weight > 0 else 1 for w in weights],
                edge_color='gray', alpha=0.6, ax=ax_graph
            )

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax_graph)
        ax_graph.set_title(f'Top {top_n_pairs} Most Frequent Consecutive Category Transitions (within 5 minutes)')
        ax_graph.axis('off')

        img_buffer_graph = io.BytesIO()
        fig_graph.savefig(img_buffer_graph, format='png', bbox_inches='tight')
        img_buffer_graph.seek(0)
        all_plots.append([
            f"Top {top_n_pairs} Most Frequent Consecutive Category Transitions (within 5 minutes)",
            base64.b64encode(img_buffer_graph.getvalue()).decode('utf-8')
        ])
        
        # Data-driven description
        if strongest_transition is not None and most_connected_category is not None:
            plot_descriptions.append(f"Users most frequently transition from Category {strongest_transition['previous_category']} to Category {strongest_transition['Category_ID']} ({strongest_transition['count']} times within 5-minute windows). Category {most_connected_category} shows the highest connectivity with {G.degree(most_connected_category)} connections, suggesting it's a central hub in user navigation. The network reveals {total_transitions:,} rapid category transitions, indicating {'highly exploratory browsing behavior' if total_transitions > 1000 else 'focused browsing patterns'}. These pathways represent cross-selling opportunities and suggest optimal product placement strategies for increasing basket size and user engagement.")
        else:
            plot_descriptions.append("Limited category transition data available - users tend to stay within single categories during their sessions, suggesting focused shopping behavior rather than exploratory browsing.")
        plt.close(fig_graph)

        # Calculate conversion rate for each category
        conversion = df_eda[df_eda['Behavior'].isin(['Buy', 'PageView'])].copy()
        conversion_pb = conversion.groupby('Category_ID')['Behavior'].value_counts().unstack().fillna(0)
        conversion_pb['Conversion_Rate'] = conversion_pb.apply(
            lambda row: row['Buy'] / row['PageView'] if row['PageView'] > 0 else 0, axis=1
        )

        conversion_10 = conversion_pb.sort_values(by='Buy', ascending=False)[:10]
        print("\nConversion rate of top 10 selling categories:")
        print(conversion_10.to_string())

        # Calculate correlation and insights
        correlation_coef = conversion_pb['Buy'].corr(conversion_pb['Conversion_Rate'])
        avg_conversion_rate = conversion_pb['Conversion_Rate'].mean()
        highest_converting_category = conversion_pb.loc[conversion_pb['Conversion_Rate'].idxmax()]
        
        fig_conversion_rate, ax_conversion_rate = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Buy', y='Conversion_Rate', data=conversion_pb, scatter_kws={'alpha':0.5}, ax=ax_conversion_rate)
        ax_conversion_rate.set_title('Correlation between Buy Count and Conversion Rate')
        ax_conversion_rate.set_xlabel('Buy Count')
        ax_conversion_rate.set_ylabel('Conversion Rate')
        ax_conversion_rate.grid(True)
        plt.tight_layout()
        img_buffer_conversion_rate = io.BytesIO()
        fig_conversion_rate.savefig(img_buffer_conversion_rate, format='png', bbox_inches='tight')
        img_buffer_conversion_rate.seek(0)
        all_plots.append(["Correlation between Buy Count and Conversion Rate", base64.b64encode(img_buffer_conversion_rate.getvalue()).decode('utf-8')])
        
        # Data-driven description
        correlation_strength = "strong" if abs(correlation_coef) > 0.7 else "moderate" if abs(correlation_coef) > 0.3 else "weak"
        correlation_direction = "positive" if correlation_coef > 0 else "negative"
        
        plot_descriptions.append(f"The correlation between purchase volume and conversion rate is {correlation_strength} and {correlation_direction} (r={correlation_coef:.3f}). Average conversion rate across categories is {avg_conversion_rate:.1%}, with the highest-converting category achieving {highest_converting_category['Conversion_Rate']:.1%}. {'High-volume categories maintain strong conversion rates' if correlation_coef > 0.3 else 'Popular categories may suffer from lower conversion due to browsing behavior' if correlation_coef < -0.3 else 'Conversion rates are independent of popularity'}, suggesting {'economies of scale in sales effectiveness' if correlation_coef > 0.3 else 'need for targeted conversion optimization in popular categories' if correlation_coef < -0.3 else 'category-specific factors drive conversion more than popularity'}.")
        plt.close(fig_conversion_rate)

    finally:
        sys.stdout = old_stdout

    return redirected_output.getvalue(), all_plots, plot_descriptions