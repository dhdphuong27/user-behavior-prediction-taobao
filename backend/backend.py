import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import numpy as np

from rfm_kmeans_module import prepare_rfm_and_elbow, perform_kmeans_clustering
from prophet_module import predict_future_hours, update_and_predict
from predictive_module import predict_next
from recommend_module import recommend_products
from eda_module import perform_eda_for_web

import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment

from flask import Flask, request, jsonify, send_file, make_response, render_template
from flask_cors import CORS
import os
import io
import tempfile
import uuid



# Global storage for processed datasets (in production, use Redis or database)
processed_datasets = {}

#packaging everything into functions for server creating
def read_dataset_from_csv(file_paths):
    column_names = ["User_ID", "Product_ID", "Category_ID", "Behavior", "Timestamp"]
    df = pd.read_csv(file_paths, names=column_names)
    return df

def clean_dataset(df):
    print("There are in total " + str(len(df)) + " records in this dataset.")
    if (df.isnull().sum().sum() != 0):
        df_cleaned = df.dropna()
    else:
        df_cleaned = df
    
    # Work with the cleaned dataframe
    df_cleaned = df_cleaned.copy()  # Make sure we're working with a copy
    df_cleaned['Datetime'] = pd.to_datetime(df_cleaned['Timestamp'], unit='s')
    behavior_mapping = {'pv': 'PageView', 'buy': 'Buy', 'cart': 'AddToCart', 'fav': 'Favorite'}
    df_cleaned['Behavior'] = df_cleaned['Behavior'].replace(behavior_mapping)
    df_cleaned['Day_of_Week'] = df_cleaned['Datetime'].dt.day_name()
    df_cleaned['Hour_of_Day'] = df_cleaned['Datetime'].dt.hour
    df_cleaned['Date'] = df_cleaned['Datetime'].dt.date
    return df_cleaned

def paginate_dataframe(df, page, page_size=10):
    """
    Paginate a dataframe
    Returns: (paginated_df, total_pages, current_page, total_records)
    """
    total_records = len(df)
    total_pages = (total_records + page_size - 1) // page_size  # Ceiling division
    
    # Ensure page is within bounds
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_df = df.iloc[start_idx:end_idx]
    
    return paginated_df, total_pages, page, total_records

app = Flask(__name__)

# Configure CORS more explicitly
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.route("/")
def home():
    return "<h1>Flask server for data processing is running!</h1>"

@app.route('/upload_and_process', methods=['OPTIONS'])
def handle_preflight():
    """Handle preflight OPTIONS requests"""
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/upload_and_process', methods=['POST'])
def upload_and_process_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # Create a temporary file using tempfile (cross-platform)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_filepath = temp_file.name
            file.save(temp_filepath)

        try:
            # Read the dataset using your function
            print(f"Reading dataset from: {temp_filepath}")
            df = read_dataset_from_csv(temp_filepath)
            print(f"Dataset loaded with {len(df)} rows")

            # Clean the dataset using your function
            df_cleaned = clean_dataset(df)
            print(f"Dataset cleaned, final shape: {df_cleaned.shape}")

            # Generate unique session ID for this dataset
            session_id = str(uuid.uuid4())
            processed_datasets[session_id] = df_cleaned
            
            # Get first page (10 rows)
            paginated_df, total_pages, current_page, total_records = paginate_dataframe(df_cleaned, 1, 10)

            # Create a CSV in memory
            output = io.StringIO()
            paginated_df.to_csv(output, index=False)
            csv_content = output.getvalue()
            
            # Generate a filename for the processed file
            original_name = os.path.splitext(file.filename)[0]
            processed_filename = f"{original_name}_processed_page1.csv"

            # Return CSV with pagination info in headers
            response = make_response(csv_content)
            response.headers['Content-Type'] = 'text/csv; charset=utf-8'
            response.headers['Content-Disposition'] = f'attachment; filename="{processed_filename}"'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
            response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition,Content-Type,X-Session-ID,X-Total-Pages,X-Current-Page,X-Total-Records'
            response.headers['Content-Length'] = str(len(csv_content.encode('utf-8')))
            
            # Add pagination info to headers
            response.headers['X-Session-ID'] = session_id
            response.headers['X-Total-Pages'] = str(total_pages)
            response.headers['X-Current-Page'] = str(current_page)
            response.headers['X-Total-Records'] = str(total_records)
            
            print(f"Sending processed file: {processed_filename}, size: {len(csv_content)} chars")
            print(f"Session ID: {session_id}, Total pages: {total_pages}")
            return response

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                print(f"Cleaned up temporary file: {temp_filepath}")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Pagination Routes

@app.route('/first_page/<session_id>', methods=['GET', 'OPTIONS'])
def get_first_page(session_id):
    """Get first page of the dataset"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        paginated_df, total_pages, current_page, total_records = paginate_dataframe(df, 1, 10)
        
        # Create CSV response
        output = io.StringIO()
        paginated_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="first_page.csv"'
        response.headers['X-Total-Pages'] = str(total_pages)
        response.headers['X-Current-Page'] = str(current_page)
        response.headers['X-Total-Records'] = str(total_records)
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/last_page/<session_id>', methods=['GET', 'OPTIONS'])
def get_last_page(session_id):
    """Get last page of the dataset"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        total_records = len(df)
        total_pages = (total_records + 9) // 10  # Ceiling division for page size 8
        
        paginated_df, total_pages, current_page, total_records = paginate_dataframe(df, total_pages, 10)
        
        # Create CSV response
        output = io.StringIO()
        paginated_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="last_page.csv"'
        response.headers['X-Total-Pages'] = str(total_pages)
        response.headers['X-Current-Page'] = str(current_page)
        response.headers['X-Total-Records'] = str(total_records)
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/next_page/<session_id>/<int:current_page>', methods=['GET', 'OPTIONS'])
def get_next_page(session_id, current_page):
    """Get next page of the dataset"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        next_page = current_page + 1
        
        paginated_df, total_pages, actual_page, total_records = paginate_dataframe(df, next_page, 10)
        
        # Check if we're at the last page
        if actual_page == current_page:  # No next page available
            return jsonify({"error": "Already at the last page"}), 400
        
        # Create CSV response
        output = io.StringIO()
        paginated_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="page_{actual_page}.csv"'
        response.headers['X-Total-Pages'] = str(total_pages)
        response.headers['X-Current-Page'] = str(actual_page)
        response.headers['X-Total-Records'] = str(total_records)
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/previous_page/<session_id>/<int:current_page>', methods=['GET', 'OPTIONS'])
def get_previous_page(session_id, current_page):
    """Get previous page of the dataset"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        prev_page = current_page - 1
        
        if prev_page < 1:
            return jsonify({"error": "Already at the first page"}), 400
        
        paginated_df, total_pages, actual_page, total_records = paginate_dataframe(df, prev_page, 10)
        
        # Create CSV response
        output = io.StringIO()
        paginated_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="page_{actual_page}.csv"'
        response.headers['X-Total-Pages'] = str(total_pages)
        response.headers['X-Current-Page'] = str(actual_page)
        response.headers['X-Total-Records'] = str(total_records)
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/go_to_page/<session_id>/<int:page_number>', methods=['GET', 'OPTIONS'])
def go_to_page(session_id, page_number):
    """Go to specific page number"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        paginated_df, total_pages, actual_page, total_records = paginate_dataframe(df, page_number, 10)
        
        # Create CSV response
        output = io.StringIO()
        paginated_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="page_{actual_page}.csv"'
        response.headers['X-Total-Pages'] = str(total_pages)
        response.headers['X-Current-Page'] = str(actual_page)
        response.headers['X-Total-Records'] = str(total_records)
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: Get pagination info without downloading CSV
@app.route('/pagination_info/<session_id>', methods=['GET', 'OPTIONS'])
def get_pagination_info(session_id):
    """Get pagination information for a session"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        total_records = len(df)
        total_pages = (total_records + 9) // 10  # Ceiling division for page size 10
        
        return jsonify({
            "session_id": session_id,
            "total_records": total_records,
            "total_pages": total_pages,
            "page_size": 8
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/eda/<session_id>', methods=['GET', 'OPTIONS'])
def eda(session_id):

    df = processed_datasets[session_id] 

    eda_text_output, eda_plots, plot_descriptions = perform_eda_for_web(df) # Call the function

    return jsonify({
        'text_output': eda_text_output,
        'plots': eda_plots,
        'plot_descriptions': plot_descriptions
    })

@app.route('/rfm_elbow/<session_id>', methods=['GET', 'OPTIONS'])
def rfm_elbow_analysis(session_id):
    """Perform RFM analysis and return elbow method graph"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        df = processed_datasets[session_id]
        
        # Check if dataset has 'Buy' behavior
        if 'Buy' not in df['Behavior'].values:
            return jsonify({"error": "No 'Buy' behavior found in dataset. RFM analysis requires purchase data."}), 400
        
        # Perform RFM analysis and elbow method
        rfm_df, wcss, elbow_fig = prepare_rfm_and_elbow(df)
        
        # Convert matplotlib figure to base64 string
        img_buffer = io.BytesIO()
        elbow_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(elbow_fig)  # Close the figure to free memory
        
        # Store RFM data for clustering
        processed_datasets[f"{session_id}_rfm"] = rfm_df
        
        return jsonify({
            'elbow_plot': img_base64,
            'wcss_values': wcss,
            'rfm_summary': {
                'total_customers': len(rfm_df),
                'avg_recency': float(rfm_df['Recency'].mean()),
                'avg_frequency': float(rfm_df['Frequency'].mean()),
                'recency_range': [float(rfm_df['Recency'].min()), float(rfm_df['Recency'].max())],
                'frequency_range': [float(rfm_df['Frequency'].min()), float(rfm_df['Frequency'].max())]
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Error during RFM analysis: {str(e)}"}), 500

@app.route('/rfm_clustering/<session_id>/<int:k>', methods=['GET', 'OPTIONS'])
def rfm_clustering(session_id, k):
    """Perform K-means clustering on RFM data with specified k"""
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
        
        # Check if RFM data exists
        rfm_key = f"{session_id}_rfm"
        if rfm_key not in processed_datasets:
            return jsonify({"error": "RFM analysis not performed. Please run elbow analysis first."}), 400
        
        # Validate k value
        if k < 2 or k > 10:
            return jsonify({"error": "K must be between 2 and 10"}), 400
        
        rfm_df = processed_datasets[rfm_key]
        
        # Perform K-means clustering
        rfm_clustered, cluster_means, customer_class, pie_fig = perform_kmeans_clustering(rfm_df, k)
        
        # Convert matplotlib figure to base64 string
        img_buffer = io.BytesIO()
        pie_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(pie_fig)  # Close the figure to free memory
        
        # Update stored RFM data with clusters
        processed_datasets[rfm_key] = rfm_clustered
        
        return jsonify({
            'pie_chart': img_base64,
            'cluster_means': cluster_means.to_dict('records'),
            'customer_class': customer_class.to_dict('records'),
            'clustering_summary': {
                'num_clusters': k,
                'total_customers': len(rfm_clustered),
                'cluster_distribution': rfm_clustered['Cluster'].value_counts().sort_index().to_dict()
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Error during clustering: {str(e)}"}), 500

# Add these routes to your Flask application

@app.route('/predict_future/<int:hours>', methods=['POST', 'OPTIONS'])
def predict_future_endpoint(hours):
    """
    Predict future hours using existing Prophet model
    
    Expected JSON payload:
    {
        "model_path": "path/to/prophet_model.pkl",
        "behavior_type": "buy"  # optional, defaults to 'buy'
    }
    """
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        # Validate hours parameter
        if hours < 1 or hours > 8760:  # Max 1 year
            return jsonify({"error": "Hours must be between 1 and 8760 (1 year)"}), 400
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON payload required"}), 400
        
        # Extract parameters
        model_path = "Trained_Model/prophet_model.pkl"  # Default model path
        behavior_type = "Buy"
        
        print(f"Predicting future {hours} hours for behavior: {behavior_type}")
        
        # Call the predict_future_hours function
        result = predict_future_hours(
            model_path=model_path,
            hours=hours,
            behavior_type=behavior_type
        )
        
        # Convert predictions DataFrame to JSON-serializable format
        predictions_json = result['predictions'].to_dict('records')
        
        # Convert datetime objects to strings for JSON serialization
        for pred in predictions_json:
            if 'ds' in pred and hasattr(pred['ds'], 'isoformat'):
                pred['ds'] = pred['ds'].isoformat()
        
        return jsonify({
            'success': True,
            'hours_predicted': hours,
            'behavior_type': behavior_type,
            'predictions': predictions_json,
            'forecast_plot': result['forecast_plot'],
            'components_plot': result['components_plot'],
            'total_predictions': len(predictions_json)
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404
    except Exception as e:
        print(f"Error in predict_future_endpoint: {str(e)}")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


@app.route('/update_and_predict/<session_id>/<int:hours>', methods=['POST', 'OPTIONS'])
def update_and_predict_endpoint(session_id, hours):

    if session_id not in processed_datasets:
            return jsonify({"error": "Session not found or expired"}), 404
    
    if request.method == 'OPTIONS':
        return handle_preflight()
    
    try:
        # Validate hours parameter
        if hours < 1 or hours > 8760:  # Max 1 year
            return jsonify({"error": "Hours must be between 1 and 8760 (1 year)"}), 400
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON payload required"}), 400
        
        # Extract parameters
        model_path = "Trained_Model/prophet_model.pkl"  # Default model path
        behavior_type = "Buy"
        
        # Call the update_and_predict function
        result = update_and_predict(
            model_path=model_path,
            df = processed_datasets[session_id],
            hours=hours,
            behavior_type=behavior_type
        )
        
        # Convert predictions DataFrame to JSON-serializable format
        predictions_json = result['predictions'].to_dict('records')
        
        # Convert datetime objects to strings for JSON serialization
        for pred in predictions_json:
            if 'ds' in pred and hasattr(pred['ds'], 'isoformat'):
                pred['ds'] = pred['ds'].isoformat()
        
        return jsonify({
            'success': True,
            'hours_predicted': hours,
            'behavior_type': behavior_type,
            'model_updated': True,
            'predictions': predictions_json,
            'forecast_plot': result['forecast_plot'],
            'components_plot': result['components_plot'],
            'total_predictions': len(predictions_json)
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404
    except Exception as e:
        print(f"Error in update_and_predict_endpoint: {str(e)}")
        return jsonify({"error": f"Error during model update and prediction: {str(e)}"}), 500

@app.route('/predict_next_user_behavior', methods=['POST', 'OPTIONS'])
def predict_user_behavior():
    if request.method == 'OPTIONS':
        return handle_preflight()
    try:
        # Get JSON data
        data = request.get_data(as_text=True)
        print("Received raw data:", data)
        python_list = json.loads(data)
        # Extract parameters

        print("alo alo: ", python_list)
        
        if not python_list:
            return jsonify({"error": "Sequence is required for prediction"}), 400
        
        result = predict_next(python_list)
        
        processed_probabilities = []
        for event_name, probability_np_float32 in result:
            processed_probabilities.append({
                "event": event_name,
                "probability": float(probability_np_float32) # Convert np.float32 to Python float
            })
        print("Processed probabilities:", processed_probabilities)
        return jsonify({
            'success': True,
            'predictions': processed_probabilities
        })
    except Exception as e:
        print(f"Error in predict_next_user_behavior endpoint: {str(e)}")
        return jsonify({"error": f"Error during model update and prediction: {str(e)}"}), 500

@app.route('/recommend_products', methods=['POST', 'OPTIONS'])
def recommend_products_route():
    """
    Call the recommend_products function.
    Expects JSON payload with necessary parameters for recommend_products.
    """
    try:
        return jsonify({
            'success': True,
            'recommended_products': [int(x) for x in recommend_products("12345")]
        })
    except Exception as e:
        print(f"Error in recommend_products_route: {str(e)}")
        return jsonify({"error": f"Error during recommendation: {str(e)}"}), 500    


@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition,X-Session-ID,X-Total-Pages,X-Current-Page,X-Total-Records'
    return response

if __name__ == "__main__":
    # Run the Flask app normally
    print("Starting Flask server...")
    print("Server will be available at: http://localhost:5000")
    print("\nAvailable pagination endpoints:")
    print("- GET /first_page/<session_id>")
    print("- GET /last_page/<session_id>")
    print("- GET /next_page/<session_id>/<current_page>")
    print("- GET /previous_page/<session_id>/<current_page>")
    print("- GET /go_to_page/<session_id>/<page_number>")
    print("- GET /pagination_info/<session_id>")
    app.run(debug=True, host='0.0.0.0', port=5000)